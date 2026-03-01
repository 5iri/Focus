"""
Temporal Head Cache — core engine for three-axis VLA inference reduction.

Axis 1: Spatial token pruning — top-K visual tokens from pilot layers
Axis 2: Head-level temporal caching — T/B always cache, V conditional, M recompute
Axis 3: Frame differencing — patch-level change drives V-head cache decisions

This module is model-agnostic. Model-specific forwards live in temporal_llama2.py.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class CacheConfig:
    """Architecture dims + algorithm hyperparameters."""
    # Architecture (filled from adapter.get_config())
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    head_dim: int = 128
    hidden_dim: int = 4096
    num_vis_tokens: int = 256
    vis_grid: Tuple[int, int] = (16, 16)

    # Token layout (BOS=0, vis tokens, then text)
    vis_start: int = 1               # first visual token position
    vis_end: int = 257               # one past last visual token (= num_vis_tokens + 1)

    # --- Algorithm knobs ---
    spatial_K: int = 32              # visual tokens kept after pilot
    cache_threshold: float = 0.1     # V-head attended-change threshold
    keyframe_interval: int = 5       # full recompute every K frames
    pilot_layers: int = 2            # layers 0..(pilot-1) run dense
    enable_last_token_ffn: bool = True  # skip middle-token FFN in safe chains

    # Head classification thresholds (from profiling)
    # Relative: classify by dominant region. min_dominant = minimum mass
    # for the dominant region to count (otherwise → M).
    min_dominant: float = 0.15       # primary region must have ≥15% mass
    bos_thresh: float = 0.10         # BOS is one token; 10% is very high

    def __post_init__(self):
        # Keep vis_end consistent with num_vis_tokens if using defaults
        if self.vis_end == 257 and self.num_vis_tokens != 256:
            self.vis_end = self.vis_start + self.num_vis_tokens


# ─── Frame differencing ─────────────────────────────────────────────────────

def compute_frame_diff(img_a: Image.Image, img_b: Image.Image,
                       grid_h: int = 16, grid_w: int = 16) -> torch.Tensor:
    """Per-patch normalised L1 difference.  Returns [grid_h, grid_w] in [0,1]."""
    import torchvision.transforms as T
    to_t = T.ToTensor()
    sz = grid_w * 14                 # match ViT 14×14 patch stride
    a = to_t(img_a.resize((sz, sz)))
    b = to_t(img_b.resize((sz, sz)))
    diff = (a - b).abs()
    C, H, W = diff.shape
    ph, pw = H // grid_h, W // grid_w
    diff = diff.reshape(C, grid_h, ph, grid_w, pw)
    diff = diff.permute(1, 3, 0, 2, 4).reshape(grid_h, grid_w, -1)
    patch = diff.mean(-1)
    mx = patch.max()
    return (patch / mx) if mx > 0 else patch


# ─── Head classification ─────────────────────────────────────────────────────

def classify_heads_from_attn(attn_by_layer: dict, cfg: CacheConfig):
    """Classify every head as V / T / B / M from one prefill pass.

    For caching, what matters is whether a head primarily attends to content
    that *changes* between frames (visual tokens → V) or content that stays
    *stable* (text + BOS → T/B).  BOS acts as an attention sink in most heads,
    so we fold BOS attention into the "stable" category.

    Classification:
        V — visual attention mass > min_dominant (head output depends on image)
        T — text+BOS dominates and text > BOS (stable; always cache)
        B — text+BOS dominates and BOS > text (stable; always cache)
        M — neither region reaches min_dominant (truly mixed)

    Args:
        attn_by_layer: {layer_idx: Tensor[B, H, Q, KV]}  full attention weights
        cfg: CacheConfig with thresholds

    Returns:
        numpy array [num_layers, num_heads] of str
    """
    vis_start = 1
    vis_end = cfg.num_vis_tokens + 1
    text_start = vis_end

    types = np.full((cfg.num_layers, cfg.num_heads), "M", dtype=object)
    for l in range(cfg.num_layers):
        if l not in attn_by_layer:
            continue
        # last-token row, per head
        a = attn_by_layer[l].float().squeeze(0)[:, -1, :]   # [H, KV]
        for h in range(min(cfg.num_heads, a.shape[0])):
            v = a[h, vis_start:vis_end].sum().item()
            t = a[h, text_start:].sum().item()
            b = a[h, 0].item()

            if v >= cfg.min_dominant and v > (t + b):
                # Visual-dominant: output depends on image
                types[l, h] = "V"
            elif (t + b) >= cfg.min_dominant:
                # Stable content dominant: safe to cache
                types[l, h] = "T" if t >= b else "B"
            # else stays M
    return types


# ─── Temporal Head Cache ─────────────────────────────────────────────────────

class TemporalHeadCache:
    """Manages all caching state across consecutive camera frames.

    Lifecycle per frame:
        cache.begin_frame(image)
        ... model forward (pilot layers write pilot_vis_attn,
            compute_spatial_mask() called after last pilot,
            deeper layers check should_recompute / read+write head cache) ...
        cache.collect_stats()
    """

    def __init__(self, cfg: CacheConfig, device="cpu", dtype=torch.bfloat16):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.enabled = True

        # Per-head last-token attention output (pre-o_proj)
        # shape: [num_layers, num_heads, head_dim]
        self.head_cache = torch.zeros(
            cfg.num_layers, cfg.num_heads, cfg.head_dim,
            device=device, dtype=dtype)
        self.cache_valid = torch.zeros(
            cfg.num_layers, dtype=torch.bool, device=device)

        # Head types — set via set_head_types() after profiling
        self.head_types: Optional[np.ndarray] = None  # [L, H] of str

        # Frame state
        self.frame_idx = -1
        self.prev_image: Optional[Image.Image] = None
        self.patch_change: Optional[torch.Tensor] = None
        self.is_keyframe = True

        # Pilot-layer accumulator for spatial pruning
        self.pilot_vis_attn: Optional[torch.Tensor] = None
        self.spatial_mask: Optional[torch.Tensor] = None       # kept vis indices
        self.spatial_keep_bool: Optional[torch.Tensor] = None  # [num_vis] bool

        # Precomputed per-frame tensors (built once after compute_spatial_mask)
        self._frame_active_token_ids: Optional[torch.Tensor] = None  # [S_active] int32
        self._frame_recompute_masks: Optional[torch.Tensor] = None   # [L, H] bool
        self._frame_active_head_ids: Optional[list] = None  # list of [n_active] int32 per layer
        self._frame_cached_head_ids: Optional[list] = None   # list of [n_cached] int64 per layer
        self._frame_o_proj_deltas: Optional[dict] = None     # {layer_idx: [1, 1, hidden_dim]}
        self._frame_ffn_last_token_only: Optional[list] = None  # [L] bool — safe to skip middle-token FFN
        self._all_deep_cached: bool = False   # True when ALL deep layers are fully cached
        self._deep_chain_done: bool = False   # Set by first deep layer after chain execution
        self._cached_visual_embeds: Optional[torch.Tensor] = None  # projected_patch_embeddings

        # Model layer references (set by apply_temporal_caching)
        self._attn_modules: Optional[list] = None
        self._decoder_layers: Optional[list] = None

        # Stats
        self.frame_stats: list = []

    # ── Head types ───────────────────────────────────────────────────────

    def set_head_types(self, types: np.ndarray):
        self.head_types = types
        for ht in ("V", "T", "B", "M"):
            n = (types == ht).sum()
            print(f"  {ht}-heads: {n} / {types.size}")

    # ── Frame lifecycle ──────────────────────────────────────────────────

    def begin_frame(self, image: Image.Image) -> bool:
        """Call before each frame's forward pass.  Returns is_keyframe."""
        self.frame_idx += 1

        if not self.enabled:
            self.is_keyframe = True
            return True

        self.is_keyframe = (self.frame_idx % self.cfg.keyframe_interval == 0)

        if self.prev_image is not None and not self.is_keyframe:
            self.patch_change = compute_frame_diff(
                self.prev_image, image,
                *self.cfg.vis_grid).to(self.device)
        else:
            self.patch_change = torch.ones(
                self.cfg.vis_grid, device=self.device)
            if self.frame_idx == 0:
                self.is_keyframe = True

        self.prev_image = image
        self.pilot_vis_attn = torch.zeros(
            self.cfg.num_vis_tokens, device=self.device, dtype=torch.float32)
        self.spatial_mask = None
        self.spatial_keep_bool = None
        self._frame_active_token_ids = None
        self._frame_recompute_masks = None
        self._frame_active_head_ids = None
        self._frame_cached_head_ids = None
        self._frame_o_proj_deltas = None
        self._frame_ffn_last_token_only = None
        self._all_deep_cached = False
        self._deep_chain_done = False
        return self.is_keyframe

    # ── Pilot-layer spatial analysis ─────────────────────────────────────

    def accumulate_pilot_attn(self, layer_idx: int,
                              attn_w: torch.Tensor):
        """Add visual-attention mass from a pilot layer.

        attn_w: [B, H, Q, KV] float attention weights.
        """
        if layer_idx >= self.cfg.pilot_layers:
            return
        vs = 1
        ve = self.cfg.num_vis_tokens + 1
        last_vis = attn_w[:, :, -1, vs:ve]            # [B, H, nv]
        self.pilot_vis_attn += last_vis.float().mean(dim=(0, 1)).to(self.device)

    def compute_spatial_mask(self):
        """Pick top-K visual tokens from accumulated pilot attention."""
        K = self.cfg.spatial_K
        nv = self.cfg.num_vis_tokens
        if K >= nv:
            self.spatial_mask = torch.arange(nv, device=self.device)
        else:
            _, topk = self.pilot_vis_attn.topk(K)
            self.spatial_mask = topk.sort().values

        self.spatial_keep_bool = torch.zeros(
            nv, dtype=torch.bool, device=self.device)
        self.spatial_keep_bool[self.spatial_mask] = True
        return self.spatial_mask

    def prepare_frame_tensors(self, seq_len: int):
        """Precompute all per-layer index tensors once per frame.

        Called once after compute_spatial_mask() from the first deep layer.
        Eliminates ~1ms/layer of Python overhead by doing all index building
        and recompute-mask computation in one shot.
        """
        cfg = self.cfg
        vs, ve = cfg.vis_start, cfg.vis_end

        # ── active_token_ids: same for all deep layers ────────────────
        bos = torch.tensor([0], device=self.device, dtype=torch.int32)
        kept_vis = self.spatial_keep_bool.nonzero(as_tuple=True)[0].to(torch.int32) + vs
        text = torch.arange(ve, seq_len, device=self.device, dtype=torch.int32)
        self._frame_active_token_ids = torch.cat([bos, kept_vis, text]).contiguous()

        # ── Vectorized recompute masks for all layers ─────────────────
        L, H = cfg.num_layers, cfg.num_heads
        masks = torch.ones(L, H, dtype=torch.bool, device=self.device)

        if self.head_types is not None and not self.is_keyframe:
            pc_mean = (self.patch_change.mean().item()
                       if self.patch_change is not None else 1.0)
            v_recompute = pc_mean >= cfg.cache_threshold

            # Vectorized: build masks from numpy head_types
            ht = self.head_types  # [L, H] numpy of str
            is_tb = torch.tensor(
                (ht == "T") | (ht == "B"), dtype=torch.bool, device=self.device)
            is_v = torch.tensor(
                ht == "V", dtype=torch.bool, device=self.device)

            # T/B → cache (False), V → conditional, M → recompute (True)
            masks[is_tb] = False
            if not v_recompute:
                masks[is_v] = False

            # Pilot layers and invalid cache layers → all recompute
            masks[:cfg.pilot_layers, :] = True
            invalid = ~self.cache_valid  # [L] bool
            masks[invalid, :] = True

        self._frame_recompute_masks = masks

        # ── Per-layer active/cached head indices ──────────────────────
        self._frame_active_head_ids = []
        self._frame_cached_head_ids = []
        for l in range(L):
            recomp = masks[l]  # [H] bool
            self._frame_active_head_ids.append(
                recomp.nonzero(as_tuple=True)[0].to(torch.int32).contiguous())
            self._frame_cached_head_ids.append(
                (~recomp).nonzero(as_tuple=True)[0])

        # ── Precompute o_proj deltas for fully-cached layers ──────────
        # Eliminates per-layer o_proj kernel launch in the skip path.
        self._frame_o_proj_deltas = {}
        if self._attn_modules is not None:
            for l in range(cfg.pilot_layers, L):
                if len(self._frame_active_head_ids[l]) == 0:
                    attn_mod = self._attn_modules[l]
                    cached_vals = self.head_cache[l]  # [H, D]
                    hd = cfg.num_heads * cfg.head_dim
                    last_tok = cached_vals.reshape(1, 1, hd).to(self.dtype)
                    with torch.no_grad():
                        self._frame_o_proj_deltas[l] = attn_mod.o_proj(last_tok)

        # ── Compute "safe for last-token-only FFN" flags ──────────────
        # A fully-cached layer can skip middle-token FFN if ALL subsequent
        # layers are also fully-cached (no downstream consumer reads middle
        # positions). Computed via a single reverse scan.
        self._frame_ffn_last_token_only = [False] * L
        if cfg.enable_last_token_ffn:
            safe_from_here = True
            for l in range(L - 1, -1, -1):
                if l < cfg.pilot_layers:
                    safe_from_here = False
                elif len(self._frame_active_head_ids[l]) == 0:
                    self._frame_ffn_last_token_only[l] = safe_from_here
                else:
                    safe_from_here = False

        # ── Check if ALL deep layers are fully cached ─────────────────
        self._all_deep_cached = all(
            len(self._frame_active_head_ids[l]) == 0
            for l in range(cfg.pilot_layers, L)
        )

    def get_spatial_attn_mod(self, kv_len: int) -> Optional[torch.Tensor]:
        """4-D additive mask to zero-out pruned visual keys.

        Returns [1, 1, 1, kv_len] (broadcasts over B, H, Q).
        Pruned positions get -inf, kept positions get 0.
        Returns None on keyframes or before spatial mask is ready.
        """
        if self.spatial_keep_bool is None or self.is_keyframe:
            return None
        vs = 1
        ve = min(self.cfg.num_vis_tokens + 1, kv_len)
        mod = torch.zeros(1, 1, 1, kv_len, device=self.device, dtype=self.dtype)
        # Vectorized: set pruned visual positions to -inf
        pruned = ~self.spatial_keep_bool[: ve - vs]
        mod[0, 0, 0, vs:ve][pruned] = torch.finfo(self.dtype).min
        return mod

    # ── Per-head recompute decisions ─────────────────────────────────────

    def get_recompute_mask(self, layer_idx: int) -> torch.Tensor:
        """Bool tensor [num_heads]: True ⇒ recompute, False ⇒ use cache."""
        H = self.cfg.num_heads
        mask = torch.ones(H, dtype=torch.bool, device=self.device)

        if (not self.enabled or self.is_keyframe
                or layer_idx < self.cfg.pilot_layers
                or self.head_types is None
                or not self.cache_valid[layer_idx]):
            return mask                          # recompute everything

        pc_mean = (self.patch_change.mean().item()
                   if self.patch_change is not None else 1.0)

        for h in range(H):
            ht = self.head_types[layer_idx, h]
            if ht in ("T", "B"):
                mask[h] = False                   # always cache
            elif ht == "V":
                mask[h] = pc_mean >= self.cfg.cache_threshold  # conditional
            # M stays True (always recompute)
        return mask

    # ── Head output storage ──────────────────────────────────────────────

    def store_heads(self, layer_idx: int, last_tok_heads: torch.Tensor):
        """Store per-head outputs.  last_tok_heads: [B, H, D]."""
        self.head_cache[layer_idx] = last_tok_heads[0].detach()
        self.cache_valid[layer_idx] = True

    def get_cached_heads(self, layer_idx: int) -> torch.Tensor:
        """Retrieve [H, D] cached head outputs."""
        return self.head_cache[layer_idx]

    # ── Stats ────────────────────────────────────────────────────────────

    def collect_stats(self):
        if self.head_types is None:
            return
        total = self.cfg.num_layers * self.cfg.num_heads
        cached = 0
        for l in range(self.cfg.num_layers):
            cached += int((~self.get_recompute_mask(l)).sum())
        self.frame_stats.append({
            "frame": self.frame_idx,
            "keyframe": self.is_keyframe,
            "cached": cached,
            "total": total,
            "cache_rate": cached / total if total else 0,
            "spatial_K": (len(self.spatial_mask)
                          if self.spatial_mask is not None
                          else self.cfg.num_vis_tokens),
            "patch_change": (self.patch_change.mean().item()
                             if self.patch_change is not None else 0),
        })

    def theoretical_speedup(self) -> float:
        """FLOPs reduction across all processed frames."""
        C = self.cfg
        non_kf = [s for s in self.frame_stats if not s["keyframe"]]
        if not non_kf:
            return 1.0
        avg_cache = np.mean([s["cache_rate"] for s in non_kf])
        kf_frac = 1.0 - len(non_kf) / len(self.frame_stats)
        spatial = C.spatial_K / C.num_vis_tokens
        head = 1.0 - avg_cache
        pilot = C.pilot_layers / C.num_layers
        deep = 1.0 - pilot
        non_kf_ratio = pilot + deep * spatial * head
        ratio = kf_frac * 1.0 + (1 - kf_frac) * non_kf_ratio
        return 1.0 / ratio if ratio > 0 else float("inf")

    def reset(self):
        """Reset all frame state (keep head_types)."""
        self.frame_idx = -1
        self.prev_image = None
        self.patch_change = None
        self.is_keyframe = True
        self.pilot_vis_attn = None
        self.spatial_mask = None
        self.spatial_keep_bool = None
        self._frame_active_token_ids = None
        self._frame_recompute_masks = None
        self._frame_active_head_ids = None
        self._frame_cached_head_ids = None
        self._frame_o_proj_deltas = None
        self._frame_ffn_last_token_only = None
        self._all_deep_cached = False
        self._deep_chain_done = False
        self._cached_visual_embeds = None
        self.head_cache.zero_()
        self.cache_valid.zero_()
        self.frame_stats.clear()
