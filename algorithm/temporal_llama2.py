"""
Modified Llama 2 forwards for temporal head caching.

Replaces attention forward with a version that:
  - Pilot layers (0..pilot-1): eager attention → captures weights for spatial pruning
  - Deep layers (pilot..N-1): sdpa + spatial masking + per-head cache read/write
  - On keyframes: full recompute, populate cache
  - Between keyframes: T/B heads → cached, V heads → conditional, M heads → fresh

Compatible with OpenVLA and CogACT (both Llama 2 7B backbone).
"""

import math
from types import MethodType

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)

import transformers
from temporal_cache import TemporalHeadCache

# Triton kernel — optional; falls back to SDPA if unavailable
try:
    from triton_attention import temporal_sparse_attention
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

# transformers >=4.46 changed LlamaDecoderLayer to unpack 2 values from self_attn
# instead of 3 (removed past_key_value from return).
_TF_RETURNS_2 = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 46)


# ─── Modified attention forward ──────────────────────────────────────────────

def temporal_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    """Llama attention with temporal head caching + spatial pruning."""
    cache: TemporalHeadCache = self.temporal_cache
    layer_idx = self.layer_idx
    cfg = cache.cfg
    bsz, q_len, _ = hidden_states.size()

    # ── Compat: transformers <4.46 uses self.num_heads, >=4.46 uses config
    num_heads = getattr(self, "num_heads", None) or self.config.num_attention_heads
    num_kv_heads = getattr(self, "num_key_value_heads", None) or self.config.num_key_value_heads

    # ── Fast path: skip QKV entirely when ALL heads are cached ────────────
    # Saves 3 large matmuls (QKV) + full attention + 275/276 of o_proj.
    # Only o_proj on the last token (from cache) is computed; other positions
    # get zero (residual connection passes them through unchanged).
    # Note: use q_len > 1 (prefill check) instead of past_key_value is None,
    # because HF creates a DynamicCache even for fresh prefill passes.
    if (_HAS_TRITON
            and cache.enabled and not cache.is_keyframe
            and layer_idx >= cfg.pilot_layers
            and cache.spatial_keep_bool is not None
            and cache.cache_valid[layer_idx]
            and q_len > 1):
        # Ensure precomputed tensors exist (lazy init on first deep layer)
        if cache._frame_active_head_ids is None:
            cache.prepare_frame_tensors(q_len)
        if (len(cache._frame_active_head_ids[layer_idx]) == 0
                and cache._frame_o_proj_deltas
                and layer_idx in cache._frame_o_proj_deltas):
            # All heads cached → use precomputed o_proj delta
            o_out = cache._frame_o_proj_deltas[layer_idx].to(hidden_states.dtype)
            attn_out = hidden_states.new_zeros(bsz, q_len, o_out.shape[-1])
            attn_out[:, -1:, :] = o_out
            # No store_heads — cached values are unchanged
            if _TF_RETURNS_2:
                return attn_out, None
            return attn_out, None, past_key_value

    # ── QKV projections ───────────────────────────────────────────────────
    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    query = query.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
    key = key.view(bsz, q_len, num_kv_heads, self.head_dim).transpose(1, 2)
    value = value.view(bsz, q_len, num_kv_heads, self.head_dim).transpose(1, 2)

    # ── RoPE ─────────────────────────────────────────────────────────────
    if position_embeddings is not None:
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

    # ── KV cache (for autoregressive decode after prefill) ───────────────
    if past_key_value is not None:
        ck = {"cache_position": cache_position} if cache_position is not None else {}
        key, value = past_key_value.update(key, value, layer_idx, ck)

    # ── GQA expansion ────────────────────────────────────────────────────
    n_groups = num_heads // num_kv_heads
    key_exp = repeat_kv(key, n_groups)
    val_exp = repeat_kv(value, n_groups)

    # ── Causal mask ──────────────────────────────────────────────────────
    causal = attention_mask
    if causal is not None and causal.dim() == 4:
        causal = causal[:, :, :, :key_exp.shape[-2]]

    # ── Check if Triton path will be used (deep layers, non-keyframes) ──
    _use_triton = (
        _HAS_TRITON
        and cache.enabled
        and not cache.is_keyframe
        and layer_idx >= cfg.pilot_layers
        and cache.spatial_keep_bool is not None
        and cache.cache_valid[layer_idx]
    )

    # ── Spatial pruning mask (only for SDPA fallback — Triton uses gathered KV)
    if (not _use_triton
            and cache.enabled and not cache.is_keyframe
            and layer_idx >= cfg.pilot_layers
            and cache.spatial_keep_bool is not None):
        sp_mod = cache.get_spatial_attn_mod(key_exp.shape[-2])
        if sp_mod is not None and causal is not None:
            causal = causal + sp_mod.to(causal.dtype)

    # ── Attention ────────────────────────────────────────────────────────
    attn_weights_out = None

    if layer_idx < cfg.pilot_layers:
        # PILOT: eager path — we need the weight matrix for spatial pruning
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(query, key_exp.transpose(-2, -1)) * scale
        if causal is not None:
            scores = scores + causal
        attn_weights_out = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn_out = torch.matmul(attn_weights_out.to(query.dtype), val_exp)

        # Feed pilot attention into cache for spatial mask
        cache.accumulate_pilot_attn(layer_idx, attn_weights_out.detach())

        # After last pilot layer → build spatial mask for deeper layers
        if layer_idx == cfg.pilot_layers - 1 and not cache.is_keyframe:
            cache.compute_spatial_mask()
    else:
        # DEEP: selective sparse attention (Triton) or fallback SDPA
        if _use_triton:
            # Fused path: kernel only runs active heads on gathered KV,
            # cached heads filled directly — no post-SDPA swap needed
            attn_out = temporal_sparse_attention(
                query, key_exp, val_exp, causal,
                cache, layer_idx)
        else:
            # Fallback: full SDPA (keyframes, first frame, no Triton)
            attn_out = F.scaled_dot_product_attention(
                query, key_exp, val_exp, attn_mask=causal, dropout_p=0.0)

            # Head caching swap for SDPA fallback path (non-keyframes only)
            if (cache.enabled and not cache.is_keyframe
                    and cache.cache_valid[layer_idx]):
                recomp = cache.get_recompute_mask(layer_idx)      # [H] bool
                cached_idx = (~recomp).nonzero(as_tuple=True)[0]
                if len(cached_idx) > 0:
                    old = cache.get_cached_heads(layer_idx)        # [H, D]
                    attn_out[:, cached_idx, -1, :] = old[cached_idx].to(attn_out.dtype)

    # Always store current last-token head outputs for future frames
    cache.store_heads(layer_idx, attn_out[:, :, -1, :])

    # ── Merge heads → o_proj ─────────────────────────────────────────────
    attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_out = self.o_proj(attn_out)

    if _TF_RETURNS_2:
        return attn_out, attn_weights_out
    return attn_out, attn_weights_out, past_key_value


# ─── Deep layer chain ────────────────────────────────────────────────────────

def _run_deep_chain(hidden_states: torch.Tensor, cache: TemporalHeadCache):
    """Process ALL fully-cached deep layers in a single function call.

    Eliminates 29 out of 30 Python function dispatches through the layer loop.
    Each layer does: o_delta add + last-token-only FFN (or full FFN for chain tail).

    Called by the first deep layer when _all_deep_cached is True.
    """
    cfg = cache.cfg
    layers = cache._decoder_layers

    for l in range(cfg.pilot_layers, cfg.num_layers):
        layer = layers[l]
        o_delta = cache._frame_o_proj_deltas[l].to(hidden_states.dtype)
        hidden_states[:, -1:, :] = hidden_states[:, -1:, :] + o_delta

        if cache._frame_ffn_last_token_only[l]:
            last = hidden_states[:, -1:, :].clone()
            last = layer.post_attention_layernorm(last)
            last = layer.mlp(last)
            hidden_states[:, -1:, :] = hidden_states[:, -1:, :] + last
        else:
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

    return hidden_states


# ─── torch.compile'd deep chain ─────────────────────────────────────────────

_compiled_chain_fn = None   # lazily created on first use


def _make_compiled_chain(layers, pilot_layers):
    """Create a torch.compile'd function for the all-LT-FFN deep chain.

    Captures layer modules (layernorm + MLP) at compile time so the compiled
    graph has zero Python object access in the hot path.
    """
    deep_layers = layers[pilot_layers:]
    norms = [l.post_attention_layernorm for l in deep_layers]
    mlps = [l.mlp for l in deep_layers]
    n = len(deep_layers)

    def _chain_fn(hidden_states, o_deltas):
        """Pure tensor chain: o_delta + LT-FFN for each deep layer."""
        for i in range(n):
            hidden_states[:, -1:, :] = hidden_states[:, -1:, :] + o_deltas[i]
            last = hidden_states[:, -1:, :].clone()
            last = norms[i](last)
            last = mlps[i](last)
            hidden_states[:, -1:, :] = hidden_states[:, -1:, :] + last
        return hidden_states

    return torch.compile(_chain_fn, mode="reduce-overhead", fullgraph=False)


def _run_deep_chain_compiled(hidden_states: torch.Tensor, cache: TemporalHeadCache):
    """Compiled version of _run_deep_chain — uses torch.compile + CUDA graphs."""
    global _compiled_chain_fn
    if _compiled_chain_fn is None:
        _compiled_chain_fn = _make_compiled_chain(
            cache._decoder_layers, cache.cfg.pilot_layers)

    # Stack o_deltas into a single tensor [N, 1, 1, hidden_dim]
    cfg = cache.cfg
    o_list = [cache._frame_o_proj_deltas[l].to(hidden_states.dtype)
              for l in range(cfg.pilot_layers, cfg.num_layers)]
    o_deltas = torch.stack(o_list, dim=0)

    return _compiled_chain_fn(hidden_states, o_deltas)


# ─── Modified decoder layer forward ──────────────────────────────────────────

def temporal_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    """Decoder layer that skips the entire attention block for fully-cached layers.

    Three execution paths (fastest to slowest):
      1. Chain passthrough: first deep layer processed us → return hidden_states unchanged
      2. Chain trigger: first deep layer, all deep cached → batch all 30 layers
      3. Single-layer fast path: this layer fully cached → o_delta + LT-FFN
      4. Normal path: delegate to original forward (pilot layers, keyframes, etc.)
    """
    attn_mod = self.self_attn
    cache: TemporalHeadCache = attn_mod.temporal_cache
    layer_idx = attn_mod.layer_idx
    cfg = cache.cfg

    # ── Path 1: Ultra-fast passthrough — chain already processed us ───
    if cache._deep_chain_done:
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs

    # ── Check if we can skip the entire attention block ───────────────
    # Use hidden_states.shape[1] > 1 (prefill) instead of past_key_value is None,
    # because HF creates DynamicCache even on fresh prefill (not truly "past" KV).
    _skip_attn = (
        _HAS_TRITON
        and cache.enabled and not cache.is_keyframe
        and layer_idx >= cfg.pilot_layers
        and cache.spatial_keep_bool is not None
        and cache.cache_valid[layer_idx]
        and hidden_states.shape[1] > 1
    )

    if _skip_attn:
        # Lazy init precomputed tensors
        if cache._frame_active_head_ids is None:
            cache.prepare_frame_tensors(hidden_states.shape[1])

        # ── Path 2: Chain trigger — batch all deep layers ─────────────
        if (layer_idx == cfg.pilot_layers
                and cache._all_deep_cached
                and cache._frame_o_proj_deltas
                and cache._frame_ffn_last_token_only is not None):
            if getattr(cache, '_use_compile', False):
                hidden_states = _run_deep_chain_compiled(hidden_states, cache)
            else:
                hidden_states = _run_deep_chain(hidden_states, cache)
            cache._deep_chain_done = True
            outputs = (hidden_states,)
            if output_attentions:
                outputs += (None,)
            if use_cache:
                outputs += (past_key_value,)
            return outputs

        # ── Path 3: Single-layer fast path (partial caching) ──────────
        if len(cache._frame_active_head_ids[layer_idx]) == 0:
            o_delta = cache._frame_o_proj_deltas[layer_idx].to(hidden_states.dtype)
            hidden_states[:, -1:, :] += o_delta

            if (cache._frame_ffn_last_token_only is not None
                    and cache._frame_ffn_last_token_only[layer_idx]):
                last = hidden_states[:, -1:, :].clone()
                last_ln = self.post_attention_layernorm(last)
                last_ffn = self.mlp(last_ln)
                hidden_states[:, -1:, :] = last + last_ffn
            else:
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = self.mlp(hidden_states)
                hidden_states = residual + hidden_states

            outputs = (hidden_states,)
            if output_attentions:
                outputs += (None,)
            if use_cache:
                outputs += (past_key_value,)
            return outputs

    # ── Path 4: Normal path — delegate to original forward ────────────
    return self._orig_forward(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )


# ─── Wiring ──────────────────────────────────────────────────────────────────

def apply_temporal_caching(model, adapter, cache: TemporalHeadCache,
                           use_compile: bool = False):
    """Replace attention forwards in a Llama-2-based VLA with temporal versions.

    Works for any model whose adapter.get_llm_layers() returns LlamaDecoderLayers
    and adapter.get_attn_module() returns LlamaSdpaAttention.

    Args:
        use_compile: If True, torch.compile the deep layer chain for CUDA graph
                     fusion. Adds ~10s warmup on first non-keyframe but eliminates
                     kernel launch overhead in subsequent frames.
    """
    layers = adapter.get_llm_layers(model)
    attn_modules = []
    for layer in layers:
        attn = adapter.get_attn_module(layer)
        attn.temporal_cache = cache
        attn.forward = MethodType(temporal_attention_forward, attn)
        attn_modules.append(attn)
        # Patch decoder layer to skip entire attention block for fully-cached layers
        layer._orig_forward = layer.forward
        layer.forward = MethodType(temporal_decoder_layer_forward, layer)
    # Store refs so prepare_frame_tensors can precompute o_proj deltas + deep chain
    cache._attn_modules = attn_modules
    cache._decoder_layers = list(layers)
    cache._use_compile = use_compile
    print(f"  Temporal caching applied to {len(layers)} attention modules"
          f" (compile={'ON' if use_compile else 'OFF'})")


def apply_visual_cache(model, cache: TemporalHeadCache):
    """Cache visual embeddings on keyframes, skip ViT+projector on non-keyframes.

    On non-keyframes the image barely changes — re-encoding through ViT is
    redundant since temporal head caching already handles image changes via
    V-head classification. Saves ~11ms/frame (the entire visual encoder cost).
    """
    orig_vis = model.vision_backbone.forward
    orig_proj = model.projector.forward

    def cached_vis(pixel_values):
        if cache.is_keyframe or cache._cached_visual_embeds is None:
            return orig_vis(pixel_values)
        # Non-keyframe: skip ViT entirely (projector also returns cached)
        return pixel_values.new_empty(0)

    def cached_proj(patch_features):
        if cache.is_keyframe or cache._cached_visual_embeds is None:
            result = orig_proj(patch_features)
            cache._cached_visual_embeds = result
            return result
        return cache._cached_visual_embeds

    model.vision_backbone.forward = cached_vis
    model.projector.forward = cached_proj
    print("  Visual embedding cache applied (skip ViT on non-keyframes)")


def remove_temporal_caching(model, adapter):
    """Restore original attention forwards (for baseline comparison)."""
    layers = adapter.get_llm_layers(model)
    for layer in layers:
        attn = adapter.get_attn_module(layer)
        if hasattr(attn, "_original_forward"):
            attn.forward = attn._original_forward
            del attn._original_forward
        if hasattr(attn, "temporal_cache"):
            del attn.temporal_cache
        # Restore decoder layer forward
        if hasattr(layer, "_orig_forward"):
            layer.forward = layer._orig_forward
            del layer._orig_forward
