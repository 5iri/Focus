"""
Triton fused kernel for selective sparse attention.

Skips computation entirely for cached heads and pruned tokens, producing
measurable wall-clock speedup vs the Python prototype that computes all heads
then overwrites cached ones.

Architecture: Flash-attention-style online softmax over gathered KV tokens.
Grid: (Q_blocks, active_heads, B) — only launches programs for heads that
actually need recomputation.

Compatible with Llama 2 7B (OpenVLA / OpenVLA-OFT):
  32 layers, 32 heads, head_dim=128, seq~276, spatial_K=32
"""

import math

import torch
import triton
import triton.language as tl

from temporal_cache import TemporalHeadCache


# ─── Triton kernel ───────────────────────────────────────────────────────────

@triton.jit
def _selective_sparse_attn_fwd(
    Q, K, V, Out,
    active_head_ids,       # [num_active] int32 — which heads to compute
    active_token_ids,      # [S_active] int32 — which KV positions to gather
    sm_scale,              # 1/sqrt(head_dim)
    # Q strides
    stride_qb, stride_qh, stride_qm, stride_qd,
    # K strides
    stride_kb, stride_kh, stride_kn, stride_kd,
    # V strides
    stride_vb, stride_vh, stride_vn, stride_vd,
    # Out strides
    stride_ob, stride_oh, stride_om, stride_od,
    S_active,              # number of active KV tokens
    Q_LEN,                 # query sequence length
    BLOCK_M: tl.constexpr, # Q tile size
    BLOCK_N: tl.constexpr, # KV tile size
    BLOCK_D: tl.constexpr, # head dim (must be power of 2)
    IS_CAUSAL: tl.constexpr,
):
    """Flash-attention over gathered KV tokens for a subset of heads.

    Grid: (cdiv(Q_LEN, BLOCK_M), num_active_heads, B)

    For each (Q-block, active-head, batch):
      1. Load Q tile [BLOCK_M, D]
      2. Loop over gathered KV in BLOCK_N tiles:
         - Load K at active_token_ids positions → scores
         - Causal mask via position comparison
         - Online softmax update
         - Load V at same positions → accumulate
      3. Write output to Out[b, real_head, q_pos, :]
    """
    pid_m = tl.program_id(0)   # Q block index
    pid_ah = tl.program_id(1)  # active head index
    pid_b = tl.program_id(2)   # batch index

    # Map active head index → actual head index
    h = tl.load(active_head_ids + pid_ah)

    # ── Q tile offsets ────────────────────────────────────────────────
    q_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    q_mask = q_offs < Q_LEN
    d_offs = tl.arange(0, BLOCK_D)                     # [BLOCK_D]

    # Load Q block: [BLOCK_M, BLOCK_D]
    q_ptrs = (Q
              + pid_b * stride_qb
              + h * stride_qh
              + q_offs[:, None] * stride_qm
              + d_offs[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    # ── Online softmax accumulators ───────────────────────────────────
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # ── Loop over gathered KV tiles ───────────────────────────────────
    for start_n in range(0, S_active, BLOCK_N):
        n_offs = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        n_mask = n_offs < S_active

        # Gather KV token positions
        tok_ids = tl.load(active_token_ids + n_offs, mask=n_mask, other=0)

        # Load K at gathered positions: [BLOCK_N, BLOCK_D]
        k_ptrs = (K
                  + pid_b * stride_kb
                  + h * stride_kh
                  + tok_ids[:, None] * stride_kn
                  + d_offs[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        # Scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, tl.trans(k)) * sm_scale

        # ── Masking ───────────────────────────────────────────────
        if IS_CAUSAL:
            causal_ok = q_offs[:, None] >= tok_ids[None, :]
            scores = tl.where(causal_ok & n_mask[None, :], scores, float("-inf"))
        else:
            scores = tl.where(n_mask[None, :], scores, float("-inf"))

        # Also mask out-of-range Q rows
        scores = tl.where(q_mask[:, None], scores, float("-inf"))

        # ── Online softmax update ─────────────────────────────────
        m_ij = tl.max(scores, axis=1)               # [BLOCK_M]
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)                  # rescale old
        p = tl.exp(scores - m_new[:, None])           # new tile probs

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V at gathered positions: [BLOCK_N, BLOCK_D]
        v_ptrs = (V
                  + pid_b * stride_vb
                  + h * stride_vh
                  + tok_ids[:, None] * stride_vn
                  + d_offs[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # ── Normalize and store ───────────────────────────────────────────
    # Guard against l_i == 0 (all tokens masked — shouldn't happen in practice)
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_safe[:, None]

    out_ptrs = (Out
                + pid_b * stride_ob
                + h * stride_oh
                + q_offs[:, None] * stride_om
                + d_offs[None, :] * stride_od)
    tl.store(out_ptrs, acc, mask=q_mask[:, None])


# ─── Python wrapper ──────────────────────────────────────────────────────────

# Tile sizes — tuned for Llama 2 7B dimensions (seq~276, head_dim=128)
BLOCK_M = 64
BLOCK_N = 64
BLOCK_D = 128   # must match head_dim


def temporal_sparse_attention(
    query: torch.Tensor,        # [B, H, Q, D]
    key_exp: torch.Tensor,      # [B, H, KV, D]
    val_exp: torch.Tensor,      # [B, H, KV, D]
    causal_mask,                # unused by kernel (causal handled internally)
    cache: TemporalHeadCache,
    layer_idx: int,
) -> torch.Tensor:
    """Fused selective sparse attention — replaces SDPA + head-cache-swap.

    Only launches Triton programs for heads that need recomputation,
    and only gathers KV tokens that survived spatial pruning.
    Cached heads are filled directly from the cache without any compute.

    Uses precomputed per-frame tensors from cache.prepare_frame_tensors()
    to avoid per-layer Python overhead (index building, recompute mask, etc.).

    Returns: [B, H, Q, D] attention output.
    """
    B, H, Q_LEN, D = query.shape

    # ── Lazy init: precompute all per-frame tensors on first deep layer ─
    if cache._frame_active_token_ids is None:
        cache.prepare_frame_tensors(key_exp.shape[2])

    # ── Use precomputed tensors (no per-layer Python overhead) ────────
    active_head_ids = cache._frame_active_head_ids[layer_idx]
    cached_idx = cache._frame_cached_head_ids[layer_idx]
    active_token_ids = cache._frame_active_token_ids
    S_active = active_token_ids.shape[0]

    # ── Allocate output — zero-init ───────────────────────────────────
    out = torch.zeros_like(query)

    # ── Fill cached heads from cache ──────────────────────────────────
    if len(cached_idx) > 0:
        out[:, cached_idx, -1, :] = cache.head_cache[layer_idx, cached_idx].to(out.dtype)

    if len(active_head_ids) == 0:
        return out  # all heads cached — no kernel launch needed

    # ── Launch kernel ─────────────────────────────────────────────────
    sm_scale = 1.0 / math.sqrt(D)
    num_active = len(active_head_ids)

    # Ensure contiguous (usually already is after transpose)
    query = query.contiguous()
    key_exp = key_exp.contiguous()
    val_exp = val_exp.contiguous()

    grid = (triton.cdiv(Q_LEN, BLOCK_M), num_active, B)

    _selective_sparse_attn_fwd[grid](
        query, key_exp, val_exp, out,
        active_head_ids, active_token_ids,
        sm_scale,
        # Q strides
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        # K strides
        key_exp.stride(0), key_exp.stride(1), key_exp.stride(2), key_exp.stride(3),
        # V strides
        val_exp.stride(0), val_exp.stride(1), val_exp.stride(2), val_exp.stride(3),
        # Out strides
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        S_active,
        Q_LEN,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=True,
    )

    return out
