"""
Wall-clock overhead profiler for the temporal attention path.

Isolates and times each logical step that occurs in a single deep layer's
temporal_sparse_attention call:
  1. get_recompute_mask        (Python for-loop over 32 heads)
  2. Building active_token_ids (torch.cat, nonzero, arange)
  3. get_spatial_attn_mod      (allocates a new [1,1,1,KV] tensor each call)
  4. Triton kernel launch      (selective sparse attention)
  5. Cache read/write          (store_heads + get_cached_heads + fill)

Usage:
    cd /home/5iri/Focus/algorithm && python profile_overhead.py
"""

import math
import time

import numpy as np
import torch

from temporal_cache import CacheConfig, TemporalHeadCache

# Try importing the Triton kernel; skip that benchmark if unavailable
try:
    from triton_attention import (
        temporal_sparse_attention,
        _selective_sparse_attn_fwd,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
    )
    import triton
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    print("[WARN] Triton not available -- kernel benchmark will be skipped")


# ── Helpers ──────────────────────────────────────────────────────────────────

def timed_us(fn, n_iters=100, warmup=10):
    """Run `fn` with CUDA sync and return mean wall-clock microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # seconds -> microseconds
    return np.mean(times), np.std(times)


def main():
    device = "cuda"
    dtype = torch.bfloat16
    n_iters = 100

    # ── 1. Build cache with realistic settings ───────────────────────────
    cfg = CacheConfig(
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        head_dim=128,
        hidden_dim=4096,
        num_vis_tokens=256,
        vis_grid=(16, 16),
        vis_start=1,
        vis_end=257,
        spatial_K=32,
        pilot_layers=2,
    )
    cache = TemporalHeadCache(cfg, device=device, dtype=dtype)

    # ── 2. Fake head_types: mostly T/B with a few V and M ────────────────
    #   Mimic what the real classifier produces:
    #     ~40% T, ~30% B, ~20% V, ~10% M
    rng = np.random.RandomState(42)
    head_types = np.empty((cfg.num_layers, cfg.num_heads), dtype=object)
    for l in range(cfg.num_layers):
        for h in range(cfg.num_heads):
            r = rng.random()
            if r < 0.40:
                head_types[l, h] = "T"
            elif r < 0.70:
                head_types[l, h] = "B"
            elif r < 0.90:
                head_types[l, h] = "V"
            else:
                head_types[l, h] = "M"
    cache.set_head_types(head_types)

    # ── 3. Simulate non-keyframe state (caches valid, spatial mask ready) ─
    cache.frame_idx = 3
    cache.is_keyframe = False
    cache.enabled = True
    cache.patch_change = torch.full(cfg.vis_grid, 0.05, device=device)

    # Populate spatial mask (top-K=32 of 256 visual tokens)
    cache.pilot_vis_attn = torch.randn(cfg.num_vis_tokens, device=device)
    cache.compute_spatial_mask()

    # Mark all layers as having valid cached data
    cache.cache_valid[:] = True
    # Fill head_cache with random data so reads are realistic
    cache.head_cache.normal_()

    # Target layer: a deep layer well past pilot
    layer_idx = 16

    # ── 4. Build realistic tensors ────────────────────────────────────────
    B, H, Q, D = 1, 32, 276, 128
    KV = Q  # prefill: KV == Q
    query = torch.randn(B, H, Q, D, device=device, dtype=dtype)
    key_exp = torch.randn(B, H, KV, D, device=device, dtype=dtype)
    val_exp = torch.randn(B, H, KV, D, device=device, dtype=dtype)

    # ── Precompute some values the kernel wrapper would compute ───────────
    vs, ve = cfg.vis_start, cfg.vis_end
    seq_len = KV

    print(f"\n{'='*65}")
    print(f" Temporal Attention Overhead Profiler")
    print(f"   B={B}, H={H}, Q={Q}, D={D}, KV={KV}")
    print(f"   spatial_K={cfg.spatial_K}, num_vis_tokens={cfg.num_vis_tokens}")
    print(f"   layer_idx={layer_idx}, n_iters={n_iters}")
    print(f"{'='*65}\n")

    results = []

    # ── Benchmark 1: get_recompute_mask ──────────────────────────────────
    def bench_recompute_mask():
        cache.get_recompute_mask(layer_idx)

    mu, sd = timed_us(bench_recompute_mask, n_iters)
    results.append(("get_recompute_mask(layer_idx)", mu, sd))
    print(f"  [1] get_recompute_mask          {mu:8.1f} +/- {sd:6.1f} us")

    # ── Benchmark 2: Building active_token_ids ───────────────────────────
    def bench_active_token_ids():
        recomp_mask = cache.get_recompute_mask(layer_idx)
        _active_head_ids = recomp_mask.nonzero(as_tuple=True)[0].to(torch.int32).contiguous()
        bos = torch.tensor([0], device=device, dtype=torch.int32)
        kept_vis = cache.spatial_keep_bool.nonzero(as_tuple=True)[0].to(torch.int32) + vs
        text = torch.arange(ve, seq_len, device=device, dtype=torch.int32)
        _active_tok = torch.cat([bos, kept_vis, text]).contiguous()

    mu, sd = timed_us(bench_active_token_ids, n_iters)
    results.append(("build active_token_ids + head_ids", mu, sd))
    print(f"  [2] build active_token/head_ids {mu:8.1f} +/- {sd:6.1f} us")

    # ── Benchmark 3: get_spatial_attn_mod ────────────────────────────────
    # Temporarily make is_keyframe False so it actually allocates
    cache.is_keyframe = False
    def bench_spatial_attn_mod():
        cache.get_spatial_attn_mod(KV)

    mu, sd = timed_us(bench_spatial_attn_mod, n_iters)
    results.append(("get_spatial_attn_mod(kv_len)", mu, sd))
    print(f"  [3] get_spatial_attn_mod        {mu:8.1f} +/- {sd:6.1f} us")

    # ── Benchmark 4: Triton kernel launch ────────────────────────────────
    if _HAS_TRITON:
        # Pre-build the index tensors once (isolate kernel launch overhead)
        recomp_mask = cache.get_recompute_mask(layer_idx)
        active_head_ids = recomp_mask.nonzero(as_tuple=True)[0].to(torch.int32).contiguous()
        bos = torch.tensor([0], device=device, dtype=torch.int32)
        kept_vis = cache.spatial_keep_bool.nonzero(as_tuple=True)[0].to(torch.int32) + vs
        text = torch.arange(ve, seq_len, device=device, dtype=torch.int32)
        active_token_ids = torch.cat([bos, kept_vis, text]).contiguous()
        S_active = active_token_ids.shape[0]
        num_active = len(active_head_ids)
        sm_scale = 1.0 / math.sqrt(D)

        q_c = query.contiguous()
        k_c = key_exp.contiguous()
        v_c = val_exp.contiguous()

        def bench_triton_kernel():
            out = torch.zeros_like(query)
            grid = (triton.cdiv(Q, BLOCK_M), num_active, B)
            _selective_sparse_attn_fwd[grid](
                q_c, k_c, v_c, out,
                active_head_ids, active_token_ids,
                sm_scale,
                q_c.stride(0), q_c.stride(1), q_c.stride(2), q_c.stride(3),
                k_c.stride(0), k_c.stride(1), k_c.stride(2), k_c.stride(3),
                v_c.stride(0), v_c.stride(1), v_c.stride(2), v_c.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                S_active, Q,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
                IS_CAUSAL=True,
            )

        mu, sd = timed_us(bench_triton_kernel, n_iters)
        results.append(("Triton kernel launch (isolated)", mu, sd))
        print(f"  [4] Triton kernel launch        {mu:8.1f} +/- {sd:6.1f} us")

        # Also time the full wrapper for comparison
        causal_mask = None  # kernel handles causal internally
        def bench_full_wrapper():
            temporal_sparse_attention(query, key_exp, val_exp, causal_mask,
                                      cache, layer_idx)

        mu, sd = timed_us(bench_full_wrapper, n_iters)
        results.append(("temporal_sparse_attention (full)", mu, sd))
        print(f"  [4b] full wrapper (end-to-end)   {mu:8.1f} +/- {sd:6.1f} us")
    else:
        print(f"  [4] Triton kernel launch         SKIPPED (no triton)")
        print(f"  [4b] full wrapper                SKIPPED (no triton)")

    # ── Benchmark 5: Cache read/write ────────────────────────────────────
    fake_attn_out = torch.randn(B, H, Q, D, device=device, dtype=dtype)

    def bench_cache_store():
        cache.store_heads(layer_idx, fake_attn_out[:, :, -1, :])

    mu, sd = timed_us(bench_cache_store, n_iters)
    results.append(("store_heads (write cache)", mu, sd))
    print(f"  [5a] store_heads (write)        {mu:8.1f} +/- {sd:6.1f} us")

    def bench_cache_read_and_fill():
        recomp_mask = cache.get_recompute_mask(layer_idx)
        cached_idx = (~recomp_mask).nonzero(as_tuple=True)[0]
        if len(cached_idx) > 0:
            cached_vals = cache.get_cached_heads(layer_idx)
            fake_attn_out[:, cached_idx, -1, :] = cached_vals[cached_idx].to(dtype)

    mu, sd = timed_us(bench_cache_read_and_fill, n_iters)
    results.append(("cache read + fill cached heads", mu, sd))
    print(f"  [5b] cache read + fill          {mu:8.1f} +/- {sd:6.1f} us")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f" Summary (sorted by mean time)")
    print(f"{'='*65}")
    results.sort(key=lambda x: x[1], reverse=True)
    total = sum(r[1] for r in results)
    for name, mu, sd in results:
        pct = mu / total * 100
        print(f"  {mu:8.1f} us  ({pct:5.1f}%)  {name}")
    print(f"  {'─'*55}")
    print(f"  {total:8.1f} us  (100.0%)  TOTAL (sum of all components)")
    print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This profiler requires a GPU.")
        raise SystemExit(1)
    main()
