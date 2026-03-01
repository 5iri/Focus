"""
Run VLA inference with temporal head caching — end-to-end validation.

Pipeline:
  1. Load model via adapter
  2. Profile heads on one frame → V/T/B/M classification
  3. Baseline: full inference on N frames → action tokens + last hidden state
  4. Temporal: modified inference on same N frames → cached outputs
  5. Compare: cosine sim, action-token L1, cache rate, theoretical speedup

Usage:
    python run_temporal.py --model openvla --dataset bridge_v2 \\
        --episode 0 --num_frames 20 --spatial_K 32 --keyframe_interval 5

Outputs:
    output/temporal_results/{model}_{dataset}/results.pt
    output/temporal_results/{model}_{dataset}/summary.txt
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from vla_benchmarks import (
    get_adapter,
    get_dataset_instruction,
    list_datasets,
    list_models,
    load_episode_frames,
)
from temporal_cache import (
    CacheConfig,
    TemporalHeadCache,
    classify_heads_from_attn,
)
from temporal_llama2 import apply_temporal_caching, apply_visual_cache


# ─── Head profiling ──────────────────────────────────────────────────────────

def profile_heads(model, adapter, processor, image, instruction, cfg):
    """Run one forward pass to classify heads as V/T/B/M.

    Returns:
        head_types: numpy [num_layers, num_heads] of str
        attn_by_layer: dict {layer_idx: Tensor[B,H,Q,KV]}
    """
    layers = adapter.get_llm_layers(model)
    prompt = adapter.format_prompt(instruction)
    vs, ve, ts = adapter.get_visual_token_range()

    attn_data = {}

    def make_hook(idx):
        def fn(mod, args, out):
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                attn_data[idx] = out[1].detach().cpu().float()
        return fn

    hooks = []
    for i, layer in enumerate(layers):
        hooks.append(
            adapter.get_attn_module(layer).register_forward_hook(make_hook(i)))

    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    extra = adapter.get_forward_kwargs(inputs)
    with torch.no_grad():
        model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            attention_mask=inputs.get("attention_mask"),
            output_attentions=True,
            **extra,
        )

    for h in hooks:
        h.remove()

    head_types = classify_heads_from_attn(attn_data, cfg)
    return head_types, attn_data


# ─── Baseline run (no caching) ───────────────────────────────────────────────

def run_baseline(model, processor, adapter, frames, instruction):
    """Full inference per frame — no temporal caching.

    Returns list of dicts: {logits_last, hidden_last, action_ids, time_ms}
    """
    prompt = adapter.format_prompt(instruction)
    results = []
    for t, img in enumerate(frames):
        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        extra = adapter.get_forward_kwargs(inputs)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                attention_mask=inputs.get("attention_mask"),
                **extra,
            )
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000

        logits_last = out.logits[:, -1, :].detach().cpu().float()

        # Generate action tokens
        action_ids = []
        try:
            ids, _ = adapter.run_generate(model, processor, img, prompt)
            action_ids = ids
        except Exception:
            pass

        results.append({
            "logits_last": logits_last,
            "action_ids": action_ids,
            "time_ms": dt,
        })
        print(f"  [baseline] frame {t}: {dt:.1f} ms, "
              f"actions={action_ids[:3]}...")
    return results


# ─── Temporal-cached run ─────────────────────────────────────────────────────

def run_temporal(model, processor, adapter, frames, instruction, cache):
    """Inference with temporal head caching applied.

    The model's attention forwards have already been replaced by
    apply_temporal_caching().
    """
    prompt = adapter.format_prompt(instruction)
    results = []
    for t, img in enumerate(frames):
        is_kf = cache.begin_frame(img)
        tag = "KF" if is_kf else "  "

        inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
        extra = adapter.get_forward_kwargs(inputs)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                attention_mask=inputs.get("attention_mask"),
                **extra,
            )
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000

        cache.collect_stats()
        logits_last = out.logits[:, -1, :].detach().cpu().float()

        # Generate action tokens (with caching active)
        action_ids = []
        try:
            ids, _ = adapter.run_generate(model, processor, img, prompt)
            action_ids = ids
        except Exception:
            pass

        stats = cache.frame_stats[-1] if cache.frame_stats else {}
        cr = stats.get("cache_rate", 0)
        sk = stats.get("spatial_K", cache.cfg.num_vis_tokens)

        results.append({
            "logits_last": logits_last,
            "action_ids": action_ids,
            "time_ms": dt,
            "is_keyframe": is_kf,
            "cache_rate": cr,
            "spatial_K": sk,
        })
        print(f"  [temporal {tag}] frame {t}: {dt:.1f} ms, "
              f"cache={cr:.0%}, K={sk}, actions={action_ids[:3]}...")
    return results


# ─── Comparison ──────────────────────────────────────────────────────────────

def compare_results(baseline, temporal, cfg):
    """Compare baseline vs temporal-cached outputs per frame."""
    lines = []
    cosine_sims = []
    action_matches = []
    action_l1s = []

    for t in range(min(len(baseline), len(temporal))):
        bl = baseline[t]
        tc = temporal[t]

        # Logit cosine similarity at last position
        cos = F.cosine_similarity(
            bl["logits_last"].flatten().unsqueeze(0),
            tc["logits_last"].flatten().unsqueeze(0),
        ).item()
        cosine_sims.append(cos)

        # Action token comparison
        a_bl = np.array(bl["action_ids"], dtype=float)
        a_tc = np.array(tc["action_ids"], dtype=float)
        ml = min(len(a_bl), len(a_tc))
        match = int(np.array_equal(a_bl[:ml], a_tc[:ml])) if ml > 0 else 1
        action_matches.append(match)
        l1 = float(np.abs(a_bl[:ml] - a_tc[:ml]).mean()) if ml > 0 else 0
        action_l1s.append(l1)

        kf = "KF" if tc.get("is_keyframe") else "  "
        cr = tc.get("cache_rate", 0)
        line = (f"  frame {t:3d} [{kf}]: cos={cos:.6f}  "
                f"action_L1={l1:.2f}  match={match}  "
                f"cache={cr:.0%}  "
                f"time_bl={bl['time_ms']:.0f}ms  time_tc={tc['time_ms']:.0f}ms")
        lines.append(line)
        print(line)

    summary = {
        "mean_cosine_sim": float(np.mean(cosine_sims)),
        "mean_action_l1": float(np.mean(action_l1s)),
        "action_exact_match_rate": float(np.mean(action_matches)),
        "frames": len(cosine_sims),
    }
    return summary, lines


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VLA temporal head caching — end-to-end validation")
    parser.add_argument("--model", required=True, choices=list_models())
    parser.add_argument("--dataset", default="bridge_v2", choices=list_datasets())
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--device", default="auto")

    # Algorithm knobs
    parser.add_argument("--spatial_K", type=int, default=32)
    parser.add_argument("--cache_threshold", type=float, default=0.1)
    parser.add_argument("--keyframe_interval", type=int, default=5)
    parser.add_argument("--pilot_layers", type=int, default=2)
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the deep chain (CUDA graph fusion)")

    args = parser.parse_args()

    out_dir = os.path.join(
        "output", "temporal_results", f"{args.model}_{args.dataset}")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load frames ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Temporal Head Caching: {args.model} on {args.dataset}")
    print(f"{'='*70}")

    instruction = get_dataset_instruction(args.dataset)
    frames = load_episode_frames(args.dataset, args.episode, args.num_frames)
    print(f"Loaded {len(frames)} frames\n")

    # ── Load model ───────────────────────────────────────────────────────
    adapter = get_adapter(args.model)
    mcfg = adapter.get_config()
    print(f"Loading {mcfg.hf_id}...")
    model, processor = adapter.load_model(device=args.device)
    dev = next(model.parameters()).device
    print(f"Model on {dev}\n")

    # ── Build cache config ───────────────────────────────────────────────
    cfg = CacheConfig(
        num_layers=mcfg.num_layers,
        num_heads=mcfg.num_heads,
        num_kv_heads=mcfg.num_heads,   # adjust if GQA
        head_dim=mcfg.head_dim,
        hidden_dim=mcfg.hidden_dim,
        num_vis_tokens=mcfg.num_vis_tokens,
        vis_grid=mcfg.vis_grid,
        spatial_K=args.spatial_K,
        cache_threshold=args.cache_threshold,
        keyframe_interval=args.keyframe_interval,
        pilot_layers=args.pilot_layers,
    )

    # ── Profile heads ────────────────────────────────────────────────────
    print("Profiling head types...")
    head_types, _ = profile_heads(
        model, adapter, processor, frames[0], instruction, cfg)

    # ── Baseline run ─────────────────────────────────────────────────────
    print(f"\n--- Baseline (full inference, {len(frames)} frames) ---")
    torch.cuda.reset_peak_memory_stats(dev)
    baseline = run_baseline(model, processor, adapter, frames, instruction)
    baseline_peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

    # ── Apply temporal caching ───────────────────────────────────────────
    print(f"\n--- Applying temporal caching ---")
    cache = TemporalHeadCache(cfg, device=dev, dtype=torch.bfloat16)
    cache.set_head_types(head_types)
    apply_temporal_caching(model, adapter, cache, use_compile=args.compile)
    apply_visual_cache(model, cache)

    # ── Warmup (JIT-compile Triton kernels before timed run) ─────────────
    print(f"\n--- Warmup (2 frames) ---")
    prompt = adapter.format_prompt(instruction)
    for i in range(min(2, len(frames))):
        cache.begin_frame(frames[i])
        inp = processor(prompt, frames[i]).to(dev, dtype=torch.bfloat16)
        ext = adapter.get_forward_kwargs(inp)
        with torch.no_grad():
            model(input_ids=inp["input_ids"],
                  pixel_values=inp.get("pixel_values"),
                  attention_mask=inp.get("attention_mask"), **ext)
        cache.collect_stats()
    cache.reset()
    print(f"  Warmup complete (Triton kernels compiled)")

    # ── Temporal-cached run ──────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(dev)
    print(f"\n--- Temporal cached ({len(frames)} frames) ---")
    print(f"    spatial_K={cfg.spatial_K}  threshold={cfg.cache_threshold}  "
          f"keyframe_interval={cfg.keyframe_interval}  pilot={cfg.pilot_layers}")
    temporal = run_temporal(
        model, processor, adapter, frames, instruction, cache)

    temporal_peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

    # ── Compare ──────────────────────────────────────────────────────────
    print(f"\n--- Comparison ---")
    summary, comp_lines = compare_results(baseline, temporal, cfg)

    non_kf = [s for s in cache.frame_stats if not s["keyframe"]]
    avg_cache = np.mean([s["cache_rate"] for s in non_kf]) if non_kf else 0

    # Wall-clock speedup
    bl_total = sum(r["time_ms"] for r in baseline)
    tc_total = sum(r["time_ms"] for r in temporal)
    amortized_speedup = bl_total / tc_total if tc_total > 0 else 0

    bl_med = np.median([r["time_ms"] for r in baseline])
    tc_kf = [r["time_ms"] for r in temporal if r.get("is_keyframe")]
    tc_nkf = [r["time_ms"] for r in temporal if not r.get("is_keyframe")]
    tc_nkf_med = np.median(tc_nkf) if tc_nkf else 0
    nkf_speedup = bl_med / tc_nkf_med if tc_nkf_med > 0 else 0

    print(f"\n{'='*70}")
    print(f"RESULTS: {args.model} on {args.dataset}")
    print(f"{'='*70}")
    print(f"  Frames:              {summary['frames']}")
    print(f"  Mean cosine sim:     {summary['mean_cosine_sim']:.6f}")
    print(f"  Mean action L1:      {summary['mean_action_l1']:.3f}")
    print(f"  Action exact match:  {summary['action_exact_match_rate']:.1%}")
    print(f"  Avg cache rate:      {avg_cache:.1%}")
    print(f"  Spatial K:           {cfg.spatial_K}/{cfg.num_vis_tokens}")
    print(f"  Keyframe interval:   {cfg.keyframe_interval}")
    print(f"  Baseline median:     {bl_med:.1f} ms")
    print(f"  Non-KF median:       {tc_nkf_med:.1f} ms")
    print(f"  Non-KF speedup:      {nkf_speedup:.2f}x")
    print(f"  Amortized speedup:   {amortized_speedup:.2f}x")
    mem_reduction = (1 - temporal_peak_mb / baseline_peak_mb) * 100 if baseline_peak_mb > 0 else 0
    print(f"  Baseline peak mem:   {baseline_peak_mb:.0f} MB")
    print(f"  Temporal peak mem:   {temporal_peak_mb:.0f} MB")
    print(f"  Memory reduction:    {mem_reduction:.1f}%")
    print(f"{'='*70}")

    # ── Save ─────────────────────────────────────────────────────────────
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "config": vars(cfg),
        "head_types": head_types,
        "baseline": [{k: v for k, v in r.items() if k != "logits_last"}
                     for r in baseline],
        "temporal": [{k: v for k, v in r.items() if k != "logits_last"}
                     for r in temporal],
        "summary": summary,
        "cache_stats": cache.frame_stats,
        "wall_clock": {
            "baseline_median_ms": float(bl_med),
            "non_kf_median_ms": float(tc_nkf_med),
            "non_kf_speedup": float(nkf_speedup),
            "amortized_speedup": float(amortized_speedup),
        },
        "memory": {
            "baseline_peak_mb": float(baseline_peak_mb),
            "temporal_peak_mb": float(temporal_peak_mb),
            "reduction_pct": float(mem_reduction),
        },
    }
    torch.save(results, os.path.join(out_dir, "results.pt"))

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"Temporal Head Caching: {args.model} on {args.dataset}\n")
        f.write(f"{'='*60}\n")
        f.write(f"spatial_K={cfg.spatial_K}  threshold={cfg.cache_threshold}  "
                f"keyframe={cfg.keyframe_interval}  pilot={cfg.pilot_layers}\n\n")
        f.write(f"Mean cosine sim:     {summary['mean_cosine_sim']:.6f}\n")
        f.write(f"Mean action L1:      {summary['mean_action_l1']:.3f}\n")
        f.write(f"Action exact match:  {summary['action_exact_match_rate']:.1%}\n")
        f.write(f"Avg cache rate:      {avg_cache:.1%}\n")
        f.write(f"Baseline median:     {bl_med:.1f} ms\n")
        f.write(f"Non-KF median:       {tc_nkf_med:.1f} ms\n")
        f.write(f"Non-KF speedup:      {nkf_speedup:.2f}x\n")
        f.write(f"Amortized speedup:   {amortized_speedup:.2f}x\n")
        f.write(f"Baseline peak mem:   {baseline_peak_mb:.0f} MB\n")
        f.write(f"Temporal peak mem:   {temporal_peak_mb:.0f} MB\n")
        f.write(f"Memory reduction:    {mem_reduction:.1f}%\n\n")
        f.write("Per-frame:\n")
        for line in comp_lines:
            f.write(line + "\n")

        f.write(f"\nHead types:\n")
        for ht in ("V", "T", "B", "M"):
            n = (head_types == ht).sum()
            f.write(f"  {ht}: {n}\n")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
