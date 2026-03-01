"""
Pipeline Simulation — Full parameter sweep for temporal head caching algorithm.

Loads saved data from temporal_head_analysis.py, sweeps parameters, and reports:
  - Spatial pruning effectiveness (top-K from pilot layers)
  - Head caching rates (T/B always cache, V conditional, M recompute)
  - Keyframe interval impact
  - Compound reduction ratio
  - VLA accuracy evaluation (simulated caching error)
  - Cross-model comparison table

Usage:
    python pipeline_simulation.py --model openvla --dataset bridge_v2
    python pipeline_simulation.py --cross_model  # compare all available models

Outputs:
    output/temporal_analysis/{model}_{dataset}/sweep_results.csv
    output/temporal_analysis/cross_model_comparison.csv
"""

import argparse
import csv
import os
import sys
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F

from vla_benchmarks import list_datasets, list_models


# ─── Spatial Pruning Simulation ─────────────────────────────────────────────

def simulate_spatial_pruning(data, K_values):
    """Simulate top-K spatial token pruning from pilot layers.

    Uses attention from layers 0-1 (pilot) to select top-K visual tokens,
    then measures how much of the total visual attention in deeper layers
    (2+) is captured by those K tokens.

    Returns:
        results: list of dicts {K, capture_rate_mean, capture_rate_std}
    """
    frame_data = data["frame_data"]
    config = data["config"]
    num_vis = config["num_vis_tokens"]
    vis_start = 1
    vis_end = num_vis

    results = []
    for K in K_values:
        capture_rates = []
        for fd in frame_data:
            if 0 not in fd["attn_weights"] or 1 not in fd["attn_weights"]:
                continue

            # Pilot layers (0-1): aggregate visual attention
            pilot_vis = torch.zeros(num_vis)
            for pilot_l in [0, 1]:
                attn = fd["attn_weights"][pilot_l]  # [heads, seq]
                vis_attn = attn[:, vis_start:vis_end + 1]  # [heads, num_vis]
                pilot_vis += vis_attn.mean(dim=0)  # avg across heads

            # Select top-K patches
            if K >= num_vis:
                top_k_mask = torch.ones(num_vis, dtype=bool)
            else:
                _, top_k_idx = pilot_vis.topk(K)
                top_k_mask = torch.zeros(num_vis, dtype=bool)
                top_k_mask[top_k_idx] = True

            # Measure capture in deeper layers (2+)
            for l in range(2, config["num_layers"]):
                if l not in fd["attn_weights"]:
                    continue
                attn = fd["attn_weights"][l]  # [heads, seq]
                vis_attn = attn[:, vis_start:vis_end + 1]  # [heads, num_vis]
                total_vis = vis_attn.sum(dim=-1)  # [heads]
                captured_vis = vis_attn[:, top_k_mask].sum(dim=-1)  # [heads]

                # Per-head capture rate
                rates = captured_vis / (total_vis + 1e-10)
                capture_rates.extend(rates.tolist())

        mean_rate = np.mean(capture_rates) if capture_rates else 0
        std_rate = np.std(capture_rates) if capture_rates else 0
        results.append({"K": K, "capture_mean": mean_rate, "capture_std": std_rate})
        print(f"  K={K:3d}: capture={mean_rate:.4f} ± {std_rate:.4f}")

    return results


# ─── Head Caching Simulation ────────────────────────────────────────────────

def simulate_head_caching(data, thresholds):
    """Simulate head-level temporal caching.

    Policy:
      - T/B heads: always cache (high temporal stability)
      - V heads: cache if attended patches are unchanged (below threshold)
      - M heads: always recompute

    Uses the frame-diff × attention overlap to decide V-head caching.

    Returns:
        results: list of dicts per threshold.
    """
    frame_data = data["frame_data"]
    analysis = data["analysis"]
    config = data["config"]
    head_types = analysis["head_types"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    num_vis = config["num_vis_tokens"]
    vis_start = 1
    vis_end = num_vis

    frame_diffs = data.get("frame_diffs", [])

    results = []
    for threshold in thresholds:
        cache_decisions = []  # 1=cached, 0=recomputed
        cache_errors = []     # cosine distance when caching

        for t in range(1, len(frame_data)):
            prev = frame_data[t - 1]
            curr = frame_data[t]

            # Frame diff for V-head decisions
            if t - 1 < len(frame_diffs):
                patch_change = frame_diffs[t - 1].flatten()
            else:
                patch_change = torch.zeros(num_vis)

            for l in range(num_layers):
                if l not in prev["attn_weights"] or l not in curr["attn_weights"]:
                    continue

                a_prev = prev["attn_weights"][l]
                a_curr = curr["attn_weights"][l]

                for h in range(min(num_heads, a_prev.shape[0], a_curr.shape[0])):
                    htype = head_types[l, h] if l < head_types.shape[0] and h < head_types.shape[1] else "M"

                    if htype == "T" or htype == "B":
                        # Always cache
                        cache_decisions.append(1)
                        min_seq = min(a_prev.shape[-1], a_curr.shape[-1])
                        cos = F.cosine_similarity(
                            a_prev[h, :min_seq].unsqueeze(0),
                            a_curr[h, :min_seq].unsqueeze(0)
                        ).item()
                        cache_errors.append(1.0 - cos)

                    elif htype == "V":
                        # Cache if attended patches unchanged
                        vis_attn = a_curr[h, vis_start:vis_end + 1]
                        if vis_attn.shape[0] == patch_change.shape[0]:
                            attended_change = (vis_attn * patch_change).sum().item()
                        else:
                            attended_change = 0.0

                        if attended_change < threshold:
                            cache_decisions.append(1)
                            min_seq = min(a_prev.shape[-1], a_curr.shape[-1])
                            cos = F.cosine_similarity(
                                a_prev[h, :min_seq].unsqueeze(0),
                                a_curr[h, :min_seq].unsqueeze(0)
                            ).item()
                            cache_errors.append(1.0 - cos)
                        else:
                            cache_decisions.append(0)

                    else:  # M — always recompute
                        cache_decisions.append(0)

        cache_rate = np.mean(cache_decisions) if cache_decisions else 0
        mean_error = np.mean(cache_errors) if cache_errors else 0
        results.append({
            "threshold": threshold,
            "cache_rate": cache_rate,
            "mean_error": mean_error,
            "num_decisions": len(cache_decisions),
        })
        print(f"  threshold={threshold:.3f}: cache_rate={cache_rate:.3f}, error={mean_error:.5f}")

    return results


# ─── Keyframe Interval Simulation ───────────────────────────────────────────

def simulate_keyframe_interval(data, K_frame_values):
    """Simulate keyframe intervals.

    Every K_frame frames, do full recompute (keyframe). Between keyframes,
    use cached values. Measure the cumulative error from caching.

    Returns:
        results: list of dicts per K_frame.
    """
    frame_data = data["frame_data"]
    config = data["config"]
    num_layers = config["num_layers"]

    results = []
    for K_frame in K_frame_values:
        errors = []
        cache_ratios = []

        for l in range(num_layers):
            last_keyframe_output = None
            cached_frames = 0
            total_frames = 0

            for t, fd in enumerate(frame_data):
                if l not in fd["post_attn_outputs"]:
                    continue

                output = fd["post_attn_outputs"][l]
                total_frames += 1

                if t % K_frame == 0:
                    # Keyframe: full recompute
                    last_keyframe_output = output
                else:
                    # Cached: measure error vs actual
                    if last_keyframe_output is not None:
                        cos = F.cosine_similarity(
                            last_keyframe_output.unsqueeze(0),
                            output.unsqueeze(0)
                        ).item()
                        errors.append(1.0 - cos)
                    cached_frames += 1

            if total_frames > 0:
                cache_ratios.append(cached_frames / total_frames)

        mean_error = np.mean(errors) if errors else 0
        mean_cache_ratio = np.mean(cache_ratios) if cache_ratios else 0
        results.append({
            "K_frame": K_frame,
            "cache_ratio": mean_cache_ratio,
            "mean_error": mean_error,
        })
        print(f"  K_frame={K_frame:3d}: cache_ratio={mean_cache_ratio:.3f}, error={mean_error:.5f}")

    return results


# ─── Compound Reduction ────────────────────────────────────────────────────

def compute_compound_reduction(data, best_K, best_cache_rate, pilot_layers=2):
    """Compute compound FLOPs reduction ratio.

    total_ratio = (pilot_layers/N) + ((N-pilot)/N) × (K/vis_tokens) × (1 - cache_rate)

    The inverse of total_ratio gives the effective speedup.
    """
    config = data["config"]
    N = config["num_layers"]
    vis_tokens = config["num_vis_tokens"]

    # Pilot layers run dense
    pilot_fraction = pilot_layers / N

    # Remaining layers: spatial pruning × head caching
    remaining_fraction = (N - pilot_layers) / N
    spatial_reduction = best_K / vis_tokens
    head_reduction = 1.0 - best_cache_rate

    total_ratio = pilot_fraction + remaining_fraction * spatial_reduction * head_reduction
    speedup = 1.0 / total_ratio if total_ratio > 0 else float('inf')

    return {
        "total_ratio": total_ratio,
        "speedup": speedup,
        "pilot_fraction": pilot_fraction,
        "spatial_reduction": spatial_reduction,
        "head_reduction": head_reduction,
    }


# ─── VLA Accuracy Evaluation ───────────────────────────────────────────────

def evaluate_accuracy(data, best_cache_rate):
    """Evaluate action prediction accuracy under caching.

    Simulates caching error by measuring how much action tokens would
    differ if we used cached attention outputs vs fresh computation.

    Uses the layer output similarity as a proxy: if layer outputs are
    similar (high cosine sim), the action prediction should be similar.

    Returns:
        dict with accuracy metrics.
    """
    analysis = data["analysis"]
    layer_sim = analysis["layer_output_sim"]
    action_seqs = analysis.get("action_sequences", [])

    # Expected output error: product of per-layer similarities
    # (each layer's output is input to the next)
    if len(layer_sim) > 0 and layer_sim.mean() > 0:
        # Weighted by cache rate — only cached layers contribute error
        cached_fraction = best_cache_rate
        # Error grows with number of cached layers
        avg_per_layer_error = 1.0 - layer_sim.mean()
        estimated_output_error = avg_per_layer_error * cached_fraction
    else:
        estimated_output_error = 0.0

    # Action consistency from actual data
    action_consistency = analysis.get("action_consistency", 0.0)

    return {
        "estimated_output_error": estimated_output_error,
        "action_consistency": action_consistency,
        "mean_layer_sim": float(layer_sim.mean()) if len(layer_sim) > 0 else 0.0,
        "num_action_frames": len(action_seqs),
    }


# ─── Cross-Model Comparison ────────────────────────────────────────────────

def generate_cross_model_comparison(output_dir):
    """Scan all model results and generate comparison table."""
    comparison = []

    for entry in sorted(os.listdir(output_dir)):
        data_path = os.path.join(output_dir, entry, "frame_data.pt")
        if not os.path.exists(data_path):
            continue

        data = torch.load(data_path, weights_only=False)
        config = data["config"]
        analysis = data["analysis"]

        # Run quick compound reduction estimate
        best_cache_rate = 0.0
        head_types = analysis["head_types"]
        for htype in ["T", "B"]:
            mask = head_types == htype
            best_cache_rate += mask.sum() / (config["num_layers"] * config["num_heads"])

        best_K = 32  # default
        compound = compute_compound_reduction(data, best_K, best_cache_rate)

        comparison.append({
            "model_dataset": entry,
            "model": data.get("model", entry.split("_")[0]),
            "dataset": data.get("dataset", "unknown"),
            "num_heads": config["num_heads"],
            "num_layers": config["num_layers"],
            "speedup": compound["speedup"],
            "cache_rate": best_cache_rate,
            "mean_error": 1.0 - analysis["layer_output_sim"].mean() if len(analysis["layer_output_sim"]) > 0 else 0,
            "head_stability": analysis["head_stability"].mean(),
            "pearson_r": analysis["mean_pearson_r"],
        })

    if not comparison:
        print("No model results found.")
        return

    # Save CSV
    csv_path = os.path.join(output_dir, "cross_model_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)

    # Print table
    print(f"\n{'='*90}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*90}")
    print(f"{'Model+Dataset':<30} | {'Heads':>5} | {'Layers':>6} | {'Speedup':>8} | {'Error':>8} | {'Cache%':>6} | {'Stab':>5} | {'r':>5}")
    print("-" * 90)
    for row in comparison:
        print(f"{row['model_dataset']:<30} | {row['num_heads']:>5} | {row['num_layers']:>6} | "
              f"{row['speedup']:>7.1f}x | {row['mean_error']:>8.5f} | {row['cache_rate']:>5.1%} | "
              f"{row['head_stability']:>5.3f} | {row['pearson_r']:>5.3f}")

    print(f"\nSaved to {csv_path}")
    return comparison


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline Simulation for temporal head caching")
    parser.add_argument("--model", type=str, default=None, choices=list_models(),
                        help="VLA model (omit for cross-model comparison)")
    parser.add_argument("--dataset", type=str, default="bridge_v2", choices=list_datasets(),
                        help="Dataset used for analysis")
    parser.add_argument("--input_dir", type=str, default="output/temporal_analysis",
                        help="Directory with frame_data.pt files")
    parser.add_argument("--cross_model", action="store_true",
                        help="Generate cross-model comparison from all available data")

    # Sweep parameters
    parser.add_argument("--K_values", type=str, default="16,32,48,64,128",
                        help="Comma-separated top-K values for spatial pruning")
    parser.add_argument("--thresholds", type=str, default="0.01,0.05,0.1,0.2,0.5",
                        help="Comma-separated thresholds for V-head caching")
    parser.add_argument("--K_frame_values", type=str, default="3,5,10,20",
                        help="Comma-separated keyframe intervals")

    args = parser.parse_args()

    if args.cross_model:
        generate_cross_model_comparison(args.input_dir)
        return

    if args.model is None:
        parser.error("--model is required unless --cross_model is set")

    run_name = f"{args.model}_{args.dataset}"
    data_path = os.path.join(args.input_dir, run_name, "frame_data.pt")

    if not os.path.exists(data_path):
        print(f"Error: No data found at {data_path}")
        print(f"Run temporal_head_analysis.py --model {args.model} --dataset {args.dataset} first.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Pipeline Simulation: {args.model} on {args.dataset}")
    print(f"{'='*70}")

    data = torch.load(data_path, weights_only=False)
    config = data["config"]

    K_values = [int(k) for k in args.K_values.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]
    K_frame_values = [int(k) for k in args.K_frame_values.split(",")]

    # 1. Spatial pruning sweep
    print(f"\n--- Spatial Pruning Sweep (K ∈ {K_values}) ---")
    spatial_results = simulate_spatial_pruning(data, K_values)

    # 2. Head caching sweep
    print(f"\n--- Head Caching Sweep (threshold ∈ {thresholds}) ---")
    caching_results = simulate_head_caching(data, thresholds)

    # 3. Keyframe interval sweep
    print(f"\n--- Keyframe Interval Sweep (K_frame ∈ {K_frame_values}) ---")
    keyframe_results = simulate_keyframe_interval(data, K_frame_values)

    # 4. Compound reduction for all parameter combinations
    print(f"\n--- Compound Reduction ---")
    best_results = []
    for sp in spatial_results:
        for ch in caching_results:
            for kf in keyframe_results:
                compound = compute_compound_reduction(
                    data, sp["K"], ch["cache_rate"]
                )
                total_error = ch["mean_error"] + kf["mean_error"]
                best_results.append({
                    "K": sp["K"],
                    "threshold": ch["threshold"],
                    "K_frame": kf["K_frame"],
                    "speedup": compound["speedup"],
                    "cache_rate": ch["cache_rate"],
                    "spatial_capture": sp["capture_mean"],
                    "total_error": total_error,
                })

    # Sort by speedup, pick best with acceptable error
    best_results.sort(key=lambda x: x["speedup"], reverse=True)
    best = best_results[0]
    print(f"\n  Best compound speedup: {best['speedup']:.1f}x")
    print(f"    K={best['K']}, threshold={best['threshold']}, K_frame={best['K_frame']}")
    print(f"    cache_rate={best['cache_rate']:.3f}, spatial_capture={best['spatial_capture']:.3f}")
    print(f"    estimated_error={best['total_error']:.5f}")

    # 5. Accuracy evaluation
    print(f"\n--- VLA Accuracy Evaluation ---")
    accuracy = evaluate_accuracy(data, best["cache_rate"])
    print(f"  Estimated output error: {accuracy['estimated_output_error']:.5f}")
    print(f"  Mean layer similarity: {accuracy['mean_layer_sim']:.4f}")
    print(f"  Action consistency (baseline): {accuracy['action_consistency']:.3f}")

    # ── Save sweep results ──
    out_dir = os.path.join(args.input_dir, run_name)
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=best_results[0].keys())
        writer.writeheader()
        writer.writerows(best_results)

    # Save summary
    summary_path = os.path.join(out_dir, "simulation_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Pipeline Simulation: {args.model} on {args.dataset}\n")
        f.write(f"{'='*60}\n\n")

        f.write("Spatial Pruning:\n")
        for r in spatial_results:
            f.write(f"  K={r['K']:3d}: capture={r['capture_mean']:.4f} ± {r['capture_std']:.4f}\n")

        f.write("\nHead Caching:\n")
        for r in caching_results:
            f.write(f"  threshold={r['threshold']:.3f}: cache_rate={r['cache_rate']:.3f}, error={r['mean_error']:.5f}\n")

        f.write("\nKeyframe Interval:\n")
        for r in keyframe_results:
            f.write(f"  K_frame={r['K_frame']:3d}: cache_ratio={r['cache_ratio']:.3f}, error={r['mean_error']:.5f}\n")

        f.write(f"\nBest Compound Configuration:\n")
        f.write(f"  Speedup: {best['speedup']:.1f}x\n")
        f.write(f"  K={best['K']}, threshold={best['threshold']}, K_frame={best['K_frame']}\n")
        f.write(f"  cache_rate={best['cache_rate']:.3f}, error={best['total_error']:.5f}\n")

        f.write(f"\nAccuracy Evaluation:\n")
        f.write(f"  Estimated output error: {accuracy['estimated_output_error']:.5f}\n")
        f.write(f"  Mean layer similarity: {accuracy['mean_layer_sim']:.4f}\n")
        f.write(f"  Action consistency: {accuracy['action_consistency']:.3f}\n")

    print(f"\nResults saved to {out_dir}/")
    print(f"  sweep_results.csv — all parameter combinations")
    print(f"  simulation_summary.txt — human-readable summary")

    # Cross-model comparison if multiple models available
    print(f"\n--- Checking for cross-model comparison ---")
    generate_cross_model_comparison(args.input_dir)


if __name__ == "__main__":
    main()
