"""
Temporal Head Analysis — Data Collection + Analysis for VLA models.

For each model/dataset combination, collects per-frame attention data across
a temporal sequence of frames, then analyzes:
  1. Head classification stability (V/T/B/M across frames)
  2. Attention pattern temporal stability (cosine sim between frames)
  3. Layer output temporal stability (post-o_proj cosine sim)
  4. Frame-diff → head staleness correlation
  5. Action prediction accuracy under caching (cached vs full-compute)

Usage:
    python temporal_head_analysis.py --model openvla --dataset bridge_v2 \
        --episode 0 --num_frames 20

Outputs:
    output/temporal_analysis/{model}_{dataset}/frame_data.pt
    output/temporal_analysis/{model}_{dataset}/summary.txt
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from vla_benchmarks import (
    get_adapter,
    get_dataset_instruction,
    list_datasets,
    list_models,
    load_episode_frames,
)

# ─── Head classification thresholds (from profiling) ────────────────────────

VIS_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.60
BOS_THRESHOLD = 0.60


def classify_head(vis_frac: float, text_frac: float, bos_frac: float) -> str:
    """Classify a head as V(isual), T(ext), B(OS), or M(ixed)."""
    if vis_frac > VIS_THRESHOLD:
        return "V"
    elif text_frac > TEXT_THRESHOLD:
        return "T"
    elif bos_frac > BOS_THRESHOLD:
        return "B"
    return "M"


def compute_frame_diff(img_a, img_b, grid_h: int = 16, grid_w: int = 16):
    """Compute per-patch L1 frame difference between two PIL images.

    Returns:
        patch_change: [grid_h, grid_w] tensor of normalized change per patch.
    """
    import torchvision.transforms as T
    to_tensor = T.ToTensor()  # [C, H, W] float in [0, 1]

    a = to_tensor(img_a.resize((grid_w * 14, grid_h * 14)))  # match ViT patch size
    b = to_tensor(img_b.resize((grid_w * 14, grid_h * 14)))

    diff = (a - b).abs()  # [C, H, W]

    # Reshape into patches and average
    C = diff.shape[0]
    patch_h = diff.shape[1] // grid_h
    patch_w = diff.shape[2] // grid_w
    diff = diff.reshape(C, grid_h, patch_h, grid_w, patch_w)
    diff = diff.permute(1, 3, 0, 2, 4).reshape(grid_h, grid_w, -1)
    patch_change = diff.mean(dim=-1)  # [grid_h, grid_w]

    # Normalize to [0, 1]
    max_val = patch_change.max()
    if max_val > 0:
        patch_change = patch_change / max_val

    return patch_change


# ─── Data Collection ────────────────────────────────────────────────────────

def collect_frame_data(model, processor, adapter, frames, instruction):
    """Collect attention data across all frames.

    For each frame, hooks every attention module and captures:
      - last_token_attn: [num_heads, seq_len] — last token's attention row
      - post_attn_output: [hidden_dim] — post-o_proj output at last token

    Also runs generate to capture action token IDs for accuracy evaluation.

    Returns:
        frame_data: list of dicts, one per frame.
    """
    config = adapter.get_config()
    vis_start, vis_end, text_start = adapter.get_visual_token_range()
    layers = adapter.get_llm_layers(model)
    prompt = adapter.format_prompt(instruction)

    frame_data = []

    for t, image in enumerate(frames):
        print(f"  Frame {t}/{len(frames)-1}...", end=" ", flush=True)

        # ── Hook attention weights and post-attn outputs ──
        attn_weights = {}
        post_attn_outputs = {}

        def make_attn_hook(layer_idx):
            def hook_fn(module, args, output):
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    # Attention weights: [batch, heads, seq, seq]
                    attn = output[1].detach().cpu().float()
                    # Keep only last-token row to save memory
                    attn_weights[layer_idx] = attn[:, :, -1, :].squeeze(0)  # [heads, seq]
                elif isinstance(output, tuple):
                    # Some models don't return attn weights; capture output[0]
                    pass
            return hook_fn

        def make_output_hook(layer_idx):
            def hook_fn(module, input, output):
                # Capture the output of the full decoder layer (post-residual)
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # Last token hidden state
                post_attn_outputs[layer_idx] = hidden[:, -1, :].detach().cpu().float().squeeze(0)
            return hook_fn

        hooks = []
        for i, layer in enumerate(layers):
            attn_mod = adapter.get_attn_module(layer)
            h = attn_mod.register_forward_hook(make_attn_hook(i))
            hooks.append(h)
            # Layer-level output hook
            h2 = layer.register_forward_hook(make_output_hook(i))
            hooks.append(h2)

        # ── Forward pass with attention output ──
        try:
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
        except Exception as e:
            # Fallback for models with different processor API
            try:
                inputs = processor(text=prompt, images=image, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                extra = adapter.get_forward_kwargs(inputs)
                with torch.no_grad():
                    model(**inputs, output_attentions=True, **extra)
            except Exception as e2:
                print(f"Error: {e2}")
                for h in hooks:
                    h.remove()
                continue

        for h in hooks:
            h.remove()

        # ── Generate action tokens for accuracy eval ──
        action_ids = []
        try:
            action_ids, _ = adapter.run_generate(model, processor, image, prompt)
        except Exception:
            pass

        frame_data.append({
            "frame_idx": t,
            "attn_weights": dict(attn_weights),      # {layer: [heads, seq]}
            "post_attn_outputs": dict(post_attn_outputs),  # {layer: [hidden_dim]}
            "action_ids": action_ids,
        })
        print(f"captured {len(attn_weights)} layers, {len(action_ids)} action tokens")

    return frame_data


# ─── Analysis ───────────────────────────────────────────────────────────────

def analyze_head_classification_stability(frame_data, config):
    """Measure how stable V/T/B/M classification is across frames per head.

    Returns:
        stability: [num_layers, num_heads] — fraction of frames with majority class.
        head_types: [num_layers, num_heads] — majority class label per head.
    """
    vis_start, vis_end = 1, config.num_vis_tokens
    text_start = vis_end + 1
    num_layers = config.num_layers
    num_heads = config.num_heads

    # Collect classifications per frame
    classifications = {(l, h): [] for l in range(num_layers) for h in range(num_heads)}

    for fd in frame_data:
        for layer_idx in range(num_layers):
            if layer_idx not in fd["attn_weights"]:
                continue
            attn = fd["attn_weights"][layer_idx]  # [heads, seq]
            seq_len = attn.shape[-1]
            for h in range(min(num_heads, attn.shape[0])):
                vis_frac = attn[h, vis_start:vis_end + 1].sum().item()
                text_frac = attn[h, text_start:min(seq_len, text_start + 200)].sum().item()
                bos_frac = attn[h, 0].item()
                cls = classify_head(vis_frac, text_frac, bos_frac)
                classifications[(layer_idx, h)].append(cls)

    stability = np.zeros((num_layers, num_heads))
    head_types = np.empty((num_layers, num_heads), dtype=object)

    for l in range(num_layers):
        for h in range(num_heads):
            classes = classifications[(l, h)]
            if not classes:
                stability[l, h] = 0
                head_types[l, h] = "?"
                continue
            from collections import Counter
            counts = Counter(classes)
            majority_cls, majority_count = counts.most_common(1)[0]
            stability[l, h] = majority_count / len(classes)
            head_types[l, h] = majority_cls

    return stability, head_types


def analyze_attention_temporal_stability(frame_data, config):
    """Per-head cosine similarity of attention patterns between consecutive frames.

    Returns:
        temporal_sim: [num_layers, num_heads] — mean cosine sim across frame pairs.
    """
    num_layers = config.num_layers
    num_heads = config.num_heads
    sims = {(l, h): [] for l in range(num_layers) for h in range(num_heads)}

    for t in range(1, len(frame_data)):
        prev = frame_data[t - 1]
        curr = frame_data[t]
        for l in range(num_layers):
            if l not in prev["attn_weights"] or l not in curr["attn_weights"]:
                continue
            a_prev = prev["attn_weights"][l]  # [heads, seq]
            a_curr = curr["attn_weights"][l]
            # Align sequence lengths (take min)
            min_seq = min(a_prev.shape[-1], a_curr.shape[-1])
            for h in range(min(num_heads, a_prev.shape[0], a_curr.shape[0])):
                p = a_prev[h, :min_seq]
                c = a_curr[h, :min_seq]
                cos = F.cosine_similarity(p.unsqueeze(0), c.unsqueeze(0)).item()
                sims[(l, h)].append(cos)

    temporal_sim = np.zeros((num_layers, num_heads))
    for l in range(num_layers):
        for h in range(num_heads):
            if sims[(l, h)]:
                temporal_sim[l, h] = np.mean(sims[(l, h)])
    return temporal_sim


def analyze_layer_output_stability(frame_data, config):
    """Post-attention output cosine sim at last token between consecutive frames.

    Returns:
        layer_sim: [num_layers] — mean cosine sim per layer.
    """
    num_layers = config.num_layers
    sims = {l: [] for l in range(num_layers)}

    for t in range(1, len(frame_data)):
        prev = frame_data[t - 1]
        curr = frame_data[t]
        for l in range(num_layers):
            if l not in prev["post_attn_outputs"] or l not in curr["post_attn_outputs"]:
                continue
            p = prev["post_attn_outputs"][l]
            c = curr["post_attn_outputs"][l]
            cos = F.cosine_similarity(p.unsqueeze(0), c.unsqueeze(0)).item()
            sims[l].append(cos)

    layer_sim = np.zeros(num_layers)
    for l in range(num_layers):
        if sims[l]:
            layer_sim[l] = np.mean(sims[l])
    return layer_sim


def analyze_frame_diff_correlation(frame_data, frames, config):
    """Correlation between frame diff and head attention change for V-heads.

    For each V-head, computes:
      attended_change = dot(head_vis_attn, patch_change)
    and correlates with the change in the head's attention pattern.

    Returns:
        correlations: dict of (layer, head) → Pearson r value (V-heads only).
        mean_r: mean Pearson r across all V-heads.
    """
    vis_start = 1
    vis_end = config.num_vis_tokens
    grid_h, grid_w = config.vis_grid
    num_layers = config.num_layers
    num_heads = config.num_heads

    # First, classify heads using first frame
    head_is_visual = np.zeros((num_layers, num_heads), dtype=bool)
    if frame_data:
        fd0 = frame_data[0]
        for l in range(num_layers):
            if l not in fd0["attn_weights"]:
                continue
            attn = fd0["attn_weights"][l]
            for h in range(min(num_heads, attn.shape[0])):
                vis_frac = attn[h, vis_start:vis_end + 1].sum().item()
                if vis_frac > VIS_THRESHOLD:
                    head_is_visual[l, h] = True

    # Collect (attended_change, attn_pattern_change) pairs per V-head
    per_head_data = {(l, h): ([], []) for l in range(num_layers)
                     for h in range(num_heads) if head_is_visual[l, h]}

    for t in range(1, min(len(frame_data), len(frames))):
        patch_change = compute_frame_diff(frames[t - 1], frames[t], grid_h, grid_w)
        patch_flat = patch_change.flatten()  # [num_vis_tokens]

        prev = frame_data[t - 1]
        curr = frame_data[t]

        for l in range(num_layers):
            if l not in prev["attn_weights"] or l not in curr["attn_weights"]:
                continue
            a_prev = prev["attn_weights"][l]
            a_curr = curr["attn_weights"][l]
            min_seq = min(a_prev.shape[-1], a_curr.shape[-1])

            for h in range(min(num_heads, a_prev.shape[0], a_curr.shape[0])):
                if not head_is_visual[l, h]:
                    continue

                # Visual attention from current frame
                vis_attn = a_curr[h, vis_start:vis_end + 1]
                if vis_attn.shape[0] != patch_flat.shape[0]:
                    continue

                # Attended change: how much the attended patches changed
                attended_change = (vis_attn * patch_flat).sum().item()

                # Attention pattern change (L1 distance)
                attn_change = (a_curr[h, :min_seq] - a_prev[h, :min_seq]).abs().sum().item()

                per_head_data[(l, h)][0].append(attended_change)
                per_head_data[(l, h)][1].append(attn_change)

    correlations = {}
    r_values = []
    for (l, h), (attended, changed) in per_head_data.items():
        if len(attended) < 3:
            continue
        r, p = stats.pearsonr(attended, changed)
        correlations[(l, h)] = r
        r_values.append(r)

    mean_r = np.mean(r_values) if r_values else 0.0
    return correlations, mean_r


def analyze_action_accuracy(frame_data):
    """Measure action prediction consistency across frames.

    Computes L1 distance between consecutive frame action predictions as a
    baseline for evaluating caching-induced error.

    Returns:
        mean_action_consistency: mean L1 distance between consecutive action vectors.
        action_sequences: list of action token ID lists per frame.
    """
    action_sequences = [fd["action_ids"] for fd in frame_data if fd["action_ids"]]

    if len(action_sequences) < 2:
        return 0.0, action_sequences

    dists = []
    for i in range(1, len(action_sequences)):
        a = np.array(action_sequences[i - 1], dtype=float)
        b = np.array(action_sequences[i], dtype=float)
        min_len = min(len(a), len(b))
        if min_len > 0:
            dists.append(np.abs(a[:min_len] - b[:min_len]).mean())

    return np.mean(dists) if dists else 0.0, action_sequences


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Temporal Head Analysis for VLA models")
    parser.add_argument("--model", type=str, required=True, choices=list_models(),
                        help="VLA model to benchmark")
    parser.add_argument("--dataset", type=str, default="bridge_v2", choices=list_datasets(),
                        help="Dataset to use for frame sequences")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode ID to analyze")
    parser.add_argument("--num_frames", type=int, default=20,
                        help="Number of frames to extract from episode")
    parser.add_argument("--output_dir", type=str, default="output/temporal_analysis",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for model loading")
    args = parser.parse_args()

    run_name = f"{args.model}_{args.dataset}"
    out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Load frames ──
    print(f"\n{'='*70}")
    print(f"Temporal Head Analysis: {args.model} on {args.dataset}")
    print(f"{'='*70}")

    instruction = get_dataset_instruction(args.dataset)
    frames = load_episode_frames(args.dataset, args.episode, args.num_frames)
    print(f"Loaded {len(frames)} frames")

    # ── Compute frame diffs ──
    print("\nComputing frame differences...")
    adapter = get_adapter(args.model)
    config = adapter.get_config()
    grid_h, grid_w = config.vis_grid

    frame_diffs = []
    for t in range(1, len(frames)):
        diff = compute_frame_diff(frames[t - 1], frames[t], grid_h, grid_w)
        frame_diffs.append(diff)
        if t <= 3 or t == len(frames) - 1:
            print(f"  Frame {t-1}→{t}: mean_diff={diff.mean():.4f}, max_diff={diff.max():.4f}")

    # ── Load model ──
    print(f"\nLoading model: {config.hf_id}")
    model, processor = adapter.load_model(device=args.device)
    print(f"Model loaded. Config: {config.num_layers} layers, {config.num_heads} heads")

    # ── Collect data ──
    print(f"\nCollecting per-frame attention data...")
    frame_data = collect_frame_data(model, processor, adapter, frames, instruction)
    print(f"Collected data for {len(frame_data)} frames")

    # ── Analysis ──
    print(f"\n{'='*70}")
    print("ANALYSIS RESULTS")
    print(f"{'='*70}")

    summary_lines = []
    summary_lines.append(f"Temporal Head Analysis: {args.model} on {args.dataset}")
    summary_lines.append(f"Episode: {args.episode}, Frames: {len(frames)}")
    summary_lines.append(f"Config: {config.num_layers}L x {config.num_heads}H, {config.num_vis_tokens} vis tokens")
    summary_lines.append("=" * 60)

    # 1. Head classification stability
    print("\n1. Head Classification Stability")
    stability, head_types = analyze_head_classification_stability(frame_data, config)
    mean_stability = stability.mean()
    print(f"   Mean stability: {mean_stability:.3f}")

    type_counts = {"V": 0, "T": 0, "B": 0, "M": 0}
    for l in range(config.num_layers):
        for h in range(config.num_heads):
            t = head_types[l, h]
            if t in type_counts:
                type_counts[t] += 1

    summary_lines.append(f"\n1. Head Classification Stability")
    summary_lines.append(f"   Mean stability: {mean_stability:.3f}")
    summary_lines.append(f"   Head type distribution: {type_counts}")

    # Stability per type
    for htype in ["V", "T", "B", "M"]:
        mask = head_types == htype
        if mask.any():
            type_stab = stability[mask].mean()
            summary_lines.append(f"   {htype}-head stability: {type_stab:.3f} ({mask.sum()} heads)")
            print(f"   {htype}-head stability: {type_stab:.3f} ({mask.sum()} heads)")

    # Per-layer breakdown
    summary_lines.append(f"\n   Per-layer head types (majority classification):")
    for l in range(config.num_layers):
        types_str = " ".join(head_types[l])
        layer_counts = {t: list(head_types[l]).count(t) for t in ["V", "T", "B", "M"]}
        summary_lines.append(f"   L{l:2d}: {types_str}  | V={layer_counts['V']:2d} T={layer_counts['T']:2d} B={layer_counts['B']:2d} M={layer_counts['M']:2d}")

    # 2. Attention temporal stability
    print("\n2. Attention Pattern Temporal Stability")
    temporal_sim = analyze_attention_temporal_stability(frame_data, config)
    mean_sim = temporal_sim.mean()
    print(f"   Mean cosine similarity: {mean_sim:.3f}")

    summary_lines.append(f"\n2. Attention Pattern Temporal Stability")
    summary_lines.append(f"   Mean cosine sim: {mean_sim:.3f}")

    for htype in ["V", "T", "B", "M"]:
        mask = head_types == htype
        if mask.any():
            type_sim = temporal_sim[mask].mean()
            summary_lines.append(f"   {htype}-head mean sim: {type_sim:.3f}")
            print(f"   {htype}-head mean sim: {type_sim:.3f}")

    # Per-layer
    summary_lines.append(f"\n   Per-layer mean temporal similarity:")
    for l in range(config.num_layers):
        layer_sim = temporal_sim[l].mean()
        summary_lines.append(f"   L{l:2d}: {layer_sim:.3f}")

    # 3. Layer output stability
    print("\n3. Layer Output Temporal Stability")
    layer_sim = analyze_layer_output_stability(frame_data, config)
    mean_layer_sim = layer_sim.mean()
    print(f"   Mean layer output cosine sim: {mean_layer_sim:.3f}")

    summary_lines.append(f"\n3. Layer Output Temporal Stability")
    summary_lines.append(f"   Mean layer output cosine sim: {mean_layer_sim:.3f}")
    for l in range(config.num_layers):
        summary_lines.append(f"   L{l:2d}: {layer_sim[l]:.3f}")

    # 4. Frame-diff correlation
    print("\n4. Frame-diff → Head Staleness Correlation")
    correlations, mean_r = analyze_frame_diff_correlation(frame_data, frames, config)
    print(f"   Mean Pearson r (V-heads): {mean_r:.3f}")
    print(f"   V-heads with r > 0.3: {sum(1 for r in correlations.values() if r > 0.3)}/{len(correlations)}")

    summary_lines.append(f"\n4. Frame-diff → Head Staleness Correlation")
    summary_lines.append(f"   Mean Pearson r (V-heads): {mean_r:.3f}")
    summary_lines.append(f"   V-heads with r > 0.3: {sum(1 for r in correlations.values() if r > 0.3)}/{len(correlations)}")
    for (l, h), r in sorted(correlations.items()):
        summary_lines.append(f"   L{l:2d} H{h:2d}: r={r:.3f}")

    # 5. Action prediction consistency
    print("\n5. Action Prediction Consistency")
    action_consistency, action_seqs = analyze_action_accuracy(frame_data)
    print(f"   Mean action L1 between consecutive frames: {action_consistency:.3f}")
    print(f"   Total frames with actions: {len(action_seqs)}")

    summary_lines.append(f"\n5. Action Prediction Consistency")
    summary_lines.append(f"   Mean action L1 between consecutive frames: {action_consistency:.3f}")
    summary_lines.append(f"   Frames with actions: {len(action_seqs)}")

    # ── Save ──
    print(f"\nSaving results to {out_dir}/")

    save_data = {
        "model": args.model,
        "dataset": args.dataset,
        "config": {
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "head_dim": config.head_dim,
            "hidden_dim": config.hidden_dim,
            "num_vis_tokens": config.num_vis_tokens,
            "vis_grid": config.vis_grid,
        },
        "frame_data": frame_data,
        "frame_diffs": frame_diffs,
        "analysis": {
            "head_stability": stability,
            "head_types": head_types,
            "temporal_sim": temporal_sim,
            "layer_output_sim": layer_sim,
            "frame_diff_correlations": correlations,
            "mean_pearson_r": mean_r,
            "action_consistency": action_consistency,
            "action_sequences": action_seqs,
        },
    }

    torch.save(save_data, os.path.join(out_dir, "frame_data.pt"))
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    print(f"Done. Saved frame_data.pt and summary.txt")


if __name__ == "__main__":
    main()
