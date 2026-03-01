"""
Deep analysis of OpenVLA spatial attention patterns and per-head specialization.

Loads the full attention weights (re-extracted at full precision) and analyzes:
1. Spatial heatmaps: which 16x16 patches are attended to, per layer
2. Per-head specialization: do heads split into visual/text/BOS specialists?
3. Head clustering: groups of heads with similar attention patterns
4. Cross-layer evolution: how spatial focus shifts across layers
"""

import os
import torch
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

MODEL_ID = "openvla/openvla-7b"
OUTPUT_DIR = "output/openvla_profile"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load model ---
print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)
model.eval()

# --- Prepare input ---
image = Image.fromarray(
    np.tile(np.linspace(0, 255, 224, dtype=np.uint8), (224, 3)).reshape(224, 224, 3).astype(np.uint8)
)
prompt = "In: What action should the robot take to pick up the red block?\nOut:"
inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)

# --- Capture full attention weights ---
attention_weights = {}

def make_hook(layer_idx):
    def hook_fn(module, args, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            attention_weights[layer_idx] = output[1].detach().cpu().float()
    return hook_fn

llm_model = model.language_model.model
hooks = []
for i, layer in enumerate(llm_model.layers):
    h = layer.self_attn.register_forward_hook(make_hook(i))
    hooks.append(h)

print("Running prefill...")
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        attention_mask=inputs.get("attention_mask"),
        output_attentions=True,
    )

for h in hooks:
    h.remove()

seq_len = attention_weights[0].shape[-1]
num_layers = len(attention_weights)
num_heads = attention_weights[0].shape[1]

# Token regions
VIS_START, VIS_END = 1, 256
TEXT_START = 257

print(f"Sequence length: {seq_len}, Layers: {num_layers}, Heads: {num_heads}")
print()

# ============================================================
# 1. SPATIAL HEATMAPS — per layer, head-averaged
# ============================================================
print("=" * 70)
print("1. SPATIAL ATTENTION HEATMAPS (16x16 grid, last token -> visual patches)")
print("=" * 70)

spatial_maps = {}  # layer -> [16, 16] numpy array

for layer_idx in range(num_layers):
    attn = attention_weights[layer_idx].squeeze(0)  # [heads, seq, seq]
    # Last token attending to visual patches
    vis_attn = attn[:, -1, VIS_START:VIS_END+1]  # [heads, 256]
    # Normalize per head so we see relative spatial distribution
    vis_attn_norm = vis_attn / (vis_attn.sum(dim=-1, keepdim=True) + 1e-10)
    avg_spatial = vis_attn_norm.mean(dim=0).reshape(16, 16).numpy()
    spatial_maps[layer_idx] = avg_spatial

# Print selected layers as ASCII heatmaps
def ascii_heatmap(arr, title, width=16):
    """Print a 16x16 array as an ASCII heatmap."""
    chars = " ░▒▓█"
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-10:
        normalized = np.zeros_like(arr)
    else:
        normalized = (arr - vmin) / (vmax - vmin)
    print(f"\n  {title}")
    print(f"  (min={vmin:.5f}, max={vmax:.5f})")
    print("  " + "─" * width)
    for row in range(arr.shape[0]):
        line = "  │"
        for col in range(arr.shape[1]):
            idx = min(int(normalized[row, col] * (len(chars) - 1)), len(chars) - 1)
            line += chars[idx]
        line += "│"
        print(line)
    print("  " + "─" * width)

for layer_idx in [0, 1, 7, 14, 20, 25, 31]:
    ascii_heatmap(spatial_maps[layer_idx], f"Layer {layer_idx}")

# Spatial concentration metrics per layer
print("\n  Layer-by-layer spatial concentration:")
print("  Layer | Gini coeff | Max/Mean | Active patches (>2x mean)")
print("  " + "-" * 60)
for layer_idx in range(num_layers):
    m = spatial_maps[layer_idx].flatten()
    # Gini coefficient
    sorted_m = np.sort(m)
    n = len(sorted_m)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_m) / (n * np.sum(sorted_m))) - (n + 1) / n
    # Max/mean ratio
    max_mean = m.max() / (m.mean() + 1e-10)
    # Active patches
    active = (m > 2 * m.mean()).sum()
    print(f"  {layer_idx:5d} | {gini:10.3f} | {max_mean:8.2f} | {active:3d}/256")


# ============================================================
# 2. PER-HEAD SPECIALIZATION
# ============================================================
print()
print("=" * 70)
print("2. PER-HEAD SPECIALIZATION")
print("=" * 70)

# For each head in each layer, compute: % attention to visual, text, BOS
head_profiles = np.zeros((num_layers, num_heads, 3))  # [layers, heads, (visual, text, bos)]

for layer_idx in range(num_layers):
    attn = attention_weights[layer_idx].squeeze(0)  # [heads, seq, seq]
    last_attn = attn[:, -1, :]  # [heads, seq]

    head_profiles[layer_idx, :, 0] = last_attn[:, VIS_START:VIS_END+1].sum(dim=-1).numpy()  # visual
    head_profiles[layer_idx, :, 1] = last_attn[:, TEXT_START:].sum(dim=-1).numpy()  # text
    head_profiles[layer_idx, :, 2] = last_attn[:, 0].numpy()  # BOS

# Classify heads
print("\n  Head classification (last token's attention distribution):")
print("  Classification: V=visual(>30%), T=text(>60%), B=BOS(>60%), M=mixed")
print()

for layer_idx in range(num_layers):
    head_labels = []
    for h in range(num_heads):
        v, t, b = head_profiles[layer_idx, h]
        if v > 0.30:
            head_labels.append("V")
        elif t > 0.60:
            head_labels.append("T")
        elif b > 0.60:
            head_labels.append("B")
        else:
            head_labels.append("M")
    counts = {l: head_labels.count(l) for l in ["V", "T", "B", "M"]}
    print(f"  Layer {layer_idx:2d}: {' '.join(head_labels)}  | V={counts['V']:2d} T={counts['T']:2d} B={counts['B']:2d} M={counts['M']:2d}")

# ============================================================
# 3. PER-HEAD SPATIAL PATTERNS (for visual-specialist heads)
# ============================================================
print()
print("=" * 70)
print("3. VISUAL-SPECIALIST HEAD SPATIAL PATTERNS")
print("=" * 70)

# Find the heads with highest visual attention in interesting layers
for layer_idx in [0, 14, 24, 31]:
    attn = attention_weights[layer_idx].squeeze(0)  # [heads, seq, seq]
    vis_attn = attn[:, -1, VIS_START:VIS_END+1]  # [heads, 256]
    vis_total = vis_attn.sum(dim=-1)  # [heads]

    # Top 3 visual heads
    top_heads = vis_total.topk(3).indices.tolist()
    print(f"\n  Layer {layer_idx} — Top visual heads: {top_heads}")
    print(f"  (visual attention fractions: {[f'{vis_total[h]:.3f}' for h in top_heads]})")

    for h in top_heads:
        head_vis = vis_attn[h]  # [256]
        head_vis_norm = head_vis / (head_vis.sum() + 1e-10)
        spatial = head_vis_norm.reshape(16, 16).numpy()

        # Find peak location
        peak_idx = head_vis_norm.argmax().item()
        peak_row, peak_col = peak_idx // 16, peak_idx % 16

        # Quadrant distribution
        q_tl = spatial[:8, :8].sum()
        q_tr = spatial[:8, 8:].sum()
        q_bl = spatial[8:, :8].sum()
        q_br = spatial[8:, 8:].sum()

        # Row/column marginals
        row_marginal = spatial.sum(axis=1)
        col_marginal = spatial.sum(axis=0)
        row_peak = row_marginal.argmax()
        col_peak = col_marginal.argmax()

        print(f"    Head {h:2d}: peak=({peak_row},{peak_col})  "
              f"quadrants=[TL={q_tl:.2f} TR={q_tr:.2f} BL={q_bl:.2f} BR={q_br:.2f}]  "
              f"row_peak={row_peak} col_peak={col_peak}")
        ascii_heatmap(spatial, f"Layer {layer_idx}, Head {h}")


# ============================================================
# 4. CROSS-LAYER EVOLUTION
# ============================================================
print()
print("=" * 70)
print("4. CROSS-LAYER SPATIAL EVOLUTION")
print("=" * 70)

# How does the set of "important" patches change across layers?
# Use head-averaged, normalized spatial maps
important_patches = {}  # layer -> set of top-32 patch indices
for layer_idx in range(num_layers):
    m = spatial_maps[layer_idx].flatten()
    top32 = np.argsort(m)[-32:]
    important_patches[layer_idx] = set(top32.tolist())

print("\n  Overlap of top-32 patches between adjacent layers:")
print("  Layer pair | Jaccard similarity | Shared patches")
print("  " + "-" * 55)
for i in range(num_layers - 1):
    s1, s2 = important_patches[i], important_patches[i+1]
    jaccard = len(s1 & s2) / len(s1 | s2)
    shared = len(s1 & s2)
    print(f"  {i:2d} -> {i+1:2d}   | {jaccard:18.3f} | {shared:2d}/32")

# Overlap with early vs late layers
print("\n  Cross-phase overlap (top-32 patches):")
early = important_patches[0]
for layer_idx in [7, 14, 20, 25, 31]:
    s = important_patches[layer_idx]
    jaccard = len(early & s) / len(early | s)
    shared = len(early & s)
    print(f"  Layer 0 vs Layer {layer_idx:2d}: Jaccard={jaccard:.3f}, shared={shared}/32")

# Row/column marginal evolution
print("\n  Row attention centroid (0=top, 15=bottom) by layer:")
centroids = []
for layer_idx in range(num_layers):
    m = spatial_maps[layer_idx]
    row_marginal = m.sum(axis=1)
    row_marginal /= row_marginal.sum() + 1e-10
    centroid = (row_marginal * np.arange(16)).sum()
    centroids.append(centroid)
    if layer_idx % 4 == 0 or layer_idx == 31:
        col_marginal = m.sum(axis=0)
        col_marginal /= col_marginal.sum() + 1e-10
        col_centroid = (col_marginal * np.arange(16)).sum()
        print(f"  Layer {layer_idx:2d}: row_centroid={centroid:.2f}, col_centroid={col_centroid:.2f}")


# ============================================================
# 5. SAVE DETAILED RESULTS
# ============================================================
results = {
    "spatial_maps": spatial_maps,                   # layer -> [16,16] numpy
    "head_profiles": head_profiles,                 # [layers, heads, 3]
    "important_patches": important_patches,         # layer -> set
    "full_attention_weights": attention_weights,     # layer -> [1, heads, seq, seq] tensor
}
torch.save(results, os.path.join(OUTPUT_DIR, "deep_profile.pt"))
print(f"\nFull results saved to {OUTPUT_DIR}/deep_profile.pt")
