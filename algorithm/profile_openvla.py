"""
Profile OpenVLA attention patterns during action token generation.

Extracts attention weights across all 32 Llama2 decoder layers during the
prefill pass and analyzes how visual vs. text tokens are attended to.

OpenVLA embedding layout: [BOS, 256 visual patches, text tokens...]
The visual patches are inserted after BOS by the model's forward() method.

Outputs:
  - output/openvla_profile/attention_stats.pt  — per-layer attention statistics
  - output/openvla_profile/summary.txt         — human-readable summary
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
print(f"Loading model: {MODEL_ID}")
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

text_len = inputs["input_ids"].shape[1]
# After model's forward concatenation: [BOS(1) + visual(256) + text_after_BOS(text_len-1)]
total_seq_len = 1 + 256 + (text_len - 1)

print(f"Text input_ids length: {text_len}")
print(f"Total sequence after visual insertion: {total_seq_len}")
print(f"  Position 0: BOS")
print(f"  Positions 1-256: visual patches (16x16)")
print(f"  Positions 257-{total_seq_len-1}: text tokens ({text_len - 1} tokens)")

# --- Hook to capture attention weights from every layer ---
attention_weights = {}

def make_hook(layer_idx):
    def hook_fn(module, args, output):
        # For eager attention, output is (attn_output, attn_weights, past_kv)
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            attention_weights[layer_idx] = output[1].detach().cpu().float()
    return hook_fn

# Access the LLM decoder layers
llm_model = model.language_model.model
hooks = []
for i, layer in enumerate(llm_model.layers):
    h = layer.self_attn.register_forward_hook(make_hook(i))
    hooks.append(h)

print(f"Registered hooks on {len(hooks)} layers")

# --- Run prefill pass ---
print("Running prefill pass with output_attentions=True...")
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        pixel_values=inputs.get("pixel_values"),
        attention_mask=inputs.get("attention_mask"),
        output_attentions=True,
    )

for h in hooks:
    h.remove()

# --- Analyze attention patterns ---
print(f"\nCaptured attention from {len(attention_weights)} layers")
if len(attention_weights) == 0:
    # Fallback: check if attentions are in the output directly
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        print("Using attentions from model output instead of hooks")
        for i, attn in enumerate(outputs.attentions):
            attention_weights[i] = attn.detach().cpu().float()
        print(f"Got attention from {len(attention_weights)} layers")

if len(attention_weights) == 0:
    print("ERROR: No attention weights captured. Exiting.")
    exit(1)

# Verify actual sequence length from attention shape
sample_attn = attention_weights[0]
actual_seq_len = sample_attn.shape[-1]
print(f"Actual attention matrix size: {sample_attn.shape}")

# Token regions in the multimodal embedding
bos_pos = 0
visual_start = 1
visual_end = 256  # inclusive, 256 patches
text_start = 257
text_end = actual_seq_len - 1

stats = {}
summary_lines = []
summary_lines.append(f"OpenVLA Attention Profile")
summary_lines.append(f"========================")
summary_lines.append(f"Model: {MODEL_ID}")
summary_lines.append(f"Actual sequence length: {actual_seq_len}")
summary_lines.append(f"Visual tokens: positions {visual_start}-{visual_end} (256 tokens)")
summary_lines.append(f"Text tokens: positions {text_start}-{text_end} ({text_end - text_start + 1} tokens)")
summary_lines.append(f"")

for layer_idx in sorted(attention_weights.keys()):
    attn = attention_weights[layer_idx].squeeze(0)  # [num_heads, seq_len, seq_len]
    num_heads = attn.shape[0]

    # Attention from the LAST token (generates first action token)
    last_token_attn = attn[:, -1, :]  # [num_heads, seq_len]

    # Attention mass on each region
    visual_attn = last_token_attn[:, visual_start:visual_end+1].sum(dim=-1)  # [num_heads]
    text_attn = last_token_attn[:, text_start:].sum(dim=-1)
    bos_attn = last_token_attn[:, bos_pos]

    visual_mean = visual_attn.mean().item()
    text_mean = text_attn.mean().item()
    bos_mean = bos_attn.mean().item()

    # Spatial attention within visual tokens (16x16 grid)
    visual_spatial = last_token_attn[:, visual_start:visual_end+1]  # [num_heads, 256]
    visual_spatial_avg = visual_spatial.mean(dim=0)  # [256]
    visual_spatial_2d = visual_spatial_avg.reshape(16, 16)

    # Entropy of visual attention (higher = more uniform)
    visual_probs = visual_spatial / (visual_spatial.sum(dim=-1, keepdim=True) + 1e-10)
    visual_entropy = -(visual_probs * (visual_probs + 1e-10).log()).sum(dim=-1).mean().item()
    max_entropy = np.log(256)

    # Top-32 concentration (12.5% of patches)
    top_k = 32
    topk_vals, _ = visual_spatial_avg.topk(top_k)
    topk_concentration = topk_vals.sum().item() / (visual_spatial_avg.sum().item() + 1e-10)

    layer_stats = {
        "visual_attn_fraction": visual_mean,
        "text_attn_fraction": text_mean,
        "bos_attn_fraction": bos_mean,
        "visual_entropy": visual_entropy,
        "visual_entropy_normalized": visual_entropy / max_entropy,
        "top32_concentration": topk_concentration,
        "visual_spatial_2d": visual_spatial_2d,
        "per_head_visual_attn": visual_attn,
    }
    stats[layer_idx] = layer_stats

    line = (
        f"Layer {layer_idx:2d}: "
        f"visual={visual_mean:.3f}  text={text_mean:.3f}  bos={bos_mean:.3f}  "
        f"entropy={visual_entropy/max_entropy:.3f}  top32={topk_concentration:.3f}"
    )
    summary_lines.append(line)
    print(line)

# Overall summary
summary_lines.append("")
summary_lines.append("--- Aggregated ---")
all_visual = [s["visual_attn_fraction"] for s in stats.values()]
all_entropy = [s["visual_entropy_normalized"] for s in stats.values()]
all_top32 = [s["top32_concentration"] for s in stats.values()]

agg_lines = [
    f"Avg visual attention fraction: {np.mean(all_visual):.3f} (std={np.std(all_visual):.3f})",
    f"Avg normalized entropy: {np.mean(all_entropy):.3f} (std={np.std(all_entropy):.3f})",
    f"Avg top-32 concentration: {np.mean(all_top32):.3f} (std={np.std(all_top32):.3f})",
    f"",
    f"Layers with highest visual attention: {sorted(range(len(all_visual)), key=lambda i: all_visual[i], reverse=True)[:5]}",
    f"Layers with lowest entropy (most concentrated): {sorted(range(len(all_entropy)), key=lambda i: all_entropy[i])[:5]}",
    f"Layers with highest top-32 concentration: {sorted(range(len(all_top32)), key=lambda i: all_top32[i], reverse=True)[:5]}",
]
for line in agg_lines:
    summary_lines.append(line)
    print(line)

# --- Save ---
torch.save(stats, os.path.join(OUTPUT_DIR, "attention_stats.pt"))
with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))

print(f"\nSaved to {OUTPUT_DIR}/")
