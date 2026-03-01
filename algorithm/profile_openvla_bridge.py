"""
Profile OpenVLA attention patterns on real Bridge V2 robotics images.

Loads 10 diverse episodes from Bridge V2 via HuggingFace streaming,
runs OpenVLA prefill on each, and aggregates attention statistics
across real manipulation scenes.

Outputs:
  - output/openvla_profile/bridge_attention_stats.pt
  - output/openvla_profile/bridge_summary.txt
  - output/openvla_profile/bridge_per_sample.txt
"""

import os
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import av

from transformers import AutoModelForVision2Seq, AutoProcessor

MODEL_ID = "openvla/openvla-7b"
OUTPUT_DIR = "output/openvla_profile"
NUM_SAMPLES = 10
SAMPLE_STRIDE = 500  # skip between samples for diversity

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Load Bridge V2 images ───
print("Loading Bridge V2 images from HuggingFace...")

REPO_ID = "Qu3tzal/bridgev2"
# Use first 10 episodes (small IDs, all in chunk-000) — no slow metadata scan
EPISODE_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

images = []
instructions = []
for ep_id in EPISODE_IDS:
    video_file = f"videos/chunk-000/observation.images.image_0/episode_{ep_id:06d}.mp4"
    print(f"  Downloading episode {ep_id}...", end=" ", flush=True)
    video_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=video_file,
        repo_type="dataset",
    )

    # Extract a frame from the middle of the episode
    container = av.open(video_path)
    frames = list(container.decode(video=0))
    container.close()
    mid_idx = len(frames) // 2
    img = frames[mid_idx].to_image().convert("RGB")

    # Generic prompt — we skip slow metadata scan
    lang = "manipulate the object on the table"
    images.append(img)
    instructions.append(lang)
    print(f"{img.size}, frame {mid_idx}/{len(frames)}")

print(f"Loaded {len(images)} Bridge V2 images")

# ─── Load model ───
print(f"\nLoading model: {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)
model.eval()

# Token regions (after multimodal embedding concatenation)
VIS_START, VIS_END = 1, 256
NUM_VIS = 256

# ─── Profile each sample ───
all_stats = []  # list of per-sample stats dicts
per_sample_lines = []

for sample_idx in range(len(images)):
    image = images[sample_idx]
    lang = instructions[sample_idx]
    prompt = f"In: What action should the robot take to {lang}?\nOut:"

    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    text_len = inputs["input_ids"].shape[1]
    total_seq_len = 1 + NUM_VIS + (text_len - 1)

    # Hook to capture attention
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

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            attention_mask=inputs.get("attention_mask"),
            output_attentions=True,
        )

    for h in hooks:
        h.remove()

    # Analyze this sample
    actual_seq_len = attention_weights[0].shape[-1]
    TEXT_START = VIS_END + 1
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]

    sample_stats = {
        "instruction": lang,
        "seq_len": actual_seq_len,
        "layers": {},
    }

    sample_line = f"\n--- Sample {sample_idx}: '{lang}' (seq_len={actual_seq_len}) ---"
    per_sample_lines.append(sample_line)
    print(sample_line)

    for layer_idx in range(num_layers):
        attn = attention_weights[layer_idx].squeeze(0)  # [heads, seq, seq]
        last_attn = attn[:, -1, :]  # [heads, seq]

        visual_attn = last_attn[:, VIS_START:VIS_END + 1].sum(dim=-1)  # [heads]
        text_attn = last_attn[:, TEXT_START:].sum(dim=-1)
        bos_attn = last_attn[:, 0]

        visual_mean = visual_attn.mean().item()
        text_mean = text_attn.mean().item()
        bos_mean = bos_attn.mean().item()

        # Spatial analysis
        vis_spatial = last_attn[:, VIS_START:VIS_END + 1]  # [heads, 256]
        vis_avg = vis_spatial.mean(dim=0)  # [256]
        vis_2d = vis_avg.reshape(16, 16)

        # Normalized entropy
        vis_probs = vis_spatial / (vis_spatial.sum(dim=-1, keepdim=True) + 1e-10)
        vis_entropy = -(vis_probs * (vis_probs + 1e-10).log()).sum(dim=-1).mean().item()
        max_entropy = np.log(256)

        # Top-32 concentration
        topk_vals, topk_idx = vis_avg.topk(32)
        top32_conc = topk_vals.sum().item() / (vis_avg.sum().item() + 1e-10)

        # Gini coefficient
        m = vis_2d.numpy().flatten()
        sorted_m = np.sort(m)
        n = len(sorted_m)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_m) / (n * np.sum(sorted_m) + 1e-10)) - (n + 1) / n

        # Head classification
        n_visual = (visual_attn > 0.30).sum().item()
        n_text = (text_attn > 0.60).sum().item()
        n_bos = (bos_attn > 0.60).sum().item()

        sample_stats["layers"][layer_idx] = {
            "visual": visual_mean,
            "text": text_mean,
            "bos": bos_mean,
            "entropy_norm": vis_entropy / max_entropy,
            "top32_conc": top32_conc,
            "gini": gini,
            "spatial_2d": vis_2d.numpy(),
            "n_visual_heads": n_visual,
            "n_text_heads": n_text,
            "n_bos_heads": n_bos,
            "top32_indices": topk_idx.tolist(),
        }

    all_stats.append(sample_stats)

    # Print summary for this sample (selected layers)
    for l in [0, 7, 14, 24, 31]:
        s = sample_stats["layers"][l]
        line = (f"  L{l:2d}: vis={s['visual']:.3f} txt={s['text']:.3f} bos={s['bos']:.3f} "
                f"ent={s['entropy_norm']:.3f} top32={s['top32_conc']:.3f} gini={s['gini']:.3f} "
                f"Vheads={s['n_visual_heads']} Theads={s['n_text_heads']} Bheads={s['n_bos_heads']}")
        per_sample_lines.append(line)
        print(line)

# ─── Aggregate across samples ───
print("\n" + "=" * 70)
print("AGGREGATED RESULTS ACROSS ALL BRIDGE V2 SAMPLES")
print("=" * 70)

num_layers = 32
summary_lines = []
summary_lines.append(f"OpenVLA Attention Profile — Bridge V2 ({len(all_stats)} samples)")
summary_lines.append(f"=" * 60)

# Per-layer aggregates
agg = {l: {"visual": [], "text": [], "bos": [], "entropy": [], "top32": [],
           "gini": [], "v_heads": [], "t_heads": [], "b_heads": []}
        for l in range(num_layers)}

for s in all_stats:
    for l in range(num_layers):
        ls = s["layers"][l]
        agg[l]["visual"].append(ls["visual"])
        agg[l]["text"].append(ls["text"])
        agg[l]["bos"].append(ls["bos"])
        agg[l]["entropy"].append(ls["entropy_norm"])
        agg[l]["top32"].append(ls["top32_conc"])
        agg[l]["gini"].append(ls["gini"])
        agg[l]["v_heads"].append(ls["n_visual_heads"])
        agg[l]["t_heads"].append(ls["n_text_heads"])
        agg[l]["b_heads"].append(ls["n_bos_heads"])

header = (f"{'Layer':>5} | {'Visual':>12} | {'Text':>12} | {'BOS':>12} | "
          f"{'Entropy':>12} | {'Top32':>12} | {'Gini':>12} | {'V-heads':>7} | {'T-heads':>7} | {'B-heads':>7}")
summary_lines.append("")
summary_lines.append(header)
summary_lines.append("-" * len(header))
print(f"\n{header}")
print("-" * len(header))

for l in range(num_layers):
    a = agg[l]
    line = (f"{l:5d} | "
            f"{np.mean(a['visual']):5.3f}±{np.std(a['visual']):.3f} | "
            f"{np.mean(a['text']):5.3f}±{np.std(a['text']):.3f} | "
            f"{np.mean(a['bos']):5.3f}±{np.std(a['bos']):.3f} | "
            f"{np.mean(a['entropy']):5.3f}±{np.std(a['entropy']):.3f} | "
            f"{np.mean(a['top32']):5.3f}±{np.std(a['top32']):.3f} | "
            f"{np.mean(a['gini']):5.3f}±{np.std(a['gini']):.3f} | "
            f"{np.mean(a['v_heads']):5.1f}  | {np.mean(a['t_heads']):5.1f}  | {np.mean(a['b_heads']):5.1f}")
    summary_lines.append(line)
    print(line)

# Cross-sample spatial consistency: do the same patches matter across different images?
summary_lines.append("")
summary_lines.append("Cross-sample top-32 patch consistency (Jaccard between samples):")
for l in [0, 7, 14, 24, 31]:
    jaccards = []
    for i in range(len(all_stats)):
        for j in range(i + 1, len(all_stats)):
            s1 = set(all_stats[i]["layers"][l]["top32_indices"])
            s2 = set(all_stats[j]["layers"][l]["top32_indices"])
            jaccards.append(len(s1 & s2) / len(s1 | s2))
    mean_j = np.mean(jaccards) if jaccards else 0
    std_j = np.std(jaccards) if jaccards else 0
    line = f"  Layer {l:2d}: mean Jaccard = {mean_j:.3f} ± {std_j:.3f}"
    summary_lines.append(line)
    print(line)

# ─── Save ───
torch.save(all_stats, os.path.join(OUTPUT_DIR, "bridge_attention_stats.pt"))
with open(os.path.join(OUTPUT_DIR, "bridge_summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))
with open(os.path.join(OUTPUT_DIR, "bridge_per_sample.txt"), "w") as f:
    f.write("\n".join(per_sample_lines))

print(f"\nResults saved to {OUTPUT_DIR}/")
