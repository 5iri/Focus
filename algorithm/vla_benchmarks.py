"""
VLA Model Adapter Registry + Dataset Registry.

Provides a model-agnostic interface for:
  - Loading VLA models and processors
  - Accessing LLM decoder layers and attention modules
  - Identifying visual/text token regions
  - Running inference (prefill + generate)

Also provides a dataset-agnostic interface for:
  - Loading robotics episode data (image sequences + language instructions)

Supported models:
  - OpenVLA (openvla-7b, Llama 2 backbone)
  - CogACT (CogACT-Base, Llama 2 backbone)
  - OpenVLA-OFT (openvla-7b-oft, Llama 2 backbone)
  - pi-0-FAST (pi0fast_base, PaliGemma/Gemma backbone)
  - TinyVLA (TinyVLA, Pythia backbone)

Supported datasets:
  - Bridge V2 (Qu3tzal/bridgev2)
  - DROID (droid_100)
  - Fractal (fractal20220817_data)
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import av
import numpy as np
import torch
from PIL import Image

# transformers v5 renamed AutoModelForVision2Seq → AutoModelForImageTextToText
try:
    from transformers import AutoModelForVision2Seq as _VLAAutoModel
except ImportError:
    from transformers import AutoModelForImageTextToText as _VLAAutoModel

# All datasets download into 3rd_party/datasets/ relative to repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY_DIR = _REPO_ROOT / "3rd_party" / "datasets"


@dataclass
class ModelConfig:
    """Static metadata about a VLA model."""
    name: str
    hf_id: str
    backbone: str
    num_layers: int
    num_heads: int
    head_dim: int
    hidden_dim: int
    num_vis_tokens: int
    vis_grid: Tuple[int, int]  # (rows, cols) for spatial layout


@dataclass
class DatasetConfig:
    """Static metadata about a robotics dataset."""
    name: str
    hf_id: str
    video_pattern: str  # pattern with {episode_id} placeholder
    default_instruction: str
    num_episodes: int = 10
    repo_type: str = "dataset"


# ─── Dataset Registry ───────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "bridge_v2": DatasetConfig(
        name="bridge_v2",
        hf_id="Qu3tzal/bridgev2",
        video_pattern="videos/chunk-000/observation.images.image_0/episode_{episode_id:06d}.mp4",
        default_instruction="manipulate the object on the table",
        num_episodes=10,
    ),
    "droid": DatasetConfig(
        name="droid",
        hf_id="droid_100",
        video_pattern="videos/episode_{episode_id:06d}.mp4",
        default_instruction="perform the manipulation task",
        num_episodes=10,
    ),
    "fractal": DatasetConfig(
        name="fractal",
        hf_id="fractal20220817_data",
        video_pattern="videos/episode_{episode_id:06d}.mp4",
        default_instruction="complete the manipulation task",
        num_episodes=10,
    ),
}


def load_episode_frames(dataset_name: str, episode_id: int,
                        num_frames: int = 20, stride: int = 1) -> List[Image.Image]:
    """Load frames from a dataset episode.

    Downloads videos from HuggingFace into 3rd_party/datasets/{dataset_name}/
    so they persist across runs. Falls back to placeholder frames if the
    dataset is unavailable.

    Returns:
        List of PIL Images (RGB).
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")

    cfg = DATASET_REGISTRY[dataset_name]

    try:
        return _load_hf_video_frames(cfg, episode_id, num_frames)
    except Exception as e:
        print(f"  Warning: Could not load {dataset_name} episode {episode_id}: {e}")
        print(f"  Generating placeholder frames for offline simulation.")
        return _generate_placeholder_frames(num_frames)


def _load_hf_video_frames(cfg: DatasetConfig, episode_id: int,
                          num_frames: int) -> List[Image.Image]:
    """Download video into 3rd_party/datasets/{name}/ and extract frames."""
    from huggingface_hub import hf_hub_download

    cache_dir = THIRD_PARTY_DIR / cfg.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    video_file = cfg.video_pattern.format(episode_id=episode_id)
    print(f"  Loading {cfg.name} episode {episode_id} "
          f"(cache: {cache_dir})...", end=" ", flush=True)
    video_path = hf_hub_download(
        repo_id=cfg.hf_id,
        filename=video_file,
        repo_type=cfg.repo_type,
        cache_dir=str(cache_dir),
    )

    container = av.open(video_path)
    all_frames = list(container.decode(video=0))
    container.close()

    # Sample num_frames evenly spaced frames
    total = len(all_frames)
    if num_frames >= total:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

    frames = [all_frames[i].to_image().convert("RGB") for i in indices]
    print(f"{len(frames)} frames extracted from {total} total")
    return frames


def _generate_placeholder_frames(num_frames: int, size: int = 224) -> List[Image.Image]:
    """Generate synthetic frames with temporal variation for offline testing."""
    frames = []
    for t in range(num_frames):
        # Smooth temporal gradient so frame-diff analysis works meaningfully
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        # Base scene
        arr[:, :, 0] = 100  # R channel
        arr[:, :, 1] = 120  # G channel
        arr[:, :, 2] = 80   # B channel
        # Moving blob to simulate object motion
        cx = int(size * (0.3 + 0.4 * t / max(num_frames - 1, 1)))
        cy = int(size * 0.5)
        rr, cc = np.ogrid[:size, :size]
        mask = ((rr - cy) ** 2 + (cc - cx) ** 2) < (size // 8) ** 2
        arr[mask, 0] = 200
        arr[mask, 1] = 50
        arr[mask, 2] = 50
        frames.append(Image.fromarray(arr))
    return frames


def get_dataset_instruction(dataset_name: str) -> str:
    """Get default instruction for a dataset."""
    return DATASET_REGISTRY[dataset_name].default_instruction


# ─── Base Model Adapter ─────────────────────────────────────────────────────

class VLAAdapter(ABC):
    """Base adapter for VLA models."""

    @abstractmethod
    def get_config(self) -> ModelConfig:
        """Return static model configuration."""
        ...

    @abstractmethod
    def load_model(self, device: str = "auto") -> Tuple:
        """Load model and processor. Returns (model, processor)."""
        ...

    @abstractmethod
    def get_llm_layers(self, model) -> list:
        """Return list of decoder layers."""
        ...

    @abstractmethod
    def get_attn_module(self, layer) -> torch.nn.Module:
        """Return the attention module from a decoder layer."""
        ...

    @abstractmethod
    def get_visual_token_range(self) -> Tuple[int, int, int]:
        """Return (vis_start, vis_end, text_start) token indices."""
        ...

    @abstractmethod
    def format_prompt(self, instruction: str) -> str:
        """Format a language instruction into model-specific prompt."""
        ...

    @abstractmethod
    def run_prefill(self, model, processor, image: Image.Image,
                    prompt: str) -> dict:
        """Run prefill (forward pass) and return inputs dict."""
        ...

    @abstractmethod
    def run_generate(self, model, processor, image: Image.Image,
                     prompt: str, max_new_tokens: int = 7) -> Tuple[list, str]:
        """Run generation. Returns (token_ids, decoded_text)."""
        ...

    def get_forward_kwargs(self, inputs: dict) -> dict:
        """Extra kwargs to pass to model() during forward calls.

        Override in subclasses that need special arguments (e.g. OFT needs
        fake labels for action mask computation).
        """
        return {}


# ─── OpenVLA Adapter ────────────────────────────────────────────────────────

class OpenVLAAdapter(VLAAdapter):
    """Adapter for OpenVLA (openvla/openvla-7b) — Llama 2 7B backbone."""

    def __init__(self, hf_id: str = "openvla/openvla-7b"):
        self.hf_id = hf_id

    def get_config(self) -> ModelConfig:
        return ModelConfig(
            name="openvla", hf_id=self.hf_id, backbone="llama2",
            num_layers=32, num_heads=32, head_dim=128, hidden_dim=4096,
            num_vis_tokens=256, vis_grid=(16, 16),
        )

    def load_model(self, device="auto"):
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)
        model = _VLAAutoModel.from_pretrained(
            self.hf_id, torch_dtype=torch.bfloat16, device_map=device,
            trust_remote_code=True, attn_implementation="eager",
        )
        model.eval()
        return model, processor

    def get_llm_layers(self, model) -> list:
        return list(model.language_model.model.layers)

    def get_attn_module(self, layer):
        return layer.self_attn

    def get_visual_token_range(self):
        return (1, 256, 257)  # BOS at 0, 256 vis tokens, text after

    def format_prompt(self, instruction: str) -> str:
        return f"In: What action should the robot take to {instruction}?\nOut:"

    def run_prefill(self, model, processor, image, prompt):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                attention_mask=inputs.get("attention_mask"),
                output_attentions=True,
            )
        return inputs

    def run_generate(self, model, processor, image, prompt, max_new_tokens=7):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                        do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        decoded = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated[0].tolist(), decoded


# ─── CogACT Adapter ─────────────────────────────────────────────────────────

class CogACTAdapter(VLAAdapter):
    """Adapter for CogACT (CogACT/CogACT-Base) — Llama 2 7B backbone.

    CogACT wraps a PrismaticVLM (identical to OpenVLA's architecture) plus
    a DiT diffusion action head.  It requires the ``prismatic`` package (from
    the openvla repo) and the CogACT repo on sys.path.

    Install prismatic::

        git clone https://github.com/openvla/openvla.git 3rd_party/openvla
        # Then this adapter adds it to sys.path automatically.

    The CogACT submodule is already at 3rd_party/CogACT.
    """

    def __init__(self, hf_id: str = "CogACT/CogACT-Base"):
        self.hf_id = hf_id
        self._processor = None  # filled by load_model

    def get_config(self) -> ModelConfig:
        return ModelConfig(
            name="cogact", hf_id=self.hf_id, backbone="llama2",
            num_layers=32, num_heads=32, head_dim=128, hidden_dim=4096,
            num_vis_tokens=256, vis_grid=(16, 16),
        )

    def _ensure_deps(self):
        """Add openvla (for prismatic) and CogACT to sys.path."""
        import sys
        openvla_dir = _REPO_ROOT / "3rd_party" / "openvla"
        cogact_dir = _REPO_ROOT / "3rd_party" / "CogACT"
        for p in [str(openvla_dir), str(cogact_dir)]:
            if p not in sys.path:
                sys.path.insert(0, p)

    def load_model(self, device="auto"):
        self._ensure_deps()
        try:
            from vla import load_vla
        except ImportError as e:
            raise ImportError(
                f"CogACT requires the prismatic package. Clone the OpenVLA repo:\n"
                f"  git clone https://github.com/openvla/openvla.git "
                f"{_REPO_ROOT / '3rd_party' / 'openvla'}\n"
                f"Original error: {e}"
            ) from e

        dev = "cuda:0" if device == "auto" and torch.cuda.is_available() else device
        model = load_vla(
            self.hf_id,
            load_for_training=False,
            action_model_type="DiT-B",
            future_action_window_size=15,
        )
        model = model.to(dev).eval()

        # Store image transform and tokenizer as a lightweight processor proxy
        self._processor = _CogACTProcessor(
            image_transform=model.vlm.vision_backbone.image_transform,
            tokenizer=model.vlm.llm_backbone.tokenizer,
        )
        return model, self._processor

    def get_llm_layers(self, model) -> list:
        return list(model.vlm.llm_backbone.llm.model.layers)

    def get_attn_module(self, layer):
        return layer.self_attn

    def get_visual_token_range(self):
        return (1, 256, 257)

    def format_prompt(self, instruction: str) -> str:
        return f"In: What action should the robot take to {instruction}?\nOut:"

    def run_prefill(self, model, processor, image, prompt):
        inputs = processor(prompt, image).to(model.vlm.device, dtype=torch.bfloat16)
        with torch.no_grad():
            model.vlm(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                attention_mask=inputs.get("attention_mask"),
                output_attentions=True,
            )
        return inputs

    def run_generate(self, model, processor, image, prompt, max_new_tokens=7):
        # CogACT uses diffusion for actions, but for token comparison we use
        # the VLM's autoregressive generation.
        inputs = processor(prompt, image).to(model.vlm.device, dtype=torch.bfloat16)
        from transformers import GenerationConfig
        with torch.no_grad():
            output_ids = model.vlm.llm_backbone.llm.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        decoded = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated[0].tolist(), decoded


class _CogACTProcessor:
    """Lightweight processor wrapper for CogACT (mimics OpenVLA processor API)."""

    def __init__(self, image_transform, tokenizer):
        self.image_transform = image_transform
        self.tokenizer = tokenizer

    def __call__(self, text, image):
        input_ids = self.tokenizer(text, truncation=True, return_tensors="pt").input_ids
        pixel_values = self.image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.unsqueeze(0)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v.unsqueeze(0) for k, v in pixel_values.items()}
        return _BatchEncoding({
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": torch.ones_like(input_ids),
        })


class _BatchEncoding(dict):
    """Minimal dict that supports .to(device, dtype)."""
    def to(self, device, dtype=None):
        out = {}
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device, dtype=dtype) if dtype and v.is_floating_point() else v.to(device)
            elif isinstance(v, dict):
                out[k] = {kk: vv.to(device, dtype=dtype) if dtype and vv.is_floating_point() else vv.to(device)
                          for kk, vv in v.items()}
            else:
                out[k] = v
        return _BatchEncoding(out)

    def get(self, key, default=None):
        return super().get(key, default)


# ─── OpenVLA-OFT Adapter ────────────────────────────────────────────────────

class OpenVLAOFTAdapter(OpenVLAAdapter):
    """Adapter for OpenVLA-OFT — same as OpenVLA but different checkpoint.

    OFT's forward() always calls _process_action_masks(labels), so we must
    provide fake labels (all IGNORE_INDEX) even during inference.
    """

    IGNORE_INDEX = -100

    def __init__(self, hf_id: str = "moojink/openvla-7b-oft-finetuned-libero-spatial"):
        super().__init__(hf_id=hf_id)

    def get_config(self) -> ModelConfig:
        cfg = super().get_config()
        cfg.name = "openvla_oft"
        cfg.hf_id = self.hf_id
        return cfg

    def get_forward_kwargs(self, inputs: dict) -> dict:
        """OFT needs fake labels for action mask computation."""
        ids = inputs["input_ids"]
        labels = torch.full_like(ids, self.IGNORE_INDEX)
        return {"labels": labels}

    def run_generate(self, model, processor, image, prompt, max_new_tokens=7):
        """OFT can't use generate() directly — use forward + argmax."""
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        extra = self.get_forward_kwargs(inputs)
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                attention_mask=inputs.get("attention_mask"),
                **extra,
            )
        # Take argmax of last-position logits as "action tokens"
        logits = out.logits[:, -1, :]
        action_ids = []
        for _ in range(max_new_tokens):
            tok = logits.argmax(dim=-1).item()
            action_ids.append(tok)
        decoded = processor.tokenizer.decode(action_ids, skip_special_tokens=True)
        return action_ids, decoded


# ─── Pi0-FAST Adapter ───────────────────────────────────────────────────────

class Pi0FastAdapter(VLAAdapter):
    """Adapter for pi-0-FAST (PaliGemma 3B / Gemma backbone)."""

    def __init__(self, hf_id: str = "lerobot/pi0fast_base"):
        self.hf_id = hf_id

    def get_config(self) -> ModelConfig:
        return ModelConfig(
            name="pi0fast", hf_id=self.hf_id, backbone="gemma",
            num_layers=18, num_heads=8, head_dim=256, hidden_dim=2048,
            num_vis_tokens=256, vis_grid=(16, 16),
        )

    def load_model(self, device="auto"):
        # pi-0-FAST uses LeRobot API — try direct HF loading first
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)
        model = _VLAAutoModel.from_pretrained(
            self.hf_id, torch_dtype=torch.bfloat16, device_map=device,
            trust_remote_code=True, attn_implementation="eager",
        )
        model.eval()
        return model, processor

    def get_llm_layers(self, model) -> list:
        # PaliGemma: model.language_model.model.layers
        if hasattr(model, 'language_model'):
            return list(model.language_model.model.layers)
        # Gemma direct
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return list(model.model.layers)
        raise AttributeError(f"Cannot find decoder layers in {type(model)}")

    def get_attn_module(self, layer):
        return layer.self_attn

    def get_visual_token_range(self):
        return (1, 256, 257)

    def format_prompt(self, instruction: str) -> str:
        return instruction

    def run_prefill(self, model, processor, image, prompt):
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs, output_attentions=True)
        return inputs

    def run_generate(self, model, processor, image, prompt, max_new_tokens=7):
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                        do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        decoded = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated[0].tolist(), decoded


# ─── TinyVLA Adapter ────────────────────────────────────────────────────────

class TinyVLAAdapter(VLAAdapter):
    """Adapter for TinyVLA (Pythia backbone, 0.4-1.3B)."""

    def __init__(self, hf_id: str = "hz1919810/TinyVLA-S"):
        self.hf_id = hf_id

    def get_config(self) -> ModelConfig:
        return ModelConfig(
            name="tinyvla", hf_id=self.hf_id, backbone="pythia",
            num_layers=24, num_heads=16, head_dim=128, hidden_dim=2048,
            num_vis_tokens=256, vis_grid=(16, 16),
        )

    def load_model(self, device="auto"):
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(self.hf_id, trust_remote_code=True)
        model = _VLAAutoModel.from_pretrained(
            self.hf_id, torch_dtype=torch.bfloat16, device_map=device,
            trust_remote_code=True, attn_implementation="eager",
        )
        model.eval()
        return model, processor

    def get_llm_layers(self, model) -> list:
        # Pythia: model.gpt_neox.layers or model.model.layers
        if hasattr(model, 'language_model'):
            lm = model.language_model
            if hasattr(lm, 'model') and hasattr(lm.model, 'layers'):
                return list(lm.model.layers)
            if hasattr(lm, 'gpt_neox') and hasattr(lm.gpt_neox, 'layers'):
                return list(lm.gpt_neox.layers)
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return list(model.model.layers)
        raise AttributeError(f"Cannot find decoder layers in {type(model)}")

    def get_attn_module(self, layer):
        if hasattr(layer, 'self_attn'):
            return layer.self_attn
        if hasattr(layer, 'attention'):
            return layer.attention
        raise AttributeError(f"Cannot find attention in {type(layer)}")

    def get_visual_token_range(self):
        return (1, 256, 257)

    def format_prompt(self, instruction: str) -> str:
        return f"In: What action should the robot take to {instruction}?\nOut:"

    def run_prefill(self, model, processor, image, prompt):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                attention_mask=inputs.get("attention_mask"),
                output_attentions=True,
            )
        return inputs

    def run_generate(self, model, processor, image, prompt, max_new_tokens=7):
        inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                        do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        decoded = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated[0].tolist(), decoded


# ─── Registry ───────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "openvla": OpenVLAAdapter,
    "cogact": CogACTAdapter,
    "openvla_oft": OpenVLAOFTAdapter,
    "pi0fast": Pi0FastAdapter,
    "tinyvla": TinyVLAAdapter,
}


def get_adapter(model_name: str, **kwargs) -> VLAAdapter:
    """Instantiate a VLA adapter by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)


def list_models() -> list:
    """Return list of available model names."""
    return list(MODEL_REGISTRY.keys())


def list_datasets() -> list:
    """Return list of available dataset names."""
    return list(DATASET_REGISTRY.keys())
