# Focus: Temporal Head Caching for Efficient VLA Inference

Building on the [Focus HPCA 2026 paper](https://arxiv.org/abs/2512.14661), this repository extends the concentration architecture to **Vision-Language-Action (VLA)** models used in robotic manipulation.

The key insight: VLA models process temporally correlated camera frames, and attention heads specialize into visual/text/BOS/mixed types. By caching stable heads and pruning redundant spatial tokens, we achieve multiplicative speedup with minimal action prediction error.

---

## Algorithm

Three-axis reduction combining spatial token pruning, head-level temporal caching, and frame differencing:

```
Per-Timestep:
1. Frame diff → patch_change[16×16]                          [~free]
2. Keyframe? (t % K == 0) → full recompute
3. Pilot layers (0-1) dense → prune 256 → top-K tokens
4. Layers 2-31: T/B heads always cache, V heads cache if
   attended patches unchanged, M heads always recompute
5. Merge cached + fresh head outputs → MLP → next layer
```

---

## Repository Structure

* **`algorithm/`** — VLA temporal head caching algorithm, profiling, and benchmarking.
  See [algorithm/README.md](algorithm/README.md).

* **`simulator/`** — Architecture performance simulator.

* **`rtl/`** — Hardware RTL implementation (systolic array, SEC/SIC).

* **`evaluation_scripts/`** — Plotting and result-analysis utilities.

* **`3rd_party/`** — Third-party dependencies:
  * `SimplerEnv-OpenVLA/` — Simulation eval for VLA models (ManiSkill2/3)
  * `lerobot/` — HuggingFace LeRobot (Pi0-FAST eval, LIBERO/MetaWorld)
  * `CogACT/` — CogACT model code + sim policy scripts
  * `scalesim/` — GEMM performance simulator
  * `cacti/` — SRAM memory modeling
  * `DRAMsim3/` — DRAM simulation

---

## Supported VLA Models

| Model | HF ID | Backbone | Eval |
|-------|--------|----------|------|
| **OpenVLA** | `openvla/openvla-7b` | Llama 2 7B | SimplerEnv |
| **CogACT** | `CogACT/CogACT-Base` | Llama 2 7B | SimplerEnv |
| **pi-0-FAST** | `lerobot/pi0fast_base` | PaliGemma 3B | SimplerEnv / LeRobot |
| **TinyVLA** | `hz1919810/TinyVLA-*` | Pythia | — |

---

## Getting Started

### Prerequisites

* Python **3.11** (conda recommended)
* CUDA-capable GPU (**≥ 24 GB VRAM**)
* HuggingFace access token

### Installation

```bash
git clone git@github.com:5iri/Focus.git
cd Focus
git submodule init && git submodule update
```

```bash
conda create -n focus python=3.11 -y
conda activate focus

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate huggingface_hub av scipy numpy pillow
```

### Run Benchmarks

```bash
cd algorithm/

# Single model
python temporal_head_analysis.py --model openvla --dataset bridge_v2 --num_frames 20
python pipeline_simulation.py --model openvla --dataset bridge_v2

# All models
./run_benchmark.sh

# Cross-model comparison
python pipeline_simulation.py --cross_model
```

---

## Citation

```bibtex
@misc{wei2025focus,
      title={Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models},
      author={Chiyue Wei and Cong Guo and Junyao Zhang and Haoxuan Shan and Yifan Xu and Ziyue Zhang and Yudong Liu and Qinsi Wang and Changchun Zhou and Hai "Helen" Li and Yiran Chen},
      year={2025},
      eprint={2512.14661},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2512.14661},
}
```
