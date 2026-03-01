# VLA Temporal Head Caching — Algorithm

Prototype and validation of a three-axis reduction algorithm for Vision-Language-Action (VLA) model inference: **spatial token pruning**, **head-level temporal caching**, and **frame differencing** for multiplicative speedup.

## Algorithm

```
Per-Timestep:
1. Frame diff → patch_change[16x16]                          [~free]
2. Keyframe? (t % K == 0) → full recompute
3. Pilot layers (0-1) dense → prune 256 → top-K tokens
4. Layers 2-31: T/B heads always cache, V heads cache if
   attended patches unchanged, M heads always recompute
5. Merge cached + fresh head outputs → MLP → next layer
```

## Supported Models

| Model | HF ID | Backbone | Vis Tokens | Eval Framework |
|-------|--------|----------|-----------|----------------|
| **OpenVLA** | `openvla/openvla-7b` | Llama 2 7B | 256 (16x16) | SimplerEnv |
| **CogACT** | `CogACT/CogACT-Base` | Llama 2 7B | 256 | SimplerEnv |
| **OpenVLA-OFT** | `moojink/openvla-7b-oft-*` | Llama 2 7B | 256 | SimplerEnv |
| **pi-0-FAST** | `lerobot/pi0fast_base` | PaliGemma 3B | 256/cam | SimplerEnv / LeRobot |
| **TinyVLA** | `hz1919810/TinyVLA-*` | Pythia | varies | — |

## Supported Datasets

| Dataset | HF ID | Download Location |
|---------|--------|-------------------|
| **Bridge V2** | `Qu3tzal/bridgev2` | `3rd_party/datasets/bridge_v2/` |
| **DROID** | `droid_100` | `3rd_party/datasets/droid/` |
| **Fractal** | `fractal20220817_data` | `3rd_party/datasets/fractal/` |

## Files

| File | Description |
|------|-------------|
| `vla_benchmarks.py` | Model adapter registry + dataset registry |
| `temporal_head_analysis.py` | Per-frame attention data collection + temporal analysis |
| `pipeline_simulation.py` | Parameter sweep + compound reduction + accuracy evaluation |
| `run_benchmark.sh` | Orchestration script for all models/datasets |
| `profile_openvla_bridge.py` | OpenVLA attention profiling on Bridge V2 |
| `profile_openvla_deep.py` | OpenVLA deep spatial/head analysis |
| `profile_openvla.py` | OpenVLA attention profiling (synthetic input) |
| `run_openvla.py` | OpenVLA vanilla inference test |

## Quick Start

```bash
cd algorithm/

# Run a single model on Bridge V2
python temporal_head_analysis.py --model openvla --dataset bridge_v2 --episode 0 --num_frames 20
python pipeline_simulation.py --model openvla --dataset bridge_v2

# Run all models
./run_benchmark.sh

# Cross-model comparison only (after individual runs)
python pipeline_simulation.py --cross_model
```

## Outputs

```
output/temporal_analysis/
├── openvla_bridge_v2/
│   ├── frame_data.pt           # Raw per-frame attention data
│   ├── summary.txt             # Head classification, stability, correlations
│   ├── sweep_results.csv       # All parameter combinations
│   └── simulation_summary.txt  # Best config + accuracy evaluation
├── cogact_bridge_v2/
│   └── ...
└── cross_model_comparison.csv  # Side-by-side model comparison
```

## Evaluation Frameworks

Simulation-based VLA evaluation is available via submodules in `3rd_party/`:

- **SimplerEnv-OpenVLA** — ManiSkill2/3 eval for OpenVLA, CogACT, Pi0-FAST
- **LeRobot** — LIBERO + MetaWorld eval for Pi0-FAST
- **CogACT** — CogACT model code + sim policy scripts
