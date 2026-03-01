#!/bin/bash
# Run temporal head caching benchmarks across all models and datasets.
#
# Usage:
#   ./run_benchmark.sh                    # Run all models on all datasets
#   ./run_benchmark.sh openvla            # Run only OpenVLA
#   ./run_benchmark.sh openvla bridge_v2  # Run OpenVLA on Bridge V2 only
#
# Output:
#   output/temporal_analysis/{model}_{dataset}/frame_data.pt
#   output/temporal_analysis/{model}_{dataset}/sweep_results.csv
#   output/temporal_analysis/cross_model_comparison.csv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default: all models in implementation priority order
ALL_MODELS="${1:-openvla cogact pi0fast tinyvla}"
ALL_DATASETS="${2:-bridge_v2}"

NUM_FRAMES="${NUM_FRAMES:-20}"
EPISODE="${EPISODE:-0}"

echo "============================================"
echo "VLA Temporal Head Caching Benchmark"
echo "============================================"
echo "Models:   $ALL_MODELS"
echo "Datasets: $ALL_DATASETS"
echo "Frames:   $NUM_FRAMES"
echo "Episode:  $EPISODE"
echo "============================================"
echo ""

FAILED=""
SUCCEEDED=""

for model in $ALL_MODELS; do
    for dataset in $ALL_DATASETS; do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Model: $model | Dataset: $dataset"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Step 1: Data collection + analysis
        echo "[1/2] Running temporal_head_analysis.py..."
        if python temporal_head_analysis.py \
            --model "$model" \
            --dataset "$dataset" \
            --episode "$EPISODE" \
            --num_frames "$NUM_FRAMES"; then
            echo "[1/2] Data collection complete."
        else
            echo "[1/2] FAILED for $model on $dataset. Skipping simulation."
            FAILED="$FAILED ${model}_${dataset}"
            continue
        fi

        # Step 2: Pipeline simulation
        echo "[2/2] Running pipeline_simulation.py..."
        if python pipeline_simulation.py \
            --model "$model" \
            --dataset "$dataset"; then
            echo "[2/2] Simulation complete."
            SUCCEEDED="$SUCCEEDED ${model}_${dataset}"
        else
            echo "[2/2] FAILED for $model on $dataset."
            FAILED="$FAILED ${model}_${dataset}"
        fi
    done
done

# Step 3: Cross-model comparison
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Cross-Model Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python pipeline_simulation.py --cross_model

echo ""
echo "============================================"
echo "BENCHMARK COMPLETE"
echo "============================================"
if [ -n "$SUCCEEDED" ]; then
    echo "Succeeded:$SUCCEEDED"
fi
if [ -n "$FAILED" ]; then
    echo "Failed:$FAILED"
fi
echo ""
echo "Results: output/temporal_analysis/"
echo "  Per-model:    {model}_{dataset}/frame_data.pt, sweep_results.csv"
echo "  Comparison:   cross_model_comparison.csv"
