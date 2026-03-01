#!/bin/bash
# Run temporal head caching benchmarks across all models and datasets.
#
# Usage:
#   ./run_benchmark.sh                    # Run all models on all datasets
#   ./run_benchmark.sh openvla            # Run only OpenVLA
#   ./run_benchmark.sh openvla bridge_v2  # Run OpenVLA on Bridge V2 only
#
# Output:
#   output/temporal_results/{model}_{dataset}/   — end-to-end algorithm results
#   output/temporal_analysis/{model}_{dataset}/  — offline analysis + sweeps
#   output/temporal_analysis/cross_model_comparison.csv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ALL_MODELS="${1:-openvla openvla_oft}"
ALL_DATASETS="${2:-bridge_v2}"

NUM_FRAMES="${NUM_FRAMES:-20}"
EPISODE="${EPISODE:-0}"
SPATIAL_K="${SPATIAL_K:-32}"
KEYFRAME="${KEYFRAME:-5}"
THRESHOLD="${THRESHOLD:-0.1}"

echo "============================================"
echo "VLA Temporal Head Caching Benchmark"
echo "============================================"
echo "Models:     $ALL_MODELS"
echo "Datasets:   $ALL_DATASETS"
echo "Frames:     $NUM_FRAMES"
echo "spatial_K:  $SPATIAL_K"
echo "keyframe:   $KEYFRAME"
echo "threshold:  $THRESHOLD"
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

        # Step 1: End-to-end temporal caching (the actual algorithm)
        echo "[1/3] Running run_temporal.py (algorithm)..."
        if python run_temporal.py \
            --model "$model" \
            --dataset "$dataset" \
            --episode "$EPISODE" \
            --num_frames "$NUM_FRAMES" \
            --spatial_K "$SPATIAL_K" \
            --keyframe_interval "$KEYFRAME" \
            --cache_threshold "$THRESHOLD"; then
            echo "[1/3] Algorithm run complete."
        else
            echo "[1/3] FAILED for $model on $dataset."
            FAILED="$FAILED ${model}_${dataset}"
            continue
        fi

        # Step 2: Offline analysis (head stability, correlations)
        echo "[2/3] Running temporal_head_analysis.py..."
        if python temporal_head_analysis.py \
            --model "$model" \
            --dataset "$dataset" \
            --episode "$EPISODE" \
            --num_frames "$NUM_FRAMES"; then
            echo "[2/3] Analysis complete."
        else
            echo "[2/3] FAILED (non-fatal)."
        fi

        # Step 3: Parameter sweep simulation
        echo "[3/3] Running pipeline_simulation.py..."
        if python pipeline_simulation.py \
            --model "$model" \
            --dataset "$dataset"; then
            echo "[3/3] Simulation complete."
            SUCCEEDED="$SUCCEEDED ${model}_${dataset}"
        else
            echo "[3/3] FAILED (non-fatal)."
            SUCCEEDED="$SUCCEEDED ${model}_${dataset}"
        fi
    done
done

# Cross-model comparison
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Cross-Model Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python pipeline_simulation.py --cross_model 2>/dev/null || true

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
echo "Results:"
echo "  Algorithm:  output/temporal_results/{model}_{dataset}/"
echo "  Analysis:   output/temporal_analysis/{model}_{dataset}/"
echo "  Comparison: output/temporal_analysis/cross_model_comparison.csv"
