#!/usr/bin/env bash
# Generate matched outputs, benchmark, and write the comparison report for BF16/FP8/INT8.
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$EXP_DIR/gpu_guard.sh"
echo "=== [$(date -Is)] generate"
"$PY_VLLM" scripts/generate_samples.py --config "$CFG" --variants bf16,fp8,int8
echo "=== [$(date -Is)] benchmark"
"$PY_VLLM" scripts/benchmark.py --config "$CFG" --variants bf16,fp8,int8
echo "=== [$(date -Is)] compare"
"$PY_VLLM" scripts/compare_results.py --config "$CFG"
cat "$OUTPUT_ROOT/results/report.md"
echo "=== [$(date -Is)] done"
