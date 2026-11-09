#!/usr/bin/env bash
# Quantize and validate one variant on the guarded GPU. Usage: quantize.sh fp8|int8
# Run inside a GPU step, e.g.: srun --jobid=$SLURM_JOB --overlap -N1 -n1 experiments/aura-checkpoint-1467/quantize.sh fp8
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$EXP_DIR/gpu_guard.sh"
variant="${1:?usage: quantize.sh fp8|int8}"
case "$variant" in
  fp8) OUT="$FP8_DIR" ;;
  int8) OUT="$INT8_DIR" ;;
  *) echo "unknown variant: $variant"; exit 2 ;;
esac
SRC=$("$PY_VLLM" -c "import sys; sys.path.insert(0, 'src'); from qwen_quant_lab.config import load_config; print(load_config('$CFG').model.source)")
echo "=== [$(date -Is)] $variant quantization"
# The quantize env has no vLLM, so runtime validation runs separately below.
"$PY_QUANT" "scripts/quantize_$variant.py" --config "$CFG" --skip-runtime-validation
echo "=== [$(date -Is)] $variant vLLM validation"
# validate_checkpoint.py prints vLLM logs on stdout; the authoritative report is
# written to $OUT/validation_report.json.
"$PY_VLLM" scripts/validate_checkpoint.py --source "$SRC" --checkpoint "$OUT" \
  --max-model-len 4096 --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --trust-remote-code \
  > "$RUN_DIR/validate_$variant.stdout" || true
"$PY_VLLM" scripts/summarize_validation.py "$OUT"
echo "=== [$(date -Is)] done"
