#!/usr/bin/env bash
# Full pipeline on one GPU: FP8 -> INT8 -> evaluation. Each stage stops the chain on failure.
# Launch from the login node so it survives disconnects:
#   nohup setsid srun --jobid=132149 --overlap -N1 -n1 experiments/aura-checkpoint-1467/run_all.sh \
#     > runs/aura-checkpoint-1467/run_all.log 2>&1 < /dev/null &
set -uo pipefail
source "$(dirname "$0")/env.sh"
for variant in fp8 int8; do
  "$EXP_DIR/quantize.sh" "$variant" || { echo "$variant failed; stopping"; exit 1; }
  status=$("$PY_VLLM" -c "import json,sys; print(json.load(open(sys.argv[1]))['overall_status'])" \
    "$([[ $variant == fp8 ]] && echo "$FP8_DIR" || echo "$INT8_DIR")/validation_report.json")
  echo "$variant validation: $status"
  case "$status" in pass|pass_with_warnings) ;; *) echo "$variant not valid; stopping"; exit 1 ;; esac
done
"$EXP_DIR/evaluate.sh"
