#!/usr/bin/env bash
# Deploy a quantized checkpoint into the Aura demo (docker container on the compute node).
# Usage (login node): deploy_demo.sh fp8|int8 [--replace <running-model-id>]
#   1. copies the checkpoint to $AURA_ROOT/checkpoints/tt-simple-bench-all-64k-fix-<VARIANT>/checkpoint-1467
#   2. registers it in aura_model_registry.json (backup kept in $AURA_ROOT/backup/)
#   3. optionally stops the model it replaces, then starts it with --two_token on $GPU_INDEX
set -euo pipefail
source "$(dirname "$0")/env.sh"
variant="${1:?usage: deploy_demo.sh fp8|int8 [--replace <model-id>]}"
replace=""
[[ "${2:-}" == "--replace" ]] && replace="${3:?model id to stop}"
case "$variant" in
  fp8) SRC="$FP8_DIR"; NAME=tt-simple-bench-all-64k-fix-FP8; PORT="${PORT:-8350}" ;;
  int8) SRC="$INT8_DIR"; NAME=tt-simple-bench-all-64k-fix-INT8; PORT="${PORT:-8351}" ;;
  *) echo "unknown variant: $variant"; exit 2 ;;
esac
DST="$AURA_ROOT/checkpoints/$NAME/checkpoint-1467"
if [[ ! -e "$DST" ]]; then
  mkdir -p "$(dirname "$DST")"
  cp -a "$SRC" "$DST"
  chmod -R g+rwX "$(dirname "$DST")"
fi
"$PY_VLLM" "$EXP_DIR/register_demo_model.py" "$AURA_ROOT/aura_model_registry.json" \
  --id "$NAME" --path "/workspace/checkpoints/$NAME/checkpoint-1467" --gpu "$GPU_INDEX" --port "$PORT"
type module >/dev/null 2>&1 && module load slurm
srun --jobid="$SLURM_JOB" --overlap -N1 -n1 bash -c "
  set -e
  if [[ -n '$replace' ]]; then docker exec $AURA_CONTAINER bash /workspace/aura_model.sh stop main '$replace'; sleep 5; fi
  used=\$(nvidia-smi -i $GPU_INDEX --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  (( used < 2048 )) || { echo \"GPU $GPU_INDEX busy (\${used}MiB); not starting\"; exit 3; }
  docker exec $AURA_CONTAINER bash /workspace/aura_model.sh start main $NAME --two_token
"
