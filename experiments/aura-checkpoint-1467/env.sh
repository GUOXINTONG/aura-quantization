# Shared settings for the checkpoint-1467 experiment. Override any value via the environment.
# Sourced by the other scripts in this directory; run them from anywhere.
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CFG="${CFG:-configs/aura-checkpoint-1467.yaml}"
RUN_DIR="${RUN_DIR:-runs/aura-checkpoint-1467}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs-aura-checkpoint-1467}"
FP8_DIR="$OUTPUT_ROOT/aura-checkpoint-1467-FP8-Dynamic"
INT8_DIR="$OUTPUT_ROOT/aura-checkpoint-1467-INT8-W8A8"

# Slurm job and the single GPU this experiment may use (hk01dgx014, physical GPU 1).
SLURM_JOB="${SLURM_JOB:-132149}"
GPU_INDEX="${GPU_INDEX:-1}"
GPU_UUID="${GPU_UUID:-GPU-d2090518-f60d-462b-45b3-a20d53f535ec}"

# Aura demo deployment (docker container on the compute node).
AURA_CONTAINER="${AURA_CONTAINER:-aura}"
AURA_ROOT="${AURA_ROOT:-/home/dyvm6xra/dyvm6xrauser36/aura_demo/main}"

mkdir -p "$RUN_DIR"
PY_QUANT="${PY_QUANT:-.venv-quantize/bin/python}"
PY_VLLM="${PY_VLLM:-.venv-vllm/bin/python}"
