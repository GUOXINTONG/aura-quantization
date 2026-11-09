# Source inside a compute-node step: pin to $GPU_INDEX and refuse to run if it is busy.
# Set ALLOW_BUSY=1 to skip the idle check.
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
read -r uuid used <<<"$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid,memory.used --format=csv,noheader,nounits | tr -d ',')"
echo "host=$(hostname) gpu$GPU_INDEX uuid=$uuid used_mib=$used"
[[ "$uuid" == "$GPU_UUID" ]] || { echo "GPU $GPU_INDEX UUID mismatch (expected $GPU_UUID)"; exit 2; }
others=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$GPU_UUID" || true)
if [[ "${ALLOW_BUSY:-0}" != 1 ]] && { (( used > 2048 )) || (( others > 0 )); }; then
  echo "GPU $GPU_INDEX is busy (used=${used}MiB, processes=${others}); aborting"; exit 3
fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
