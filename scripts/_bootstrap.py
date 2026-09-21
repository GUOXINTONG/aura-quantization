from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# vLLM's V1 engine forks EngineCore by default on Linux. If CUDA has already
# been initialized in the parent (common after importing torch/transformers),
# fork fails with "Cannot re-initialize CUDA in forked subprocess".
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

