from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from .model_utils import ensure_vllm_multiprocessing_spawn, release_gpu_memory


def read_prompts(path: str | Path) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not value.get("id") or not value.get("prompt"):
                raise ValueError(f"Prompt at {path}:{line_number} needs id and prompt")
            prompts.append(value)
    return prompts


def generate_with_vllm(
    model: str,
    prompts: Iterable[dict[str, Any]],
    *,
    variant: str,
    trust_remote_code: bool,
    seed: int,
    temperature: float,
    max_tokens: int,
    max_model_len: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> list[dict[str, Any]]:
    ensure_vllm_multiprocessing_spawn()
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vllm is required; install requirements-vllm.txt") from exc
    prompt_list = list(prompts)
    llm = None
    try:
        llm = LLM(
            model=model,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            gdn_prefill_backend="triton",
        )
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        started = time.perf_counter()
        outputs = llm.generate(
            [item["prompt"] for item in prompt_list], params, use_tqdm=False
        )
        elapsed = time.perf_counter() - started
        records = []
        per_request = elapsed / len(outputs) if outputs else 0.0
        for prompt, result in zip(prompt_list, outputs):
            candidate = result.outputs[0] if result.outputs else None
            text = candidate.text if candidate else ""
            output_tokens = len(candidate.token_ids) if candidate else 0
            records.append(
                {
                    "model_variant": variant,
                    "prompt_id": prompt["id"],
                    "category": prompt.get("category", "unknown"),
                    "prompt": prompt["prompt"],
                    "expected_keys": prompt.get("expected_keys"),
                    "output": text,
                    "latency_seconds": per_request,
                    "input_tokens": len(result.prompt_token_ids or []),
                    "output_tokens": output_tokens,
                    "tokens_per_second": output_tokens / per_request if per_request else None,
                    "success": bool(candidate),
                    "error": None if candidate else "vLLM returned no candidate",
                }
            )
        return records
    finally:
        if llm is not None:
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel

                destroy_model_parallel()
            except Exception:
                pass
        llm = None
        release_gpu_memory()
