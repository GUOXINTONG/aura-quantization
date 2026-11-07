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
            if any(item["id"] == value["id"] for item in prompts):
                raise ValueError(f"Duplicate prompt id at {path}:{line_number}: {value['id']}")
            prompts.append(value)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
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
        # All prompts run as one offline batch, so per-request wall-clock latency
        # is not observable here. Record the batch time and an amortized value
        # under names that cannot be mistaken for real request latency.
        amortized = elapsed / len(outputs) if outputs else None
        batch_output_tokens = sum(
            len(item.outputs[0].token_ids) for item in outputs if item.outputs
        )
        records = []
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
                    "finish_reason": getattr(candidate, "finish_reason", None),
                    "batch_size": len(prompt_list),
                    "batch_elapsed_seconds": elapsed,
                    "amortized_latency_seconds": amortized,
                    "batch_output_tokens_per_second": (
                        batch_output_tokens / elapsed if elapsed else None
                    ),
                    "input_tokens": len(result.prompt_token_ids or []),
                    "output_tokens": output_tokens,
                    "success": bool(candidate),
                    "error": None if candidate else "vLLM returned no candidate",
                }
            )
        returned = {record["prompt_id"] for record in records}
        for prompt in prompt_list:
            if prompt["id"] not in returned:
                records.append(
                    {
                        "model_variant": variant,
                        "prompt_id": prompt["id"],
                        "category": prompt.get("category", "unknown"),
                        "prompt": prompt["prompt"],
                        "expected_keys": prompt.get("expected_keys"),
                        "output": "",
                        "success": False,
                        "error": "vLLM returned no result for this prompt",
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
