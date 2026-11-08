#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any

import _bootstrap  # noqa: F401

from qwen_quant_lab.config import cli_overrides, comma_list, load_config
from qwen_quant_lab.model_utils import (
    directory_size,
    ensure_vllm_multiprocessing_spawn,
    load_config_json,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark variants with offline vLLM")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model")
    parser.add_argument("--variants", default="bf16,fp8,int8")
    parser.add_argument("--backend", choices=["vllm"], default="vllm")
    parser.add_argument("--output")
    return parser


BENCHMARK_PROMPTS = (
    "Explain in two sentences why calibration data should match the target domain.",
    "List three practical differences between FP8 and INT8 inference on GPUs.",
    "Write a short Python function that returns the n-th Fibonacci number.",
    "用三句话解释为什么模型量化后需要做回归评测。",
)


def _gpu_used_bytes() -> int | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0].strip()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(visible))
        except ValueError:
            handle = pynvml.nvmlDeviceGetHandleByUUID(visible)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
    except Exception:
        return None


class PeakMemorySampler:
    """Poll NVML in a background thread so the peak during generation is captured.

    Only the first visible GPU is sampled; with tensor_parallel_size > 1 the
    value is a per-GPU figure, not a total.
    """

    def __init__(self, interval_seconds: float = 0.05):
        self.interval_seconds = interval_seconds
        self.peak: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        value = _gpu_used_bytes()
        if value is not None and (self.peak is None or value > self.peak):
            self.peak = value

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "PeakMemorySampler":
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._sample()


def _checkpoint_size(model: str) -> int | None:
    path = Path(model)
    if path.exists():
        return directory_size(path)
    try:
        _, files = load_config_json(model)
        return sum(item.get("size") or 0 for item in files if item["name"].endswith(".safetensors"))
    except Exception:
        return None


def _mean(values) -> float | None:
    usable = [value for value in values if value is not None]
    return mean(usable) if usable else None


def _shutdown_engine(llm: Any) -> None:
    if llm is not None:
        try:
            engine = getattr(llm, "llm_engine", None)
            if engine is not None and hasattr(engine, "shutdown"):
                engine.shutdown()
        except Exception:
            pass
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
        except Exception:
            pass


def benchmark_variant(name: str, model: str, config) -> dict[str, Any]:
    ensure_vllm_multiprocessing_spawn()
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vllm is required; install requirements-vllm.txt") from exc
    llm = None
    before = _gpu_used_bytes()
    try:
        load_started = time.perf_counter()
        # Do not pass --quantization: the checkpoint configuration selects its format.
        llm = LLM(
            model=model,
            trust_remote_code=config.model.trust_remote_code,
            max_model_len=config.evaluation.max_model_len,
            tensor_parallel_size=config.evaluation.tensor_parallel_size,
            gpu_memory_utilization=config.evaluation.gpu_memory_utilization,
            seed=config.evaluation.seed,
            gdn_prefill_backend="triton",
        )
        load_seconds = time.perf_counter() - load_started
        idle = _gpu_used_bytes()
        params = SamplingParams(
            temperature=config.evaluation.temperature,
            max_tokens=config.evaluation.max_tokens,
            seed=config.evaluation.seed,
            skip_special_tokens=config.evaluation.skip_special_tokens,
        )
        # Warm up once so CUDA graph capture / kernel compilation is not timed.
        llm.generate([BENCHMARK_PROMPTS[0]], params, use_tqdm=False)
        batch_results = []
        first_samples = []
        with PeakMemorySampler() as sampler:
            for batch_size in config.evaluation.concurrency:
                for repeat in range(config.evaluation.repeat_count):
                    prompts = [
                        BENCHMARK_PROMPTS[index % len(BENCHMARK_PROMPTS)]
                        for index in range(batch_size)
                    ]
                    started = time.perf_counter()
                    outputs = llm.generate(prompts, params, use_tqdm=False)
                    elapsed = time.perf_counter() - started
                    successful = sum(bool(item.outputs) for item in outputs)
                    output_tokens = sum(
                        len(item.outputs[0].token_ids) for item in outputs if item.outputs
                    )
                    if not first_samples:
                        first_samples = [
                            item.outputs[0].text for item in outputs if item.outputs
                        ]
                    batch_results.append(
                        {
                            "batch_size": batch_size,
                            "repeat": repeat,
                            "batch_elapsed_seconds": elapsed,
                            "output_tokens": output_tokens,
                            "request_throughput": successful / elapsed if elapsed else None,
                            "output_tokens_per_second": (
                                output_tokens / elapsed if elapsed else None
                            ),
                            "successful_requests": successful,
                            "failed_requests": batch_size - successful,
                        }
                    )
        peak = sampler.peak
        by_batch_size = {}
        for batch_size in dict.fromkeys(config.evaluation.concurrency):
            rows = [item for item in batch_results if item["batch_size"] == batch_size]
            by_batch_size[str(batch_size)] = {
                "runs": len(rows),
                "mean_batch_elapsed_seconds": _mean(
                    item["batch_elapsed_seconds"] for item in rows
                ),
                "mean_request_throughput": _mean(
                    item["request_throughput"] for item in rows
                ),
                "mean_output_tokens_per_second": _mean(
                    item["output_tokens_per_second"] for item in rows
                ),
            }
        single = by_batch_size.get("1", {})
        largest = by_batch_size[str(max(config.evaluation.concurrency))]
        return {
            "variant": name,
            "model": model,
            "checkpoint_size_bytes": _checkpoint_size(model),
            "startup_time_seconds": load_seconds,
            "idle_gpu_memory_bytes": (idle - before) if idle is not None and before is not None else None,
            "peak_gpu_memory_bytes": (peak - before) if peak is not None and before is not None else None,
            "gpu_memory_note": (
                "NVML used memory on the first visible GPU, relative to before load; "
                "vLLM pre-allocates KV cache up to gpu_memory_utilization"
            ),
            "time_to_first_token_seconds": None,
            "time_to_first_token_note": "offline batched vLLM does not expose streaming TTFT",
            # Batch size 1 is the only case where batch time equals request latency.
            "single_request_latency_seconds": single.get("mean_batch_elapsed_seconds"),
            "max_batch_size": max(config.evaluation.concurrency),
            "max_batch_output_tokens_per_second": largest["mean_output_tokens_per_second"],
            "max_batch_request_throughput": largest["mean_request_throughput"],
            "by_batch_size": by_batch_size,
            "successful_requests": sum(item["successful_requests"] for item in batch_results),
            "failed_requests": sum(item["failed_requests"] for item in batch_results),
            "standard_vllm_test": "pass" if first_samples else "fail",
            "text_generation_smoke_test": "pass" if first_samples else "fail",
            "image_text_smoke_test": "not implemented in experimental harness",
            "measurement_note": (
                "Offline vLLM batches: batch_size is the number of prompts submitted "
                "in one generate() call, not concurrent online-serving clients."
            ),
            "batches": batch_results,
            "sample_outputs": first_samples,
        }
    finally:
        _shutdown_engine(llm)
        llm = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config, overrides=cli_overrides(model=args.model))
    models = {
        "bf16": config.model.source,
        "fp8": config.output.fp8_dir,
        "int8": config.output.int8_dir,
    }
    results = {}
    for variant in comma_list(args.variants):
        if variant not in models:
            raise ValueError(f"Unknown variant: {variant}")
        if variant != "bf16" and not Path(models[variant]).is_dir():
            raise FileNotFoundError(f"Checkpoint for {variant} does not exist: {models[variant]}")
        print(f"Benchmarking {variant} alone; model={models[variant]}")
        results[variant] = benchmark_variant(variant, models[variant], config)
    output = Path(args.output or Path(config.output.results_dir) / "benchmark.json")
    write_json(output, results)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
