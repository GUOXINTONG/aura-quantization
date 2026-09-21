#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
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


def _checkpoint_size(model: str) -> int | None:
    path = Path(model)
    if path.exists():
        return directory_size(path)
    try:
        _, files = load_config_json(model)
        return sum(item.get("size") or 0 for item in files if item["name"].endswith(".safetensors"))
    except Exception:
        return None


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
        )
        batch_results = []
        first_samples = []
        for concurrency in config.evaluation.concurrency:
            for repeat in range(config.evaluation.repeat_count):
                prompts = [
                    "Explain in two sentences why calibration data should match the target domain."
                    for _ in range(concurrency)
                ]
                started = time.perf_counter()
                outputs = llm.generate(prompts, params, use_tqdm=False)
                elapsed = time.perf_counter() - started
                successful = sum(bool(item.outputs) for item in outputs)
                output_tokens = sum(
                    len(item.outputs[0].token_ids) for item in outputs if item.outputs
                )
                if not first_samples:
                    first_samples = [item.outputs[0].text for item in outputs if item.outputs]
                batch_results.append(
                    {
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "end_to_end_latency_seconds": elapsed,
                        "request_throughput": successful / elapsed if elapsed else None,
                        "output_tokens_per_second": output_tokens / elapsed if elapsed else None,
                        "successful_requests": successful,
                        "failed_requests": concurrency - successful,
                    }
                )
        peak = _gpu_used_bytes()
        return {
            "variant": name,
            "model": model,
            "checkpoint_size_bytes": _checkpoint_size(model),
            "startup_time_seconds": load_seconds,
            "idle_gpu_memory_bytes": (idle - before) if idle is not None and before is not None else None,
            "peak_gpu_memory_bytes": (peak - before) if peak is not None and before is not None else None,
            "time_to_first_token_seconds": None,
            "time_to_first_token_note": "offline batched vLLM does not expose streaming TTFT",
            "mean_end_to_end_latency_seconds": mean(item["end_to_end_latency_seconds"] for item in batch_results),
            "mean_output_tokens_per_second": mean(item["output_tokens_per_second"] for item in batch_results if item["output_tokens_per_second"] is not None),
            "mean_request_throughput": mean(item["request_throughput"] for item in batch_results if item["request_throughput"] is not None),
            "successful_requests": sum(item["successful_requests"] for item in batch_results),
            "failed_requests": sum(item["failed_requests"] for item in batch_results),
            "standard_vllm_test": "pass",
            "text_generation_smoke_test": "pass" if first_samples else "fail",
            "image_text_smoke_test": "not implemented in experimental harness",
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
