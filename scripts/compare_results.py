#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from qwen_quant_lab.config import cli_overrides, load_config
from qwen_quant_lab.reporting import summarize_regression, write_comparison_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare benchmark and generation results")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model")
    parser.add_argument("--benchmark")
    parser.add_argument("--generations")
    return parser


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config, overrides=cli_overrides(model=args.model))
    results_dir = Path(config.output.results_dir)
    benchmark = _read_json(Path(args.benchmark or results_dir / "benchmark.json")) or {}
    generations = _read_jsonl(Path(args.generations or results_dir / "generations.jsonl"))
    regression = summarize_regression(generations) if generations else {}
    checkpoint_paths = {
        "fp8": Path(config.output.fp8_dir),
        "int8": Path(config.output.int8_dir),
    }
    comparison: dict[str, dict[str, Any]] = {}
    for variant in ("bf16", "fp8", "int8"):
        bench = benchmark.get(variant, {})
        reg = regression.get(variant, {})
        quantization_time = None
        validation = "N/A"
        if variant in checkpoint_paths:
            metadata = _read_json(checkpoint_paths[variant] / "experiment_metadata.json") or {}
            quantization_time = metadata.get("elapsed_seconds")
            report = _read_json(checkpoint_paths[variant] / "validation_report.json") or {}
            validation = report.get("overall_status", "missing")
        comparison[variant] = {
            "checkpoint_size_bytes": bench.get("checkpoint_size_bytes"),
            "quantization_time_seconds": quantization_time,
            "load_time_seconds": bench.get("startup_time_seconds"),
            "idle_gpu_memory_bytes": bench.get("idle_gpu_memory_bytes"),
            "peak_gpu_memory_bytes": bench.get("peak_gpu_memory_bytes"),
            "time_to_first_token_seconds": bench.get("time_to_first_token_seconds"),
            "single_request_latency_seconds": bench.get("single_request_latency_seconds"),
            "max_batch_size": bench.get("max_batch_size"),
            "max_batch_output_tokens_per_second": bench.get(
                "max_batch_output_tokens_per_second"
            ),
            "max_batch_requests_per_second": bench.get("max_batch_request_throughput"),
            "generation_success_rate": reg.get("generation_success_rate"),
            "compared_prompts": reg.get("compared_prompts"),
            "exact_match_against_bf16": reg.get("exact_match"),
            "normalized_match_against_bf16": reg.get("normalized_match"),
            "token_similarity_against_bf16": reg.get("token_similarity"),
            "json_prompts": reg.get("json_prompts"),
            "json_parse_success": reg.get("json_parse_success"),
            "expected_key_presence": reg.get("expected_key_presence"),
            "checkpoint_validation": validation,
            "standard_vllm_test": bench.get("standard_vllm_test"),
        }
    write_comparison_outputs(results_dir, comparison)
    print(f"Wrote comparison.json, comparison.csv, and report.md to {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

