#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_quant_lab.checkpoint import quantization_metadata, validate_checkpoint
from qwen_quant_lab.config import cli_overrides, load_config
from qwen_quant_lab.inspection import inspect_source
from qwen_quant_lab.model_utils import (
    collect_environment_metadata,
    load_config_json,
    prepare_output_directory,
    preserve_model_assets,
    utc_now,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an FP8 dynamic checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ignore-linear-attention", action="store_true")
    parser.add_argument(
        "--skip-runtime-validation",
        action="store_true",
        help="Run file/tensor checks but skip vLLM generation",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(
        args.config,
        overrides=cli_overrides(
            model=args.model,
            ignore_linear_attention=args.ignore_linear_attention,
        ),
    )
    print("Effective configuration:")
    print(config.to_json())
    ignored = config.ignored_layers()
    inspect_source(
        config.model.source,
        ignore_patterns=ignored,
        output_paths=[config.output.fp8_dir],
    )
    output = prepare_output_directory(
        config.output.fp8_dir,
        source=config.model.source,
        overwrite=args.overwrite,
    )
    started_at = utc_now()
    started = time.perf_counter()
    metadata = {
        "operation": "fp8_quantization",
        "status": "running",
        "source_model": config.model.source,
        "output_directory": str(output),
        "quantization_scheme": config.fp8.scheme,
        "ignored_module_patterns": ignored,
        "start_time": started_at,
        "environment": collect_environment_metadata(),
        "effective_config": config.to_dict(),
    }
    try:
        try:
            from llmcompressor import model_free_ptq
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Installed llmcompressor does not provide model_free_ptq. "
                "Install a compatible current llmcompressor release."
            ) from exc
        model_free_ptq(
            model_stub=config.model.source,
            save_directory=str(output),
            scheme=config.fp8.scheme,
            ignore=ignored,
            max_workers=config.fp8.max_workers,
            device=config.fp8.device,
        )
        metadata["asset_preservation"] = preserve_model_assets(
            config.model.source,
            output,
            trust_remote_code=config.model.trust_remote_code,
        )
        output_config, _ = load_config_json(str(output))
        quant_config = quantization_metadata(output_config)
        if not quant_config:
            raise RuntimeError("FP8 output config has no quantization metadata")
        metadata["output_quantization_config"] = quant_config
        metadata["status"] = "quantized"
    except Exception as exc:
        metadata["status"] = "error"
        metadata["error"] = str(exc)
        raise
    finally:
        metadata["end_time"] = utc_now()
        metadata["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "experiment_metadata.json", metadata)

    report = validate_checkpoint(
        config.model.source,
        output,
        trust_remote_code=config.model.trust_remote_code,
        run_runtime=not args.skip_runtime_validation,
        max_model_len=config.evaluation.max_model_len,
        tensor_parallel_size=config.evaluation.tensor_parallel_size,
        gpu_memory_utilization=config.evaluation.gpu_memory_utilization,
    )
    metadata["validation_status"] = report["overall_status"]
    validation_passed = report["overall_status"] in {"pass", "pass_with_warnings"}
    metadata["status"] = (
        "quantized_pending_runtime_validation"
        if validation_passed and args.skip_runtime_validation
        else ("complete" if validation_passed else "invalid")
    )
    write_json(output / "experiment_metadata.json", metadata)
    if not validation_passed:
        raise RuntimeError(f"FP8 checkpoint validation failed; see {output / 'validation_report.json'}")
    print(f"FP8 checkpoint written: {output} (status={metadata['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
