#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401

from qwen_quant_lab.calibration import LOW_CALIBRATION_WARNING, prepare_calibration_dataset
from qwen_quant_lab.checkpoint import validate_checkpoint
from qwen_quant_lab.config import cli_overrides, load_config
from qwen_quant_lab.inspection import inspect_source
from qwen_quant_lab.model_utils import (
    collect_environment_metadata,
    load_model_and_processor,
    prepare_output_directory,
    print_quantization_selection,
    release_gpu_memory,
    utc_now,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a calibrated INT8 W8A8 checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ignore-linear-attention", action="store_true")
    parser.add_argument("--skip-runtime-validation", action="store_true")
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
    if config.int8.calibration_samples < 128:
        print(LOW_CALIBRATION_WARNING)
    ignored = config.ignored_layers()
    inspect_source(
        config.model.source,
        ignore_patterns=ignored,
        output_paths=[config.output.int8_dir],
    )
    output = prepare_output_directory(
        config.output.int8_dir,
        source=config.model.source,
        overwrite=args.overwrite,
    )
    started = time.perf_counter()
    metadata = {
        "operation": "int8_quantization",
        "status": "running",
        "source_model": config.model.source,
        "output_directory": str(output),
        "quantization_scheme": "W8A8",
        "ignored_module_patterns": ignored,
        "calibration": {
            "dataset": config.int8.calibration_dataset,
            "split": config.int8.calibration_split,
            "seed": config.int8.seed,
            "requested_samples": config.int8.calibration_samples,
            "max_sequence_length": config.int8.max_sequence_length,
        },
        "start_time": utc_now(),
        "environment": collect_environment_metadata(),
        "effective_config": config.to_dict(),
    }
    model = processor = None
    try:
        model, processor, _ = load_model_and_processor(
            config.model.source,
            trust_remote_code=config.model.trust_remote_code,
            dtype=config.model.dtype,
        )
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for INT8 quantization") from exc
        torch.manual_seed(config.int8.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.int8.seed)
        linear_modules = [
            name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
        ]
        metadata["module_selection"] = print_quantization_selection(linear_modules, ignored)
        dataset, stats = prepare_calibration_dataset(
            config.int8.calibration_dataset,
            config.int8.calibration_split,
            config.int8.calibration_samples,
            config.int8.seed,
            processor,
        )
        metadata["calibration"].update(stats)
        print(f"Calibration examples processed={stats['processed']} dropped={stats['dropped']}")
        try:
            from llmcompressor import oneshot
            from llmcompressor.modifiers.gptq import GPTQModifier
            from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Installed llmcompressor lacks the GPTQ/SmoothQuant APIs required "
                "by this experiment. Install requirements-quantize.txt."
            ) from exc
        recipe = [
            SmoothQuantModifier(
                smoothing_strength=config.int8.smoothquant_strength,
                ignore=ignored,
            ),
            # Static act-order is meaningful for grouped GPTQ, but vLLM rejects
            # it for the per-channel weights defined by W8A8.
            GPTQModifier(
                targets="Linear",
                scheme="W8A8",
                ignore=ignored,
                actorder=None,
            ),
        ]
        oneshot(
            model=model,
            processor=processor,
            dataset=dataset,
            recipe=recipe,
            max_seq_length=config.int8.max_sequence_length,
            num_calibration_samples=stats["processed"],
            shuffle_calibration_samples=False,
        )
        model.save_pretrained(output, save_compressed=True)
        processor.save_pretrained(output)
        metadata["status"] = "quantized"
    except Exception as exc:
        metadata["status"] = "error"
        metadata["error"] = str(exc)
        raise
    finally:
        metadata["end_time"] = utc_now()
        metadata["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "experiment_metadata.json", metadata)
        model = None
        processor = None
        release_gpu_memory()

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
        raise RuntimeError(f"INT8 checkpoint validation failed; see {output / 'validation_report.json'}")
    print(f"INT8 checkpoint written: {output} (status={metadata['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
