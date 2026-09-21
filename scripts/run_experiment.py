#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_quant_lab.config import cli_overrides, comma_list, load_config


VALID_STEPS = ("inspect", "fp8", "int8", "validate", "generate", "benchmark", "compare")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Qwen quantization experiment")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model")
    parser.add_argument(
        "--steps", default="inspect,fp8,int8,validate,generate,benchmark,compare"
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ignore-linear-attention", action="store_true")
    parser.add_argument("--skip-runtime-validation", action="store_true")
    return parser


def _base(script: str, args) -> list[str]:
    command = [sys.executable, str(Path(__file__).with_name(script)), "--config", args.config]
    if args.model:
        command.extend(["--model", args.model])
    return command


def planned_commands(args, config) -> list[tuple[str, list[str]]]:
    steps = comma_list(args.steps)
    unknown = sorted(set(steps) - set(VALID_STEPS))
    if unknown:
        raise ValueError(f"Unknown steps: {', '.join(unknown)}")
    commands: list[tuple[str, list[str]]] = []
    for step in steps:
        if step == "inspect":
            command = _base("inspect_model.py", args)
            if args.ignore_linear_attention:
                command.append("--ignore-linear-attention")
            commands.append((step, command))
        elif step in {"fp8", "int8"}:
            output = Path(config.output.fp8_dir if step == "fp8" else config.output.int8_dir)
            if args.skip_existing and output.is_dir() and any(output.iterdir()):
                print(f"Skipping existing {step} checkpoint: {output}")
                continue
            command = _base(f"quantize_{step}.py", args)
            if args.overwrite:
                command.append("--overwrite")
            if args.ignore_linear_attention:
                command.append("--ignore-linear-attention")
            if args.skip_runtime_validation:
                command.append("--skip-runtime-validation")
            commands.append((step, command))
        elif step == "validate":
            for variant, checkpoint in (
                ("fp8", config.output.fp8_dir),
                ("int8", config.output.int8_dir),
            ):
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("validate_checkpoint.py")),
                    "--source",
                    config.model.source,
                    "--checkpoint",
                    checkpoint,
                    "--max-model-len",
                    str(config.evaluation.max_model_len),
                    "--tensor-parallel-size",
                    str(config.evaluation.tensor_parallel_size),
                    "--gpu-memory-utilization",
                    str(config.evaluation.gpu_memory_utilization),
                ]
                if args.skip_runtime_validation:
                    command.append("--skip-runtime")
                commands.append((f"validate-{variant}", command))
        elif step == "generate":
            commands.append((step, _base("generate_samples.py", args)))
        elif step == "benchmark":
            commands.append((step, _base("benchmark.py", args)))
        elif step == "compare":
            commands.append((step, _base("compare_results.py", args)))
    return commands


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(
        args.config,
        overrides=cli_overrides(
            model=args.model,
            ignore_linear_attention=args.ignore_linear_attention,
        ),
    )
    commands = planned_commands(args, config)
    if args.dry_run:
        print("Dry run: no models will be downloaded or loaded and no output will be modified.")
        print("Effective configuration:")
        print(config.to_json())
        print(f"Source: {config.model.source}")
        print(f"FP8 output: {config.output.fp8_dir}")
        print(f"INT8 output: {config.output.int8_dir}")
        print("Planned commands:")
        for label, command in commands:
            print(f"  [{label}] {' '.join(command)}")
        return 0
    for label, command in commands:
        print(f"Running [{label}]: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
    print("Requested experiment steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

