#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from qwen_quant_lab.config import cli_overrides, load_config
from qwen_quant_lab.inspection import inspect_source
from qwen_quant_lab.model_utils import print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a source Qwen checkpoint")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", help="Hugging Face model ID or local checkpoint")
    parser.add_argument("--ignore-linear-attention", action="store_true")
    parser.add_argument("--report", help="Optional JSON report path")
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
    report_path = args.report or str(Path(config.output.root) / "inspection_report.json")
    report = inspect_source(
        config.model.source,
        ignore_patterns=config.ignored_layers(),
        output_paths=[config.output.fp8_dir, config.output.int8_dir],
        report_path=report_path,
    )
    print("Inspection report:")
    print_json(report)
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

