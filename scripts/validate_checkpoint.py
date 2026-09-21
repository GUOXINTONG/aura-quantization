#!/usr/bin/env python3
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from qwen_quant_lab.checkpoint import validate_checkpoint
from qwen_quant_lab.model_utils import print_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an offline quantized checkpoint")
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-runtime", action="store_true", help="Skip vLLM load/generation")
    parser.add_argument(
        "--repair-mtp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the official compressed-tensors helper for missing mtp.* tensors",
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_checkpoint(
        args.source,
        args.checkpoint,
        trust_remote_code=args.trust_remote_code,
        run_runtime=not args.skip_runtime,
        repair_mtp=args.repair_mtp,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print_json(report)
    return 0 if report["overall_status"] in {"pass", "pass_with_warnings"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
