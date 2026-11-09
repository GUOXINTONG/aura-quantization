#!/usr/bin/env python3
"""Print a short summary of <checkpoint>/validation_report.json and experiment_metadata.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

DETAIL_CHECKS = ("generation", "transformers_config", "vllm_initialization", "transformers_processor")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_validation.py <checkpoint-dir>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    report = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
    print("overall:", report["overall_status"])
    print("warnings:", report["warnings"])
    for name, check in report["checks"].items():
        detail = check.get("detail") if name in DETAIL_CHECKS else ""
        print(f"  {name}: {check['status']} {str(detail or '')[:200]}")
    metadata_path = root / "experiment_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print("metadata status:", metadata.get("status"),
              "| quantization seconds:", round(metadata.get("elapsed_seconds") or 0, 1))
    tensors = report["checks"].get("tensor_index", {}).get("detail") or {}
    if isinstance(tensors, dict) and "source_tensor_count" in tensors:
        print("tensors source/output:", tensors["source_tensor_count"], tensors["output_tensor_count"],
              "| missing:", len(tensors["missing_source_tensors"]),
              "| unexpected:", len(tensors["unexpected_output_tensors"]))
    ignored = report["checks"].get("ignored_tensor_precision", {}).get("detail") or {}
    if isinstance(ignored, dict):
        print("ignored tensors checked:", ignored.get("ignored_tensors_checked"),
              "| vision:", ignored.get("vision_tensors_checked"),
              "| dtype/shape mismatches:", len(ignored.get("dtype_or_shape_mismatches") or {}))
    return 0 if report["overall_status"] in {"pass", "pass_with_warnings"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
