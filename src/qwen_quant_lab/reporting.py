from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .metrics import compare_output
from .model_utils import write_json


def _average(values: Iterable[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return mean(usable) if usable else None


def summarize_regression(
    records: list[dict[str, Any]], baseline_variant: str = "bf16"
) -> dict[str, dict[str, Any]]:
    by_key = {(item["model_variant"], item["prompt_id"]): item for item in records}
    variants = sorted({item["model_variant"] for item in records})
    result: dict[str, dict[str, Any]] = {}
    baseline_ids = {
        item["prompt_id"] for item in records if item["model_variant"] == baseline_variant
    }
    for variant in variants:
        comparisons = []
        for prompt_id in baseline_ids:
            baseline = by_key.get((baseline_variant, prompt_id))
            candidate = by_key.get((variant, prompt_id))
            if not baseline or not candidate:
                continue
            comparisons.append(
                compare_output(
                    baseline.get("output", ""),
                    candidate.get("output", ""),
                    candidate.get("expected_keys"),
                )
            )
        result[variant] = {
            "compared_prompts": len(comparisons),
            "exact_match": _average(int(item["exact_match"]) for item in comparisons),
            "normalized_match": _average(
                int(item["normalized_match"]) for item in comparisons
            ),
            "token_similarity": _average(item["token_similarity"] for item in comparisons),
            "json_parse_success": _average(
                int(item["json_parse_success"]) for item in comparisons
            ),
            "expected_key_presence": _average(
                int(item["expected_key_presence"])
                for item in comparisons
                if item["expected_key_presence"] is not None
            ),
        }
    return result


def write_comparison_outputs(
    results_dir: str | Path,
    comparison: dict[str, dict[str, Any]],
) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    write_json(results_dir / "comparison.json", comparison)
    metrics = sorted({key for values in comparison.values() for key in values})
    variants = [name for name in ("bf16", "fp8", "int8") if name in comparison]
    variants.extend(name for name in comparison if name not in variants)
    with (results_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", *variants])
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {"metric": metric, **{variant: comparison[variant].get(metric) for variant in variants}}
            )
    lines = [
        "# Quantization experiment report",
        "",
        "| Metric | " + " | ".join(name.upper() for name in variants) + " |",
        "|---|" + "---:|" * len(variants),
    ]
    for metric in metrics:
        values = [comparison[variant].get(metric) for variant in variants]
        formatted = ["N/A" if value is None else str(value) for value in values]
        lines.append(f"| {metric.replace('_', ' ').title()} | " + " | ".join(formatted) + " |")
    lines.extend(
        [
            "",
            "> Exact and token-level similarity are lightweight regression signals. They are not a complete model-quality evaluation.",
            "",
        ]
    )
    (results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

