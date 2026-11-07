from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .metrics import compare_output, failed_output_metrics, json_checks
from .model_utils import write_json


def _average(values: Iterable[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return mean(usable) if usable else None


def _succeeded(record: dict[str, Any] | None) -> bool:
    # Older records without a ``success`` field are treated as successful.
    return bool(record) and record.get("success", True) is not False


def summarize_regression(
    records: list[dict[str, Any]], baseline_variant: str = "bf16"
) -> dict[str, dict[str, Any]]:
    """Summarize each variant against the baseline, prompt by prompt.

    - A missing or failed candidate generation counts as a mismatch (and as a
      JSON failure for JSON prompts); it is never silently skipped.
    - When the baseline itself failed there is no valid reference, so lexical
      metrics skip that prompt, but the candidate's JSON checks still count.
    """
    by_key = {(item["model_variant"], item["prompt_id"]): item for item in records}
    variants = sorted({item["model_variant"] for item in records})
    result: dict[str, dict[str, Any]] = {}
    baseline_ids = sorted(
        item["prompt_id"] for item in records if item["model_variant"] == baseline_variant
    )
    for variant in variants:
        lexical: list[dict[str, Any]] = []
        json_results: list[dict[str, Any]] = []
        missing = failed = baseline_failed = 0
        for prompt_id in baseline_ids:
            baseline = by_key.get((baseline_variant, prompt_id))
            candidate = by_key.get((variant, prompt_id))
            expected_keys = (candidate or {}).get("expected_keys") or (
                baseline or {}
            ).get("expected_keys")
            if candidate is None:
                missing += 1
                metrics = failed_output_metrics(expected_keys)
            elif not _succeeded(candidate):
                failed += 1
                metrics = failed_output_metrics(expected_keys)
            elif not _succeeded(baseline):
                baseline_failed += 1
                if expected_keys:
                    json_results.append(json_checks(candidate.get("output", ""), expected_keys))
                continue
            else:
                metrics = compare_output(
                    baseline.get("output", ""),
                    candidate.get("output", ""),
                    expected_keys,
                )
            lexical.append(metrics)
            if metrics["json_parse_success"] is not None:
                json_results.append(metrics)
        total = len(baseline_ids)
        result[variant] = {
            "baseline_prompts": total,
            "compared_prompts": len(lexical),
            "missing_prompts": missing,
            "failed_generations": failed,
            "baseline_failed_prompts": baseline_failed,
            "generation_success_rate": (
                (total - missing - failed) / total if total else None
            ),
            "exact_match": _average(int(item["exact_match"]) for item in lexical),
            "normalized_match": _average(int(item["normalized_match"]) for item in lexical),
            "token_similarity": _average(item["token_similarity"] for item in lexical),
            "json_prompts": len(json_results),
            "json_parse_success": _average(
                int(item["json_parse_success"]) for item in json_results
            ),
            "expected_key_presence": _average(
                int(item["expected_key_presence"]) for item in json_results
            ),
        }
    return result


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_comparison_outputs(
    results_dir: str | Path,
    comparison: dict[str, dict[str, Any]],
) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    write_json(results_dir / "comparison.json", comparison)
    # Keep the metric order chosen by the caller instead of sorting alphabetically.
    metrics = list(dict.fromkeys(key for values in comparison.values() for key in values))
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
        formatted = [_format_value(value) for value in values]
        lines.append(f"| {metric.replace('_', ' ').title()} | " + " | ".join(formatted) + " |")
    lines.extend(
        [
            "",
            "> Exact and token-level similarity are lightweight regression signals. They are not a complete model-quality evaluation.",
            "> Failed or missing generations count as mismatches. JSON metrics only cover prompts with `expected_keys`.",
            "> Throughput comes from offline vLLM batches (batch size = prompts per generate() call), not an online serving load test.",
            "",
        ]
    )
    (results_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

