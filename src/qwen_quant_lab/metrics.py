from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def token_edit_similarity(reference: str, candidate: str) -> float:
    return SequenceMatcher(
        None, normalize_text(reference).split(), normalize_text(candidate).split()
    ).ratio()


def parse_json_output(text: str) -> Any | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.I)
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return None


def compare_output(
    reference: str, candidate: str, expected_keys: list[str] | None = None
) -> dict[str, Any]:
    """Compare one candidate output with the BF16 reference output.

    JSON checks only apply to prompts that declare ``expected_keys``; for those
    prompts, output that cannot be parsed counts as a key-presence failure
    instead of being dropped from the average.
    """
    result: dict[str, Any] = {
        "exact_match": reference == candidate,
        "normalized_match": normalize_text(reference) == normalize_text(candidate),
        "token_similarity": token_edit_similarity(reference, candidate),
        "json_parse_success": None,
        "expected_key_presence": None,
    }
    if expected_keys:
        result.update(json_checks(candidate, expected_keys))
    return result


def json_checks(candidate: str, expected_keys: list[str]) -> dict[str, bool]:
    parsed = parse_json_output(candidate)
    return {
        "json_parse_success": parsed is not None,
        "expected_key_presence": isinstance(parsed, dict)
        and all(key in parsed for key in expected_keys),
    }


def failed_output_metrics(expected_keys: list[str] | None = None) -> dict[str, Any]:
    """Metrics for a missing or failed generation: it never counts as a match."""
    return {
        "exact_match": False,
        "normalized_match": False,
        "token_similarity": 0.0,
        "json_parse_success": False if expected_keys else None,
        "expected_key_presence": False if expected_keys else None,
    }
