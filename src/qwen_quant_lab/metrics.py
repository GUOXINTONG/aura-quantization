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
    parsed = parse_json_output(candidate)
    return {
        "exact_match": reference == candidate,
        "normalized_match": normalize_text(reference) == normalize_text(candidate),
        "token_similarity": token_edit_similarity(reference, candidate),
        "json_parse_success": parsed is not None,
        "expected_key_presence": (
            all(key in parsed for key in expected_keys)
            if expected_keys and isinstance(parsed, dict)
            else None
        ),
    }

