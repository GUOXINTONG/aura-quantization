from __future__ import annotations

import random
from typing import Any, Iterable


LOW_CALIBRATION_WARNING = (
    "WARNING: fewer than 128 calibration samples were requested. This is useful "
    "only for pipeline validation, not for a final INT8 checkpoint."
)


def render_calibration_examples(
    examples: Iterable[dict[str, Any]], processor: Any
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Render chat messages to text; tokenization is left to LLM Compressor.

    Leaving the text un-tokenized lets the compressor tokenize it once and avoids
    accidentally adding a second set of BOS/EOS tokens.
    """
    rendered: list[dict[str, str]] = []
    dropped = 0
    for example in examples:
        messages = example.get("messages")
        if not isinstance(messages, list) or not messages:
            dropped += 1
            continue
        try:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            dropped += 1
            continue
        if not isinstance(text, str) or not text.strip():
            dropped += 1
            continue
        rendered.append({"text": text})
    return rendered, {"processed": len(rendered), "dropped": dropped}


def prepare_calibration_dataset(
    dataset_name: str,
    split: str,
    sample_count: int,
    seed: int,
    processor: Any,
):
    try:
        from datasets import Dataset, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required for INT8 calibration. "
            "Install requirements-quantize.txt."
        ) from exc

    random.seed(seed)
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(min(sample_count, len(dataset))))
    rendered, stats = render_calibration_examples(dataset, processor)
    if not rendered:
        raise ValueError("No valid calibration examples could be rendered")
    return Dataset.from_list(rendered), stats

