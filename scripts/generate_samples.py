#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap

from qwen_quant_lab.config import cli_overrides, comma_list, load_config
from qwen_quant_lab.inference import generate_with_vllm, read_prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate matched BF16/FP8/INT8 samples")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model")
    parser.add_argument("--variants", default="bf16,fp8,int8")
    parser.add_argument(
        "--prompts",
        default=str(_bootstrap.ROOT / "data" / "evaluation_prompts.jsonl"),
        help="JSONL prompts file (default: data/evaluation_prompts.jsonl in the repository)",
    )
    parser.add_argument("--output")
    return parser


def _apply_chat_template(prompts, source: str, trust_remote_code: bool, template_kwargs=None):
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=trust_remote_code)
    except Exception as exc:
        print(f"WARNING: using raw prompts because the source tokenizer could not load: {exc}")
        return prompts
    formatted = []
    for item in prompts:
        value = dict(item)
        try:
            value["prompt"] = tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                **(template_kwargs or {}),
            )
        except Exception as exc:
            print(f"WARNING: chat template failed for {item['id']}, using raw prompt: {exc}")
            value["prompt"] = item["prompt"]
        formatted.append(value)
    return formatted


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config, overrides=cli_overrides(model=args.model))
    prompts = _apply_chat_template(
        read_prompts(args.prompts),
        config.model.source,
        config.model.trust_remote_code,
        config.evaluation.chat_template_kwargs,
    )
    models = {
        "bf16": config.model.source,
        "fp8": config.output.fp8_dir,
        "int8": config.output.int8_dir,
    }
    records = []
    for variant in comma_list(args.variants):
        if variant not in models:
            raise ValueError(f"Unknown variant: {variant}")
        if variant != "bf16" and not Path(models[variant]).is_dir():
            raise FileNotFoundError(f"Checkpoint for {variant} does not exist: {models[variant]}")
        print(f"Generating {variant} samples from {models[variant]}")
        records.extend(
            generate_with_vllm(
                models[variant],
                prompts,
                variant=variant,
                trust_remote_code=config.model.trust_remote_code,
                seed=config.evaluation.seed,
                temperature=config.evaluation.temperature,
                max_tokens=config.evaluation.max_tokens,
                max_model_len=config.evaluation.max_model_len,
                tensor_parallel_size=config.evaluation.tensor_parallel_size,
                gpu_memory_utilization=config.evaluation.gpu_memory_utilization,
                skip_special_tokens=config.evaluation.skip_special_tokens,
            )
        )
    output = Path(args.output or Path(config.output.results_dir) / "generations.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} generation records to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

