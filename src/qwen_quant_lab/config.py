from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from .constants import LINEAR_ATTENTION_PATTERN


@dataclass
class ModelConfig:
    source: str = "Qwen/Qwen3.5-9B"
    trust_remote_code: bool = True
    dtype: str = "auto"


@dataclass
class OutputConfig:
    root: str = "outputs"
    fp8_dir: str = "outputs/Qwen3.5-9B-FP8-Dynamic"
    int8_dir: str = "outputs/Qwen3.5-9B-INT8-W8A8"
    results_dir: str = "outputs/results"


@dataclass
class QuantizationConfig:
    ignored_layers: list[str] = field(
        default_factory=lambda: [
            "lm_head",
            "re:.*embed_tokens.*",
            "re:.*visual.*",
            r"re:^mtp\..*",
            r"re:.*linear_attn\.conv1d.*",
        ]
    )
    ignore_linear_attention: bool = False


@dataclass
class FP8Config:
    scheme: str = "FP8_DYNAMIC"
    device: str = "cuda:0"
    max_workers: int = 1


@dataclass
class INT8Config:
    calibration_dataset: str = "HuggingFaceH4/ultrachat_200k"
    calibration_split: str = "train_sft"
    calibration_samples: int = 512
    max_sequence_length: int = 2048
    seed: int = 42
    smoothquant_strength: float = 0.8


@dataclass
class EvaluationConfig:
    seed: int = 42
    temperature: float = 0.0
    max_tokens: int = 256
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    repeat_count: int = 3
    concurrency: list[int] = field(default_factory=lambda: [1, 2, 4])


@dataclass
class ServingConfig:
    standard_vllm_smoke_test: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    fp8: FP8Config = field(default_factory=FP8Config)
    int8: INT8Config = field(default_factory=INT8Config)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def ignored_layers(self, force_linear_attention: bool = False) -> list[str]:
        patterns = list(dict.fromkeys(self.quantization.ignored_layers))
        if self.quantization.ignore_linear_attention or force_linear_attention:
            if LINEAR_ATTENTION_PATTERN not in patterns:
                patterns.append(LINEAR_ATTENTION_PATTERN)
        return patterns


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _set_dotted(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    target = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot override {dotted_key}: {part} is not a mapping")
        target = child
    target[parts[-1]] = value


def _construct(data: dict[str, Any]) -> ExperimentConfig:
    allowed = {
        "model",
        "output",
        "quantization",
        "fp8",
        "int8",
        "evaluation",
        "serving",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {', '.join(unknown)}")
    try:
        config = ExperimentConfig(
            model=ModelConfig(**data.get("model", {})),
            output=OutputConfig(**data.get("output", {})),
            quantization=QuantizationConfig(**data.get("quantization", {})),
            fp8=FP8Config(**data.get("fp8", {})),
            int8=INT8Config(**data.get("int8", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
            serving=ServingConfig(**data.get("serving", {})),
        )
    except TypeError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc
    validate_config(config)
    return config


def load_config(
    path: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    data = _deep_merge(ExperimentConfig().to_dict(), loaded)
    for key, value in (overrides or {}).items():
        if value is not None:
            _set_dotted(data, key, value)
    return _construct(data)


def validate_config(config: ExperimentConfig) -> None:
    if config.fp8.scheme != "FP8_DYNAMIC":
        raise ValueError("This experiment supports only fp8.scheme=FP8_DYNAMIC")
    if config.fp8.max_workers < 1:
        raise ValueError("fp8.max_workers must be at least 1")
    if config.int8.calibration_samples < 1:
        raise ValueError("int8.calibration_samples must be at least 1")
    if config.int8.max_sequence_length < 1:
        raise ValueError("int8.max_sequence_length must be at least 1")
    if not 0 <= config.int8.smoothquant_strength <= 1:
        raise ValueError("int8.smoothquant_strength must be between 0 and 1")
    if config.evaluation.max_tokens < 1 or config.evaluation.max_model_len < 1:
        raise ValueError("evaluation token limits must be positive")
    if not config.evaluation.concurrency or any(
        int(value) < 1 for value in config.evaluation.concurrency
    ):
        raise ValueError("evaluation.concurrency must contain positive integers")
    if config.evaluation.repeat_count < 1:
        raise ValueError("evaluation.repeat_count must be at least 1")
    if config.evaluation.tensor_parallel_size < 1:
        raise ValueError("evaluation.tensor_parallel_size must be at least 1")
    if not 0 < config.evaluation.gpu_memory_utilization <= 1:
        raise ValueError("evaluation.gpu_memory_utilization must be in (0, 1]")
    if config.evaluation.temperature < 0:
        raise ValueError("evaluation.temperature cannot be negative")
    if not config.model.source.strip():
        raise ValueError("model.source cannot be empty")


def cli_overrides(
    *,
    model: str | None = None,
    ignore_linear_attention: bool = False,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if model:
        values["model.source"] = model
    if ignore_linear_attention:
        values["quantization.ignore_linear_attention"] = True
    return values


def comma_list(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]
