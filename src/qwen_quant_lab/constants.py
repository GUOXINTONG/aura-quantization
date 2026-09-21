from __future__ import annotations

LINEAR_ATTENTION_PATTERN = "re:.*linear_attn.*"
IMPORTANT_TENSOR_MARKERS = ("visual", "embed_tokens", "lm_head", "norm")
QUANTIZATION_CONFIG_KEYS = (
    "quantization_config",
    "compression_config",
    "quantization_status",
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)
PROCESSOR_FILES = (
    "processor_config.json",
    "preprocessor_config.json",
    "image_processor_config.json",
)
VISION_FILE_MARKERS = ("preprocessor", "image_processor", "video_processor")
DEFAULT_VALIDATION_PROMPT = "Reply with exactly: quantization check"

