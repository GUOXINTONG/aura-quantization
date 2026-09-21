from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpoint import tensor_index
from .model_utils import (
    ensure_source_output_separate,
    evaluate_ignore_patterns,
    load_config_json,
    print_quantization_selection,
    write_json,
)


def infer_linear_modules(tensor_names: set[str]) -> list[str]:
    """Infer likely Linear module names from a tensor index.

    Safetensors indexes contain names but not shapes, so this intentionally uses
    projection/name conventions. Exact module types are printed after model load
    by each quantizer.
    """
    markers = (
        "proj",
        "fc",
        "linear",
        "lm_head",
        "classifier",
        "router",
        "gate",
        "experts",
    )
    modules = []
    for tensor in tensor_names:
        if not tensor.endswith(".weight"):
            continue
        module = tensor[: -len(".weight")]
        lowered = module.lower()
        if any(marker in lowered for marker in markers):
            modules.append(module)
    return sorted(set(modules))


def inspect_source(
    source: str,
    *,
    ignore_patterns: list[str],
    output_paths: list[str] | None = None,
    report_path: str | Path | None = None,
    print_details: bool = True,
) -> dict[str, Any]:
    if output_paths:
        ensure_source_output_separate(source, output_paths)
    local_path = Path(source).expanduser()
    if (
        local_path.is_dir()
        and (local_path / "adapter_config.json").is_file()
        and not (local_path / "config.json").is_file()
    ):
        raise ValueError(
            "The local source contains a LoRA adapter but no merged model config. "
            "Merge it with its BF16 base checkpoint before offline quantization."
        )
    config, files = load_config_json(source)
    names = {item["name"] for item in files}
    adapter_files = sorted(
        name for name in names if name in {"adapter_config.json", "adapter_model.safetensors"}
    )
    weight_files = sorted(
        name
        for name in names
        if name.endswith(".safetensors") or name.endswith(".safetensors.index.json")
    )
    tensor_names, index_error = tensor_index(source)
    modules = infer_linear_modules(tensor_names)
    architecture = config.get("architectures") or []
    model_type = str(config.get("model_type", ""))
    identity = " ".join([model_type, *map(str, architecture)]).lower()
    appears_qwen = "qwen" in identity
    if adapter_files and not any(
        name.startswith("model") and name.endswith(".safetensors") for name in names
    ):
        raise ValueError(
            "The local source appears to contain only a LoRA adapter. Merge it "
            "with its BF16 base checkpoint before offline quantization."
        )
    if not weight_files:
        raise ValueError("The source has no safetensors checkpoint or safetensors index")
    if index_error:
        raise ValueError(f"Could not inspect source tensors: {index_error}")
    if not appears_qwen:
        raise ValueError(
            f"Source does not appear to be a Qwen checkpoint (model_type={model_type!r})"
        )
    if not modules:
        raise ValueError("No likely linear quantization targets were found in the tensor index")
    selection = (
        print_quantization_selection(modules, ignore_patterns)
        if print_details
        else _selection(modules, ignore_patterns)
    )
    report = {
        "source": source,
        "model_class": architecture[0] if architecture else None,
        "architectures": architecture,
        "model_type": model_type,
        "appears_qwen": appears_qwen,
        "is_local": local_path.exists(),
        "appears_lora_adapter": bool(adapter_files),
        "adapter_files": adapter_files,
        "safetensors_files": weight_files,
        "estimated_checkpoint_bytes": sum(
            item.get("size") or 0
            for item in files
            if item["name"].endswith(".safetensors")
        ),
        "tensor_count": len(tensor_names),
        "likely_linear_modules": modules,
        "linear_module_detection": "tensor-name heuristic; exact types are checked after model load",
        "selection": selection,
    }
    if report_path:
        write_json(report_path, report)
    return report


def _selection(module_names: list[str], patterns: list[str]) -> dict[str, Any]:
    matches = evaluate_ignore_patterns(module_names, patterns)
    ignored = {name for values in matches.values() for name in values}
    selected = len(module_names) - len(ignored)
    if selected == 0:
        raise ValueError("No target linear modules remain after applying ignore patterns")
    linear_attention = [name for name in module_names if "linear_attn" in name.lower()]
    vision = [
        name
        for name in module_names
        if "visual" in name.lower() or "vision" in name.lower()
    ]
    return {
        "patterns": matches,
        "total_linear_modules": len(module_names),
        "selected_for_quantization": selected,
        "ignored_modules": len(ignored),
        "ignored_vision_modules": len(set(vision) & ignored),
        "lm_head_found": any(name.endswith("lm_head") for name in module_names),
        "lm_head_ignored": any(name.endswith("lm_head") for name in ignored),
        "linear_attention_found": bool(linear_attention),
        "linear_attention_will_be_quantized": any(name not in ignored for name in linear_attention),
    }
