from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_VALIDATION_PROMPT,
    IMPORTANT_TENSOR_MARKERS,
    PROCESSOR_FILES,
    QUANTIZATION_CONFIG_KEYS,
    TOKENIZER_FILES,
    VISION_FILE_MARKERS,
)
from .model_utils import (
    ensure_vllm_multiprocessing_spawn,
    load_config_json,
    match_module_pattern,
    package_version,
    release_gpu_memory,
    write_json,
)


def _result(status: str, detail: Any = None) -> dict[str, Any]:
    return {"status": status, "detail": detail}


def _weight_files(files: list[dict[str, Any]]) -> list[str]:
    return [
        item["name"]
        for item in files
        if item["name"].endswith(".safetensors")
        or item["name"].endswith(".safetensors.index.json")
    ]


def quantization_metadata(config: dict[str, Any]) -> dict[str, Any] | None:
    for key in QUANTIZATION_CONFIG_KEYS:
        value = config.get(key)
        if value:
            return {"key": key, "value": value}
    return None


def tensor_index(source: str) -> tuple[set[str], str | None]:
    """Return tensor names without materializing tensor values."""
    source_path = Path(source).expanduser()
    config, files = load_config_json(source)
    del config
    names = {item["name"] for item in files}
    index_name = next(
        (name for name in names if name.endswith(".safetensors.index.json")), None
    )
    if index_name:
        if source_path.exists():
            index_path = source_path / index_name
        else:
            try:
                from huggingface_hub import hf_hub_download

                index_path = Path(hf_hub_download(source, index_name))
            except Exception as exc:
                return set(), f"could not download tensor index: {exc}"
        with index_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        weight_map = value.get("weight_map", {})
        if not weight_map:
            return set(), f"{index_name} has an empty weight_map"
        index_dir = str(Path(index_name).parent)
        prefix = "" if index_dir == "." else index_dir + "/"
        shards = sorted({prefix + shard for shard in weight_map.values()})
        missing_shards = [shard for shard in shards if shard not in names]
        if missing_shards:
            return set(), f"{index_name} references missing shards: {missing_shards}"
        if source_path.exists():
            # Do not trust the index alone: read each local shard header and
            # confirm that the index and the real tensors agree.
            actual: dict[str, str] = {}
            for shard in shards:
                try:
                    for key in safetensors_header_keys(source_path / shard):
                        actual[key] = shard
                except Exception as exc:
                    return set(), f"cannot read shard {shard}: {exc}"
            expected = {key: prefix + shard for key, shard in weight_map.items()}
            # Extra unindexed tensors are tolerated; indexed tensors that are
            # absent or live in a different shard would break loading.
            not_in_shards = sorted(set(expected) - set(actual))
            wrong_shard = sorted(
                key for key in set(expected) & set(actual) if expected[key] != actual[key]
            )
            if not_in_shards or wrong_shard:
                return set(), (
                    "tensor index does not match shard contents: "
                    f"indexed_but_absent={not_in_shards[:20]}, "
                    f"wrong_shard={wrong_shard[:20]}"
                )
        return set(weight_map), None

    shards = [name for name in names if name.endswith(".safetensors")]
    if not shards:
        return set(), "no safetensors files found"
    if not source_path.exists():
        return set(), "remote unsharded checkpoint has no index; shard was not downloaded"
    tensors: set[str] = set()
    for shard in shards:
        try:
            tensors.update(safetensors_header_keys(source_path / shard))
        except Exception as exc:
            return set(), f"cannot read shard {shard}: {exc}"
    return tensors, None


def safetensors_header_keys(path: str | Path) -> list[str]:
    """Read tensor names from a safetensors header without loading tensor data.

    Format: 8-byte little-endian header length, then a JSON header. This also
    detects truncated or non-safetensors files.
    """
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError("file is too small to be safetensors")
        length = int.from_bytes(raw_length, "little")
        if length <= 0 or 8 + length > size:
            raise ValueError("invalid safetensors header length")
        header = json.loads(handle.read(length).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    data_size = size - 8 - length
    keys = []
    for key, info in header.items():
        if key == "__metadata__":
            continue
        offsets = info.get("data_offsets") if isinstance(info, dict) else None
        if not offsets or len(offsets) != 2 or offsets[1] > data_size:
            raise ValueError(f"tensor {key} points outside the file (truncated shard?)")
        keys.append(key)
    return keys


def _weight_map(source: str) -> dict[str, str]:
    source_path = Path(source).expanduser()
    _, files = load_config_json(source)
    names = {item["name"] for item in files}
    index_name = next(
        (name for name in names if name.endswith(".safetensors.index.json")), None
    )
    if index_name:
        if source_path.exists():
            index_path = source_path / index_name
        else:
            from huggingface_hub import hf_hub_download

            index_path = Path(hf_hub_download(source, index_name))
        with index_path.open("r", encoding="utf-8") as handle:
            return dict(json.load(handle).get("weight_map", {}))
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required for tensor header checks") from exc
    mapping: dict[str, str] = {}
    for shard in sorted(name for name in names if name.endswith(".safetensors")):
        if not source_path.exists():
            raise RuntimeError("remote unsharded checkpoint has no tensor index")
        with safe_open(source_path / shard, framework="pt", device="cpu") as handle:
            mapping.update({name: shard for name in handle.keys()})
    return mapping


def tensor_headers(source: str, names: set[str]) -> dict[str, dict[str, Any]]:
    """Read dtype and shape from safetensors headers without loading values."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required for tensor header checks") from exc
    mapping = _weight_map(source)
    by_shard: dict[str, list[str]] = {}
    for name in names:
        shard = mapping.get(name)
        if shard:
            by_shard.setdefault(shard, []).append(name)
    source_path = Path(source).expanduser()
    headers: dict[str, dict[str, Any]] = {}
    for shard, shard_names in by_shard.items():
        if source_path.exists():
            shard_path = source_path / shard
        else:
            from huggingface_hub import hf_hub_download

            shard_path = Path(hf_hub_download(source, shard))
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for name in shard_names:
                view = handle.get_slice(name)
                headers[name] = {
                    "dtype": str(view.get_dtype()),
                    "shape": list(view.get_shape()),
                }
    return headers


def compare_ignored_tensor_precision(
    source: str, checkpoint: str | Path
) -> dict[str, Any]:
    try:
        source_names, source_error = tensor_index(source)
        output_names, output_error = tensor_index(str(checkpoint))
        if source_error or output_error:
            raise RuntimeError(f"source={source_error}; output={output_error}")
        with (Path(checkpoint) / "config.json").open("r", encoding="utf-8") as handle:
            output_config = json.load(handle)
        quant_metadata = quantization_metadata(output_config) or {}
        ignore_patterns = quant_metadata.get("value", {}).get("ignore", [])
        experiment_path = Path(checkpoint) / "experiment_metadata.json"
        if experiment_path.is_file():
            try:
                with experiment_path.open("r", encoding="utf-8") as handle:
                    experiment = json.load(handle)
                configured_patterns = experiment.get("ignored_module_patterns")
                if isinstance(configured_patterns, list):
                    ignore_patterns = configured_patterns
            except Exception:
                pass
        ignored_names = {
            name
            for name in source_names
            if any(
                match_module_pattern(name.rsplit(".", 1)[0], pattern)
                or match_module_pattern(name, pattern)
                for pattern in ignore_patterns
            )
        }
        # Vision preservation is a hard requirement even if a converter expands
        # or rewrites the ignore list to concrete module names.
        vision_names = {name for name in source_names if "visual" in name.lower()}
        checked_names = ignored_names | vision_names
        if not checked_names:
            return {
                "status": "pass",
                "detail": {
                    "ignore_patterns": ignore_patterns,
                    "ignored_tensors_checked": 0,
                    "vision_tensors_checked": 0,
                    "missing_ignored_tensors": [],
                    "dtype_or_shape_mismatches": {},
                    "missing_vision_tensors": [],
                    "vision_dtype_or_shape_mismatches": {},
                },
            }
        missing = sorted(checked_names - output_names)
        source_headers = tensor_headers(source, checked_names)
        output_headers = tensor_headers(str(checkpoint), checked_names & output_names)
        mismatches = {
            name: {"source": source_headers.get(name), "output": output_headers.get(name)}
            for name in sorted(checked_names & output_names)
            if source_headers.get(name) != output_headers.get(name)
        }
        vision_missing = sorted(vision_names - output_names)
        vision_mismatches = {
            name: detail for name, detail in mismatches.items() if name in vision_names
        }
        return {
            "status": "fail" if missing or mismatches else "pass",
            "detail": {
                "ignore_patterns": ignore_patterns,
                "ignored_tensors_checked": len(ignored_names),
                "vision_tensors_checked": len(vision_names),
                "missing_ignored_tensors": missing,
                "dtype_or_shape_mismatches": mismatches,
                "missing_vision_tensors": vision_missing,
                "vision_dtype_or_shape_mismatches": vision_mismatches,
            },
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def inspect_checkpoint_files(
    checkpoint: str | Path, source: str | None = None
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        return {"checkpoint_directory": _result("fail", f"not found: {checkpoint}")}
    config_path = checkpoint / "config.json"
    config: dict[str, Any] = {}
    if config_path.is_file():
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            config_check = _result("pass")
        except Exception as exc:
            config_check = _result("error", str(exc))
    else:
        config_check = _result("fail", "config.json is missing")
    all_names = {str(item.relative_to(checkpoint)) for item in checkpoint.rglob("*") if item.is_file()}
    weights = [
        name
        for name in all_names
        if name.endswith(".safetensors") or name.endswith(".safetensors.index.json")
    ]
    tokenizer = sorted(all_names.intersection(TOKENIZER_FILES))
    processor = sorted(all_names.intersection(PROCESSOR_FILES))
    chat_template_files = sorted(
        name
        for name in all_names
        if Path(name).name in {"chat_template.json", "chat_template.jinja"}
    )
    chat_template = bool(chat_template_files)
    if not chat_template and (checkpoint / "tokenizer_config.json").is_file():
        try:
            with (checkpoint / "tokenizer_config.json").open("r", encoding="utf-8") as handle:
                chat_template = bool(json.load(handle).get("chat_template"))
        except Exception:
            pass

    source_has_vision = False
    if source:
        try:
            source_config, source_files = load_config_json(source)
            source_names = {item["name"] for item in source_files}
            source_has_vision = "vision_config" in source_config or any(
                marker in name for marker in VISION_FILE_MARKERS for name in source_names
            )
        except Exception:
            pass
    return {
        "checkpoint_directory": _result("pass"),
        "config": config_check,
        "quantization_metadata": _result(
            "pass" if quantization_metadata(config) else "fail",
            quantization_metadata(config) or "no quantization/compression config in config.json",
        ),
        "safetensors": _result("pass" if weights else "fail", weights),
        "tokenizer_assets": _result("pass" if tokenizer else "fail", tokenizer),
        "chat_template": _result(
            "pass" if chat_template else "fail", chat_template_files
        ),
        "processor_assets": _result(
            "pass" if processor else ("fail" if source_has_vision else "skip"),
            processor or ("source appears multimodal" if source_has_vision else "text-only source"),
        ),
    }


def compare_tensor_indexes(source: str, checkpoint: str | Path) -> dict[str, Any]:
    try:
        source_names, source_error = tensor_index(source)
    except Exception as exc:
        source_names, source_error = set(), str(exc)
    try:
        output_names, output_error = tensor_index(str(checkpoint))
    except Exception as exc:
        output_names, output_error = set(), str(exc)
    if source_error or output_error:
        return {
            "status": "error",
            "detail": {"source": source_error, "output": output_error},
        }
    missing = sorted(source_names - output_names)
    added = sorted(output_names - source_names)
    scale_markers = ("scale", "zero_point", "weight_shape", "g_idx")
    added_scales = [name for name in added if any(part in name for part in scale_markers)]
    unexpected = [name for name in added if name not in added_scales]
    important_missing = [
        name for name in missing if any(marker in name.lower() for marker in IMPORTANT_TENSOR_MARKERS)
    ]
    source_mtp = sorted(name for name in source_names if name.lower().startswith("mtp."))
    output_mtp = sorted(name for name in output_names if name.lower().startswith("mtp."))
    mtp_missing = [name for name in missing if "mtp" in name.lower()]
    mtp_shard = Path(checkpoint) / "model_mtp.safetensors"
    if source_mtp and not mtp_missing and mtp_shard.is_file():
        mtp_preservation: Any = {
            "status": "preserved_by_upstream_helper",
            "helper": "compressed_tensors.utils.save_mtp_tensors_to_checkpoint",
            "source_tensor_count": len(source_mtp),
            "output_tensor_count": len(output_mtp),
            "shard": mtp_shard.name,
        }
    else:
        mtp_preservation = {
            "status": "present_in_primary_checkpoint" if source_mtp and not mtp_missing else "not_needed",
            "source_tensor_count": len(source_mtp),
            "output_tensor_count": len(output_mtp),
        }
    # Compressed-tensors retains original tensor keys and adds quantization
    # metadata tensors. Therefore, any missing source key is unexplained and
    # unsafe, even when it is not one of the especially important markers.
    status = "fail" if missing else "pass"
    return {
        "status": status,
        "detail": {
            "source_tensor_count": len(source_names),
            "output_tensor_count": len(output_names),
            "present_in_both": len(source_names & output_names),
            "missing_source_tensors": missing,
            "expected_transformed_tensors": [],
            "important_missing_tensors": important_missing,
            "missing_mtp_tensors": mtp_missing,
            "expected_added_scale_tensors": added_scales,
            "unexpected_output_tensors": unexpected,
            "mtp_repair": mtp_preservation,
        },
    }


def transformers_load_checks(
    checkpoint: str | Path, *, trust_remote_code: bool
) -> dict[str, Any]:
    try:
        import transformers
    except ImportError:
        return {"transformers": _result("skip", "transformers is not installed")}
    checks: dict[str, Any] = {}
    is_multimodal = False
    try:
        config = transformers.AutoConfig.from_pretrained(
            checkpoint, trust_remote_code=trust_remote_code
        )
        architecture = " ".join(getattr(config, "architectures", []) or []).lower()
        is_multimodal = hasattr(config, "vision_config") or any(
            marker in architecture
            for marker in ("conditionalgeneration", "vision", "multimodal")
        )
        checks["transformers_config"] = _result(
            "pass",
            {
                "model_type": getattr(config, "model_type", None),
                "architectures": getattr(config, "architectures", None),
            },
        )
    except Exception as exc:
        checks["transformers_config"] = _result("error", str(exc))
    try:
        transformers.AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=trust_remote_code
        )
        checks["transformers_tokenizer"] = _result("pass")
    except Exception as exc:
        checks["transformers_tokenizer"] = _result("error", str(exc))
    try:
        transformers.AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=trust_remote_code
        )
        checks["transformers_processor"] = _result("pass")
    except Exception as exc:
        checks["transformers_processor"] = _result(
            "error" if is_multimodal else "skip", str(exc)
        )
    return checks


def vllm_generation_check(
    checkpoint: str | Path,
    *,
    trust_remote_code: bool,
    max_model_len: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> dict[str, Any]:
    if package_version("vllm") is None:
        return {
            "vllm_initialization": _result("skip", "vllm is not installed"),
            "generation": _result("skip", "vllm is not installed"),
        }
    llm = None
    try:
        ensure_vllm_multiprocessing_spawn()
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(checkpoint),
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=0,
            enforce_eager=True,
            gdn_prefill_backend="triton",
        )
    except Exception as exc:
        return {
            "vllm_initialization": _result("error", str(exc)),
            "generation": _result("skip", "vLLM initialization failed"),
        }
    initialized = _result("pass")
    try:
        output = llm.generate(
            [DEFAULT_VALIDATION_PROMPT],
            SamplingParams(temperature=0.0, max_tokens=8, seed=0),
            use_tqdm=False,
        )
        text = output[0].outputs[0].text if output and output[0].outputs else ""
        generated = _result("pass" if text.strip() else "fail", text)
    except Exception as exc:
        generated = _result("error", str(exc))
    finally:
        if llm is not None:
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel

                destroy_model_parallel()
            except Exception:
                pass
        llm = None
        release_gpu_memory()
    return {"vllm_initialization": initialized, "generation": generated}


def validate_checkpoint(
    source: str,
    checkpoint: str | Path,
    *,
    trust_remote_code: bool = True,
    run_runtime: bool = True,
    repair_mtp: bool = True,
    max_model_len: int = 2048,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
) -> dict[str, Any]:
    checks = inspect_checkpoint_files(checkpoint, source)
    tensor_check = compare_tensor_indexes(source, checkpoint)
    missing_mtp = tensor_check.get("detail", {}).get("missing_mtp_tensors", [])
    if missing_mtp:
        repair_detail: dict[str, Any] = {
            "detected_tensors": missing_mtp,
            "attempted": repair_mtp,
        }
        if repair_mtp:
            try:
                try:
                    from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
                except ImportError:
                    # compressed-tensors 0.18 exposes the public helper from this module.
                    from compressed_tensors.utils.mtp import (
                        save_mtp_tensors_to_checkpoint,
                    )

                save_mtp_tensors_to_checkpoint(source, str(checkpoint))
                tensor_check = compare_tensor_indexes(source, checkpoint)
                remaining = tensor_check.get("detail", {}).get(
                    "missing_mtp_tensors", []
                )
                repair_detail.update(
                    {
                        "status": "pass" if not remaining else "fail",
                        "remaining_tensors": remaining,
                        "helper": "compressed_tensors.utils.save_mtp_tensors_to_checkpoint",
                    }
                )
            except Exception as exc:
                repair_detail.update({"status": "error", "error": str(exc)})
        else:
            repair_detail["status"] = "skip"
        tensor_check.setdefault("detail", {})["mtp_repair"] = repair_detail
    checks["tensor_index"] = tensor_check
    checks["ignored_tensor_precision"] = compare_ignored_tensor_precision(
        source, checkpoint
    )
    checks.update(
        transformers_load_checks(checkpoint, trust_remote_code=trust_remote_code)
    )
    if run_runtime:
        checks.update(
            vllm_generation_check(
                checkpoint,
                trust_remote_code=trust_remote_code,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        )
    else:
        checks["vllm_initialization"] = _result("skip", "runtime checks disabled")
        checks["generation"] = _result("skip", "runtime checks disabled")
    structural_names = (
        "checkpoint_directory",
        "config",
        "quantization_metadata",
        "safetensors",
        "tokenizer_assets",
        "chat_template",
        "processor_assets",
        "tensor_index",
        "ignored_tensor_precision",
    )
    structural_failed = any(
        checks.get(name, {}).get("status") in {"fail", "error"}
        for name in structural_names
    )
    transformer_failures = [
        name
        for name in (
            "transformers_config",
            "transformers_tokenizer",
            "transformers_processor",
            "transformers",
        )
        if checks.get(name, {}).get("status") in {"fail", "error"}
    ]
    warnings: list[str] = []
    if structural_failed:
        overall = "fail"
    elif run_runtime:
        required_runtime = (
            checks.get("vllm_initialization", {}).get("status"),
            checks.get("generation", {}).get("status"),
        )
        if any(status in {"fail", "error"} for status in required_runtime):
            overall = "fail"
        elif required_runtime != ("pass", "pass"):
            overall = "incomplete"
        elif transformer_failures:
            overall = "pass_with_warnings"
            warnings.append(
                "Standard vLLM initialization and generation passed, but separate "
                "Transformers checks failed: " + ", ".join(transformer_failures)
            )
        else:
            overall = "pass"
    elif transformer_failures:
        overall = "fail"
    else:
        overall = "pass"
    report = {
        "overall_status": overall,
        "source": source,
        "checkpoint": str(checkpoint),
        "warnings": warnings,
        "checks": checks,
    }
    if Path(checkpoint).is_dir():
        write_json(Path(checkpoint) / "validation_report.json", report)
        metadata_path = Path(checkpoint) / "experiment_metadata.json"
        if metadata_path.is_file():
            try:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                metadata["validation_status"] = overall
                metadata["runtime_validation_requested"] = run_runtime
                if overall in {"pass", "pass_with_warnings"}:
                    metadata["status"] = (
                        "complete"
                        if run_runtime
                        else "quantized_pending_runtime_validation"
                    )
                elif overall == "incomplete":
                    metadata["status"] = "quantized_pending_runtime_validation"
                else:
                    metadata["status"] = "invalid"
                write_json(metadata_path, metadata)
            except Exception:
                # The validation report remains authoritative if old metadata is malformed.
                pass
    return report
