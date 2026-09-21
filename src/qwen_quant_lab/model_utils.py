from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_vllm_multiprocessing_spawn() -> None:
    """Force vLLM worker processes to use spawn instead of fork.

    Offline ``LLM()`` defaults to fork on Linux. That breaks when CUDA is
    already initialized in the parent process.
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def require(module: str, purpose: str | None = None) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        message = f"Missing required package '{module}'"
        if purpose:
            message += f" for {purpose}"
        message += ". Install the appropriate requirements file."
        raise RuntimeError(message) from exc


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def collect_environment_metadata() -> dict[str, Any]:
    packages = {
        name: package_version(distribution)
        for name, distribution in {
            "torch": "torch",
            "transformers": "transformers",
            "llmcompressor": "llmcompressor",
            "compressed_tensors": "compressed-tensors",
            "datasets": "datasets",
            "vllm": "vllm",
        }.items()
    }
    cuda_runtime = None
    gpu_name = None
    gpu_capability = None
    try:
        torch = importlib.import_module("torch")
        cuda_runtime = getattr(getattr(torch, "version", None), "cuda", None)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            gpu_capability = f"{capability[0]}.{capability[1]}"
    except (ImportError, RuntimeError):
        pass

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    # nvidia-smi uses physical ordinals, while torch remaps a CUDA visibility
    # mask to logical cuda:0. Query the first explicitly visible physical GPU.
    nvidia_selector = "0"
    if visible_devices:
        first_visible = visible_devices.split(",", 1)[0].strip()
        if first_visible:
            nvidia_selector = first_visible
    query = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader",
            "-i",
            nvidia_selector,
        ]
    )
    gpu_uuid = None
    driver_version = None
    if query:
        fields = [part.strip() for part in query.splitlines()[0].split(",")]
        if len(fields) >= 3:
            gpu_name = gpu_name or fields[0]
            gpu_uuid, driver_version = fields[1], fields[2]

    git_commit = _command_output(["git", "rev-parse", "HEAD"])
    return {
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
        },
        "cuda_visible_devices": visible_devices,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "packages": packages,
        "cuda_runtime_version": cuda_runtime,
        "nvidia_driver_version": driver_version,
        "gpu": {
            "name": gpu_name,
            "uuid": gpu_uuid,
            "compute_capability": gpu_capability,
        },
        "command": sys.argv,
        "git_commit": git_commit,
    }


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def is_local_source(source: str) -> bool:
    return Path(source).expanduser().exists()


def ensure_source_output_separate(source: str, outputs: Iterable[str | Path]) -> None:
    if not is_local_source(source):
        return
    source_path = Path(source).expanduser().resolve()
    for output in outputs:
        output_path = Path(output).expanduser().resolve()
        if output_path == source_path:
            raise ValueError(f"Output directory is the source checkpoint: {output_path}")
        if source_path in output_path.parents or output_path in source_path.parents:
            raise ValueError(
                "Source and output directories overlap, which is unsafe: "
                f"source={source_path}, output={output_path}"
            )


def prepare_output_directory(
    output: str | Path,
    *,
    source: str,
    overwrite: bool,
) -> Path:
    ensure_source_output_separate(source, [output])
    path = Path(output).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Pass --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def match_module_pattern(module_name: str, pattern: str) -> bool:
    if pattern.startswith("re:"):
        return re.search(pattern[3:], module_name) is not None
    return module_name == pattern or module_name.endswith(f".{pattern}")


def evaluate_ignore_patterns(
    module_names: Iterable[str], patterns: Iterable[str]
) -> dict[str, list[str]]:
    names = sorted(set(module_names))
    return {
        pattern: [name for name in names if match_module_pattern(name, pattern)]
        for pattern in patterns
    }


def print_quantization_selection(
    module_names: Iterable[str], patterns: Iterable[str]
) -> dict[str, Any]:
    names = sorted(set(module_names))
    matches = evaluate_ignore_patterns(names, patterns)
    ignored = {name for values in matches.values() for name in values}
    linear_attention = [name for name in names if "linear_attn" in name.lower()]
    vision = [
        name
        for name in names
        if "visual" in name.lower() or "vision" in name.lower()
    ]
    print("Ignore patterns and matches:")
    for pattern, values in matches.items():
        print(f"  {pattern}: {len(values)}")
        for name in values:
            print(f"    - {name}")
    summary = {
        "total_linear_modules": len(names),
        "selected_for_quantization": len(names) - len(ignored),
        "ignored_modules": len(ignored),
        "ignored_vision_modules": len(set(vision) & ignored),
        "lm_head_found": any(name.endswith("lm_head") for name in names),
        "lm_head_ignored": any(name.endswith("lm_head") for name in ignored),
        "linear_attention_found": bool(linear_attention),
        "linear_attention_will_be_quantized": any(
            name not in ignored for name in linear_attention
        ),
    }
    print_json(summary)
    if summary["selected_for_quantization"] == 0:
        raise ValueError("No target linear modules remain after applying ignore patterns")
    return {"patterns": matches, **summary}


def load_config_json(source: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read config and list files without downloading checkpoint shards."""
    path = Path(source).expanduser()
    if path.exists():
        config_path = path / "config.json" if path.is_dir() else path
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config.json in local checkpoint: {path}")
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        root = config_path.parent
        files = [
            {"name": str(item.relative_to(root)), "size": item.stat().st_size}
            for item in root.rglob("*")
            if item.is_file()
        ]
        return config, files

    hub = require("huggingface_hub", "remote checkpoint inspection")
    api = hub.HfApi()
    try:
        info = api.model_info(source, files_metadata=True)
        config_path = hub.hf_hub_download(source, "config.json")
    except Exception as exc:
        raise RuntimeError(f"Could not inspect Hugging Face model '{source}': {exc}") from exc
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    files = [
        {"name": sibling.rfilename, "size": getattr(sibling, "size", None)}
        for sibling in info.siblings
    ]
    return config, files


def choose_model_class(transformers: Any, config: Any) -> Any:
    architecture = " ".join(getattr(config, "architectures", []) or []).lower()
    model_type = str(getattr(config, "model_type", "")).lower()
    multimodal = any(
        marker in architecture or marker in model_type
        for marker in ("conditionalgeneration", "vision", "vl", "multimodal")
    ) or hasattr(config, "vision_config")
    if multimodal and hasattr(transformers, "AutoModelForImageTextToText"):
        return transformers.AutoModelForImageTextToText
    if hasattr(transformers, "AutoModelForCausalLM"):
        return transformers.AutoModelForCausalLM
    raise RuntimeError("Installed Transformers has no compatible auto model class")


def load_model_and_processor(source: str, *, trust_remote_code: bool, dtype: str):
    transformers = require("transformers", "model loading")
    config = transformers.AutoConfig.from_pretrained(
        source, trust_remote_code=trust_remote_code
    )
    model_class = choose_model_class(transformers, config)
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "device_map": "auto",
    }
    if not dtype or dtype == "auto":
        # Avoid accidentally expanding a BF16 checkpoint to FP32 in memory.
        kwargs["torch_dtype"] = "auto"
    else:
        torch = require("torch", "model dtype selection")
        aliases = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
        dtype_name = aliases.get(dtype.lower(), dtype.lower())
        if not hasattr(torch, dtype_name):
            raise ValueError(f"Unsupported model dtype: {dtype}")
        kwargs["torch_dtype"] = getattr(torch, dtype_name)
    model = model_class.from_pretrained(source, **kwargs)
    try:
        processor = transformers.AutoProcessor.from_pretrained(
            source, trust_remote_code=trust_remote_code
        )
    except Exception:
        processor = transformers.AutoTokenizer.from_pretrained(
            source, trust_remote_code=trust_remote_code
        )
    return model, processor, config


def preserve_model_assets(
    source: str, output: str | Path, *, trust_remote_code: bool
) -> dict[str, Any]:
    """Save lightweight processor/tokenizer/generation assets, never model weights."""
    transformers = require("transformers", "saving tokenizer and processor assets")
    output = Path(output)
    status: dict[str, Any] = {}
    try:
        processor = transformers.AutoProcessor.from_pretrained(
            source, trust_remote_code=trust_remote_code
        )
        processor.save_pretrained(output)
        status["processor"] = "saved"
    except Exception as exc:
        status["processor"] = f"unavailable: {exc}"
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            source, trust_remote_code=trust_remote_code
        )
        tokenizer.save_pretrained(output)
        status["tokenizer"] = "saved"
    except Exception as exc:
        status["tokenizer"] = f"unavailable: {exc}"
    try:
        generation = transformers.GenerationConfig.from_pretrained(source)
        generation.save_pretrained(output)
        status["generation_config"] = "saved"
    except Exception as exc:
        status["generation_config"] = f"unavailable: {exc}"
    return status


def directory_size(path: str | Path) -> int:
    root = Path(path)
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def release_gpu_memory(*objects: Any) -> None:
    for obj in objects:
        del obj
    try:
        import gc

        gc.collect()
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
