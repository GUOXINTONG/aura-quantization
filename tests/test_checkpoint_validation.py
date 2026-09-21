import json
from pathlib import Path
import tempfile
import unittest

from qwen_quant_lab.checkpoint import compare_tensor_indexes, validate_checkpoint


def make_checkpoint(path: Path, tensors: list[str], quantized: bool) -> None:
    config = {"model_type": "qwen", "architectures": ["QwenForCausalLM"]}
    if quantized:
        config["quantization_config"] = {"format": "packed-quantized"}
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"placeholder")
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: "model.safetensors" for name in tensors},
            }
        ),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8"
    )
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


class CheckpointTests(unittest.TestCase):
    def test_structure_and_tensor_index_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            source.mkdir()
            output.mkdir()
            tensors = [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.norm.weight",
                "lm_head.weight",
            ]
            make_checkpoint(source, tensors, quantized=False)
            make_checkpoint(output, tensors + ["model.layers.0.self_attn.q_proj.weight_scale"], quantized=True)
            report = validate_checkpoint(
                str(source), output, run_runtime=False, trust_remote_code=False
            )
            self.assertEqual(report["overall_status"], "pass")
            self.assertEqual(report["checks"]["tensor_index"]["status"], "pass")

    def test_important_missing_tensor_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            source.mkdir()
            output.mkdir()
            make_checkpoint(source, ["model.visual.proj.weight", "lm_head.weight"], False)
            make_checkpoint(output, ["lm_head.weight"], True)
            result = compare_tensor_indexes(str(source), output)
            self.assertEqual(result["status"], "fail")
            self.assertEqual(
                result["detail"]["important_missing_tensors"],
                ["model.visual.proj.weight"],
            )

    def test_any_unexplained_missing_tensor_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            source.mkdir()
            output.mkdir()
            make_checkpoint(source, ["model.layers.0.self_attn.q_proj.weight"], False)
            make_checkpoint(output, [], True)
            result = compare_tensor_indexes(str(source), output)
            self.assertEqual(result["status"], "fail")
            self.assertEqual(
                result["detail"]["missing_source_tensors"],
                ["model.layers.0.self_attn.q_proj.weight"],
            )


if __name__ == "__main__":
    unittest.main()
