import json
from pathlib import Path
import tempfile
import unittest

from qwen_quant_lab.inspection import inspect_source


class InspectionTests(unittest.TestCase):
    def test_standalone_lora_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "LoRA adapter"):
                inspect_source(root.as_posix(), ignore_patterns=["lm_head"])

    def test_qwen_tensor_index_is_inspected_without_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps(
                    {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
                ),
                encoding="utf-8",
            )
            header = {
                name: {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}
                for name in ("model.layers.0.self_attn.q_proj.weight", "lm_head.weight")
            }
            raw = json.dumps(header).encode("utf-8")
            (root / "model.safetensors").write_bytes(len(raw).to_bytes(8, "little") + raw)
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
                            "lm_head.weight": "model.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            report = inspect_source(
                root.as_posix(), ignore_patterns=["lm_head"], print_details=False
            )
            self.assertTrue(report["appears_qwen"])
            self.assertEqual(report["selection"]["selected_for_quantization"], 1)


if __name__ == "__main__":
    unittest.main()

