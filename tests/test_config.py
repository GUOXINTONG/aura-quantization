from pathlib import Path
import tempfile
import unittest

from qwen_quant_lab.config import LINEAR_ATTENTION_PATTERN, load_config


class ConfigTests(unittest.TestCase):
    def test_cli_style_overrides_win(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "model:\n  source: old/model\nint8:\n  calibration_samples: 32\n",
                encoding="utf-8",
            )
            config = load_config(
                path,
                overrides={
                    "model.source": "/models/new",
                    "quantization.ignore_linear_attention": True,
                },
            )
        self.assertEqual(config.model.source, "/models/new")
        self.assertEqual(config.int8.calibration_samples, 32)
        self.assertIn(LINEAR_ATTENTION_PATTERN, config.ignored_layers())

    def test_invalid_calibration_count_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("int8:\n  calibration_samples: 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "calibration_samples"):
                load_config(path)

    def test_zero_repeat_count_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("evaluation:\n  repeat_count: 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "repeat_count"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()

