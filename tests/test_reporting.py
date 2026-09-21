import json
from pathlib import Path
import tempfile
import unittest

from qwen_quant_lab.reporting import summarize_regression, write_comparison_outputs


class ReportingTests(unittest.TestCase):
    def test_regression_and_files(self):
        records = [
            {"model_variant": "bf16", "prompt_id": "a", "output": "Hello, world!"},
            {"model_variant": "fp8", "prompt_id": "a", "output": "hello world"},
        ]
        summary = summarize_regression(records)
        self.assertEqual(summary["fp8"]["exact_match"], 0.0)
        self.assertEqual(summary["fp8"]["normalized_match"], 1.0)
        with tempfile.TemporaryDirectory() as directory:
            write_comparison_outputs(directory, summary)
            root = Path(directory)
            self.assertTrue((root / "comparison.json").is_file())
            self.assertTrue((root / "comparison.csv").is_file())
            self.assertIn("regression signals", (root / "report.md").read_text(encoding="utf-8"))
            self.assertIn("fp8", json.loads((root / "comparison.json").read_text()))


if __name__ == "__main__":
    unittest.main()

