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

    def test_invalid_json_counts_as_key_failure(self):
        keys = ["a"]
        records = [
            {"model_variant": "bf16", "prompt_id": "j1", "output": '{"a": 1}', "expected_keys": keys},
            {"model_variant": "bf16", "prompt_id": "j2", "output": '{"a": 2}', "expected_keys": keys},
            {"model_variant": "fp8", "prompt_id": "j1", "output": '{"a": 1}', "expected_keys": keys},
            {"model_variant": "fp8", "prompt_id": "j2", "output": "not json", "expected_keys": keys},
        ]
        summary = summarize_regression(records)
        self.assertEqual(summary["fp8"]["json_parse_success"], 0.5)
        self.assertEqual(summary["fp8"]["expected_key_presence"], 0.5)

    def test_failed_or_missing_generation_is_not_a_match(self):
        records = [
            {"model_variant": "bf16", "prompt_id": "a", "output": "x", "success": True},
            {"model_variant": "bf16", "prompt_id": "b", "output": "y", "success": True},
            {"model_variant": "fp8", "prompt_id": "a", "output": "", "success": False},
        ]
        summary = summarize_regression(records)
        self.assertEqual(summary["fp8"]["exact_match"], 0.0)
        self.assertEqual(summary["fp8"]["generation_success_rate"], 0.0)
        self.assertEqual(summary["fp8"]["missing_prompts"], 1)
        self.assertEqual(summary["fp8"]["failed_generations"], 1)

    def test_both_failed_is_not_reported_as_identical(self):
        records = [
            {"model_variant": "bf16", "prompt_id": "a", "output": "", "success": False},
            {"model_variant": "fp8", "prompt_id": "a", "output": "", "success": False},
        ]
        summary = summarize_regression(records)
        self.assertEqual(summary["fp8"]["exact_match"], 0.0)
        self.assertEqual(summary["fp8"]["generation_success_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()

