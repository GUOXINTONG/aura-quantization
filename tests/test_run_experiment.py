import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_experiment  # noqa: E402
from qwen_quant_lab.config import load_config  # noqa: E402


def make_config(directory: Path, trust_remote_code: bool = True):
    path = directory / "config.yaml"
    path.write_text(
        "model:\n"
        "  source: Qwen/Qwen3.5-9B\n"
        f"  trust_remote_code: {str(trust_remote_code).lower()}\n"
        "output:\n"
        f"  fp8_dir: {directory / 'fp8'}\n"
        f"  int8_dir: {directory / 'int8'}\n",
        encoding="utf-8",
    )
    return path, load_config(path)


def make_args(config_path: Path, **values):
    defaults = dict(
        config=str(config_path),
        model=None,
        steps="fp8",
        skip_existing=False,
        overwrite=False,
        dry_run=True,
        ignore_linear_attention=False,
        skip_runtime_validation=False,
    )
    defaults.update(values)
    return argparse.Namespace(**defaults)


class RunExperimentTests(unittest.TestCase):
    def test_validate_passes_trust_remote_code_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            path, config = make_config(Path(directory), trust_remote_code=False)
            commands = run_experiment.planned_commands(
                make_args(path, steps="validate"), config
            )
            for _, command in commands:
                self.assertIn("--no-trust-remote-code", command)

    def test_skip_existing_reuses_only_complete_matching_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, config = make_config(root)
            output = root / "fp8"
            output.mkdir()
            metadata = {
                "status": "complete",
                "source_model": config.model.source,
                "ignored_module_patterns": config.ignored_layers(),
            }
            (output / "experiment_metadata.json").write_text(json.dumps(metadata))
            args = make_args(path, skip_existing=True)
            self.assertEqual(run_experiment.planned_commands(args, config), [])

            metadata["status"] = "error"
            (output / "experiment_metadata.json").write_text(json.dumps(metadata))
            with self.assertRaises(SystemExit):
                run_experiment.planned_commands(args, config)

            metadata.update(status="complete", source_model="/models/other")
            (output / "experiment_metadata.json").write_text(json.dumps(metadata))
            with self.assertRaises(SystemExit):
                run_experiment.planned_commands(args, config)

    def test_incomplete_directory_is_rebuilt_with_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, config = make_config(root)
            (root / "fp8").mkdir()
            (root / "fp8" / "partial.safetensors").write_bytes(b"")
            args = make_args(path, skip_existing=True, overwrite=True)
            commands = run_experiment.planned_commands(args, config)
            self.assertEqual(len(commands), 1)
            self.assertIn("--overwrite", commands[0][1])


if __name__ == "__main__":
    unittest.main()
