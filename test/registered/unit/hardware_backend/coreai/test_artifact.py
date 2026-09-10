import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def write_bundle(path: Path):
    config = {"model_type": "qwen3", "vocab_size": 32}
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.aimodel").mkdir()
    document = {
        "schema_version": 1,
        "model_type": "qwen3",
        "context_length": 16,
        "prefill_chunk_size": 4,
        "vocab_size": 32,
        "dtype": "float32",
        "model_config_sha256": hashlib.sha256(
            (path / "config.json").read_bytes()
        ).hexdigest(),
        "states": [
            {"name": name, "shape": [1, 2, 16, 8], "dtype": "float32"}
            for name in ("keys", "values")
        ]
        + [{"name": "next_token", "shape": [1], "dtype": "int32"}],
        "packages": {
            "torch": "2.13.0",
            "coreai-core": "1.0.0b2",
            "coreai-torch": "0.4.2",
        },
        "source_model": "tiny-qwen3",
        "source_revision": None,
    }
    (path / "coreai-manifest.json").write_text(json.dumps(document))
    return document


class TestCoreAIArtifact(unittest.TestCase):
    def test_loads_the_serving_contract(self):
        from sglang.srt.hardware_backend.coreai.artifact import load_manifest

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_bundle(path)
            manifest = load_manifest(path)
            self.assertEqual(manifest.context_length, 16)
            self.assertEqual(manifest.states[0].shape, (1, 2, 16, 8))
            self.assertEqual(manifest.prefill_chunk_size, 4)

    def test_rejects_a_different_model_config(self):
        from sglang.srt.hardware_backend.coreai.artifact import load_manifest

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_bundle(path)
            (path / "config.json").write_text('{"model_type":"qwen3","vocab_size":64}')
            with self.assertRaisesRegex(ValueError, "config"):
                load_manifest(path)

    def test_rejects_incompatible_or_invalid_contracts(self):
        from sglang.srt.hardware_backend.coreai.artifact import load_manifest

        changes = (
            {"schema_version": 2},
            {"model_type": "qwen3_moe"},
            {"context_length": 0},
            {"context_length": True},
            {"prefill_chunk_size": 17},
            {"vocab_size": 0},
            {"dtype": "int4"},
            {"states": []},
            {"states": [{"name": "next_token", "shape": [1], "dtype": "int32"}]},
            {"states": [{"name": "x", "shape": [1, 2, -1, 8], "dtype": "float32"}]},
            {"packages": {"coreai-core": "wrong"}},
            {"packages": None},
        )
        for change in changes:
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory)
                document = write_bundle(path)
                document.update(change)
                (path / "coreai-manifest.json").write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    load_manifest(path)

    def test_rejects_a_missing_compiled_asset(self):
        from sglang.srt.hardware_backend.coreai.artifact import load_manifest

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_bundle(path)
            (path / "model.aimodel").rmdir()
            with self.assertRaisesRegex(ValueError, "aimodel"):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
