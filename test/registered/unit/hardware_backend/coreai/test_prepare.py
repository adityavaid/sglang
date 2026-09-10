"""Standard-loaded native weights -> Core AI execution; CPU reference, not GPU."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.coreai_utils import checkpoint_pair

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def tiny_qwen3():
    return checkpoint_pair()[0]


class TestPrepareLoadedQwen3(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("coreai_torch"), "Core AI extra is not installed"
    )
    def test_real_export_prefill_decode_and_request_reuse(self):
        from coreai.runtime import AIModel, NDArray, SpecializationOptions

        from sglang.srt.hardware_backend.coreai.prepare import prepare_loaded_qwen3
        from sglang.srt.hardware_backend.coreai.session import CoreAISession

        class ReferenceSession(CoreAISession):
            async def _load_model(self):
                options = (
                    SpecializationOptions.cpu_only()
                    if SpecializationOptions.is_supported()
                    else None
                )
                return await AIModel.load(
                    self._bundle / "model.aimodel", specialization_options=options
                )

            def _new_state(self):
                return {
                    s.name: NDArray(np.zeros(s.shape, dtype=s.dtype))
                    for s in self.manifest.states
                }

        model, oracle = checkpoint_pair()
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "qwen3"
            prepare_loaded_qwen3(
                model, bundle, max_context_length=16, prefill_chunk_size=4
            )
            with patch("sglang.srt.hardware_backend.coreai.session.validate_runtime"):
                session = ReferenceSession(bundle)
            try:
                for prompt in ([1], [1, 3, 5, 7, 2], [8, 3]):
                    session.reset()
                    tokens = list(prompt)
                    with torch.no_grad():
                        expected = int(
                            oracle(torch.tensor([tokens])).logits[0, -1].argmax()
                        )
                    actual = session.extend(tokens, 0)
                    self.assertEqual(actual, expected)
                    for _ in range(3):
                        tokens.append(actual)
                        with torch.no_grad():
                            expected = int(
                                oracle(torch.tensor([tokens])).logits[0, -1].argmax()
                            )
                        actual = session.extend([actual], len(tokens) - 1)
                        self.assertEqual(actual, expected)
            finally:
                session.close()

    def test_existing_output_is_not_overwritten(self):
        from sglang.srt.hardware_backend.coreai.prepare import prepare_loaded_qwen3

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            marker = path / "keep.txt"
            marker.write_text("existing work")
            with self.assertRaises(FileExistsError):
                prepare_loaded_qwen3(tiny_qwen3(), path)
            self.assertEqual(marker.read_text(), "existing work")

    def test_invalid_prefill_size_is_rejected_before_export(self):
        from sglang.srt.hardware_backend.coreai.prepare import prepare_loaded_qwen3

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen3"
            with self.assertRaisesRegex(ValueError, "prefill"):
                prepare_loaded_qwen3(
                    tiny_qwen3(), path, max_context_length=16, prefill_chunk_size=32
                )
            self.assertFalse(path.exists())

    def test_unsupported_dtype_is_rejected_before_export(self):
        from sglang.srt.hardware_backend.coreai.prepare import prepare_loaded_qwen3

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen3"
            with (
                patch(
                    "sglang.srt.hardware_backend.coreai.prepare._load_coreai_dependencies",
                    side_effect=ImportError("optional converter unavailable"),
                ),
                self.assertRaisesRegex(ValueError, "float16 or float32"),
            ):
                prepare_loaded_qwen3(
                    tiny_qwen3().to(torch.bfloat16),
                    path,
                    max_context_length=16,
                    prefill_chunk_size=4,
                )
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
