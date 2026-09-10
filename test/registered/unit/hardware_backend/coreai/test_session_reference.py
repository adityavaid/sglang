"""Real Core AI interpreter execution, NOT macOS 27 GPU qualification."""

import hashlib
import importlib.metadata
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _StatefulTokenStep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cache", torch.zeros(1))
        self.register_buffer("next_token", torch.zeros(1, dtype=torch.int32))

    def forward(self, input_ids, start_position):
        previous = torch.where(
            start_position == 0, torch.zeros_like(self.cache), self.cache
        )
        self.cache.copy_(previous + input_ids.float().sum())
        self.next_token.copy_(self.cache.to(torch.int32))
        return ()


@unittest.skipUnless(
    importlib.util.find_spec("coreai_torch"), "Core AI extra is not installed"
)
class TestCoreAISessionReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from coreai_torch import TorchConverter, get_decomp_table

        from sglang.srt.hardware_backend.coreai.artifact import (
            CoreAIManifest,
            StateSpec,
            save_manifest,
        )

        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.bundle = Path(cls.directory.name)
        (cls.bundle / "config.json").write_text(
            json.dumps({"model_type": "qwen3", "vocab_size": 32})
        )
        module = _StatefulTokenStep().eval()
        converter = TorchConverter()
        for name, length in (("decode", 1), ("prefill", 4)):
            exported = torch.export.export(
                module,
                (
                    torch.zeros((1, length), dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                ),
            ).run_decompositions(get_decomp_table())
            converter.add_exported_program(
                exported,
                input_names=("input_ids", "start_position"),
                state_names=("cache", "next_token"),
                output_names=(),
                entrypoint_name=name,
            )
        program = converter.to_coreai()
        program.optimize()
        program.save_asset(cls.bundle / "model.aimodel")
        save_manifest(
            cls.bundle,
            CoreAIManifest(
                schema_version=1,
                model_type="qwen3",
                context_length=16,
                prefill_chunk_size=4,
                vocab_size=32,
                dtype="float32",
                model_config_sha256=hashlib.sha256(
                    (cls.bundle / "config.json").read_bytes()
                ).hexdigest(),
                states=(
                    StateSpec("cache", (1,), "float32"),
                    StateSpec("next_token", (1,), "int32"),
                ),
                packages={
                    name: importlib.metadata.version(name)
                    for name in ("torch", "coreai-core", "coreai-torch")
                },
                source_model="stateful-contract-fixture",
                source_revision=None,
            ),
        )

    def new_reference_session(self):
        from coreai.runtime import AIModel, NDArray, SpecializationOptions

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

        # Replace only hardware selection/backing, not conversion or execution.
        with patch("sglang.srt.hardware_backend.coreai.session.validate_runtime"):
            session = ReferenceSession(self.bundle)
        self.addCleanup(session.close)
        return session

    def test_prefill_decode_share_real_persistent_state(self):
        session = self.new_reference_session()
        self.assertEqual(session.extend([1, 2, 3, 4, 5], 0), 15)
        self.assertEqual(session.extend([2], 5), 17)
        self.assertEqual(session.position, 6)

    def test_reset_reuses_state_without_an_old_request_leaking(self):
        session = self.new_reference_session()
        self.assertEqual(session.extend([1, 2, 3, 4], 0), 10)
        state = session._state
        arrays = dict(state)
        session.reset()
        self.assertIs(session._state, state)
        for name, array in arrays.items():
            self.assertIs(session._state[name], array)
        self.assertEqual(session.extend([3], 0), 3)

    def test_invalid_inputs_do_not_mutate_the_cache(self):
        session = self.new_reference_session()
        self.assertEqual(session.extend([2], 0), 2)
        for tokens, position in (
            ([], 1),
            ([32], 1),
            ([-1], 1),
            ([True], 1),
            ([2], 0),
            ([1] * 16, 1),
        ):
            with self.subTest(tokens=tokens, position=position):
                with self.assertRaises(ValueError):
                    session.extend(tokens, position)
                self.assertEqual(session.position, 1)
        self.assertEqual(session.extend([3], 1), 5)

    def test_execution_failure_poisons_the_session(self):
        session = self.new_reference_session()

        async def fail(*args, **kwargs):
            raise RuntimeError("execution failed")

        session._functions["decode"] = fail
        with self.assertRaisesRegex(RuntimeError, "execution failed"):
            session.extend([1], 0)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            session.reset()
        with self.assertRaisesRegex(RuntimeError, "failed"):
            session.extend([1], 0)

    def test_closed_session_cannot_execute(self):
        session = self.new_reference_session()
        session.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.extend([1], 0)

    def test_functions_produce_no_allocated_outputs(self):
        session = self.new_reference_session()
        self.assertEqual(session._functions["decode"].desc.output_names, [])
        self.assertEqual(session._functions["prefill"].desc.output_names, [])
        for _ in range(100):
            session.reset()
            self.assertEqual(session.extend([1], 0), 1)


if __name__ == "__main__":
    unittest.main()
