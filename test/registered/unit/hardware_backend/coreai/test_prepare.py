"""Standard-loaded native weights -> Core AI execution; CPU reference, not GPU."""

import importlib.util
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.coreai_utils import checkpoint_pair
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=45, suite="base-a-test-cpu")


def tiny_qwen3():
    return checkpoint_pair()[0]


class TestPrepareLoadedQwen3(CustomTestCase):
    def test_unknown_weight_quantization_is_rejected(self):
        from sglang.srt.hardware_backend.coreai.prepare import prepare_loaded_qwen3

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen3"
            with self.assertRaisesRegex(ValueError, "weight_quantization"):
                prepare_loaded_qwen3(tiny_qwen3(), path, weight_quantization="unknown")
            self.assertFalse(path.exists())

    def test_missing_compressor_does_not_create_a_bundle(self):
        from sglang.srt.hardware_backend.coreai.prepare import prepare_loaded_qwen3

        model = tiny_qwen3()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen3"
            with (
                patch.dict(sys.modules, {"coreai_opt.base_model_compressor": None}),
                self.assertRaisesRegex(RuntimeError, "requirements-quantization"),
            ):
                prepare_loaded_qwen3(
                    model,
                    path,
                    max_context_length=16,
                    prefill_chunk_size=4,
                    weight_quantization="int4",
                )
            self.assertFalse(path.exists())

    @unittest.skipUnless(
        importlib.util.find_spec("coreai_opt"), "Core AI compression extra is absent"
    )
    def test_invalid_projection_weights_do_not_create_a_bundle(self):
        from sglang.srt.hardware_backend.coreai.prepare import prepare_loaded_qwen3

        for failure in ("divisible", "nonfinite"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                model = tiny_qwen3()
                if failure == "nonfinite":
                    with torch.no_grad():
                        model.model.layers[0].self_attn.qkv_proj.weight[0, 0] = float(
                            "nan"
                        )
                path = Path(directory) / "qwen3"
                with self.assertRaisesRegex(ValueError, failure):
                    prepare_loaded_qwen3(
                        model,
                        path,
                        max_context_length=16,
                        prefill_chunk_size=4,
                        weight_quantization="int4",
                    )
                self.assertFalse(path.exists())

    @unittest.skipUnless(
        importlib.util.find_spec("coreai_opt"), "Core AI compression extra is absent"
    )
    def test_compressed_projections_preserve_loaded_model_and_logits(self):
        from sglang.srt.hardware_backend.coreai.compression import (
            Qwen3QuantizedWeights,
        )
        from sglang.srt.hardware_backend.coreai.native_forward import Qwen3NativeForward
        from sglang.srt.hardware_backend.coreai.prepare import _GreedyStep

        model, _ = checkpoint_pair(
            intermediate_size=64, tie_word_embeddings=True, attention_bias=True
        )
        originals = dict(model.named_parameters())
        snapshots = {name: value.detach().clone() for name, value in originals.items()}
        for mode, tolerance in (("int8", 0.03), ("int4", 0.2)):
            with self.subTest(mode=mode):
                weights = Qwen3QuantizedWeights(model, mode)
                self.assertEqual(len(weights.names), 4 * model.config.num_hidden_layers)
                self.assertFalse(
                    any("lm_head" in n or "embed_tokens" in n for n in weights.names)
                )
                view = Qwen3NativeForward(model, 16)
                args = (
                    torch.tensor([[2, 9, 5]], dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                )
                with view.export_context():
                    reference = view(*args)
                    view.reset()
                    actual = torch.func.functional_call(view, weights(), args)
                relative_error = (
                    actual.float() - reference.float()
                ).norm() / reference.float().norm()
                self.assertLess(relative_error.item(), tolerance)
                view.reset()
                step = _GreedyStep(view, mode)
                with view.export_context():
                    exported = torch.export.export(step, args)
                    with (
                        patch.object(
                            view, "forward", side_effect=RuntimeError("forward failed")
                        ),
                        self.assertRaisesRegex(RuntimeError, "forward failed"),
                    ):
                        torch.func.functional_call(view, weights(), args)
                dequant_ops = [
                    node
                    for node in exported.graph.nodes
                    if "coreai.constexpr_blockwise_shift_scale" in str(node.target)
                ]
                self.assertEqual(len(dequant_ops), len(weights.names))
                self.assertTrue(
                    all(node.args[-1] == getattr(torch, mode) for node in dequant_ops)
                )
                for name, parameter in model.named_parameters():
                    self.assertIs(parameter, originals[name])
                    torch.testing.assert_close(
                        parameter, snapshots[name], rtol=0, atol=0
                    )
                self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

    @unittest.skipUnless(
        importlib.util.find_spec("coreai_torch"), "Core AI extra is not installed"
    )
    def test_real_export_prefill_decode_and_request_reuse(self):
        self._check_real_export("adapter")

    @unittest.skipUnless(
        os.environ.get("SGLANG_TEST_COREAI_NATIVE") == "1",
        "Opt-in macOS 27 native Core AI test",
    )
    def test_native_export_prefill_decode_and_request_reuse(self):
        # Exercise actual Metal-backed state: CPU reference tests cannot catch
        # the MPS compiler crash caused by a write-only next_token handle.
        self._check_real_export("adapter", native_runtime=True)

    @unittest.skipUnless(
        importlib.util.find_spec("coreai_torch"), "Core AI extra is not installed"
    )
    def test_original_forward_export_prefill_decode_and_request_reuse(self):
        self._check_real_export("native")

    @unittest.skipUnless(
        importlib.util.find_spec("coreai_opt"), "Core AI compression extra is absent"
    )
    def test_quantized_export_prefill_decode_and_request_reuse(self):
        sizes = {}
        for mode in ("int8", "int4"):
            for forward_path in ("native", "adapter"):
                with self.subTest(mode=mode, forward_path=forward_path):
                    sizes[mode, forward_path] = self._check_real_export(
                        forward_path, mode
                    )
        for forward_path in ("native", "adapter"):
            self.assertLess(sizes["int4", forward_path], sizes["int8", forward_path])

    def _check_real_export(
        self, forward_path, weight_quantization="none", *, native_runtime=False
    ):
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

        model, oracle = checkpoint_pair(
            intermediate_size=64 if weight_quantization != "none" else 48
        )
        originals = dict(model.named_parameters())
        snapshots = {name: value.detach().clone() for name, value in originals.items()}
        if weight_quantization != "none":
            from sglang.srt.hardware_backend.coreai.compression import (
                Qwen3QuantizedWeights,
            )
            from sglang.srt.hardware_backend.coreai.native_forward import (
                Qwen3NativeForward,
            )

            reference = Qwen3NativeForward(model, 16)
            dense_weights = Qwen3QuantizedWeights(model, weight_quantization)()

        def expected_token(tokens):
            if weight_quantization == "none":
                return int(oracle(torch.tensor([tokens])).logits[0, -1].argmax())
            reference.reset()
            with reference.export_context():
                logits = torch.func.functional_call(
                    reference,
                    dense_weights,
                    (
                        torch.tensor([tokens], dtype=torch.int32),
                        torch.zeros(1, dtype=torch.int32),
                    ),
                )
            return int(logits.argmax())

        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "qwen3"
            prepare_loaded_qwen3(
                model,
                bundle,
                max_context_length=16,
                prefill_chunk_size=4,
                forward_path=forward_path,
                weight_quantization=weight_quantization,
            )
            runtime_check = (
                nullcontext()
                if native_runtime
                else patch(
                    "sglang.srt.hardware_backend.coreai.session.validate_runtime"
                )
            )
            with runtime_check:
                session = (CoreAISession if native_runtime else ReferenceSession)(
                    bundle
                )
            try:
                self.assertEqual(
                    session.manifest.weight_quantization, weight_quantization
                )
                for prompt in ([1], [1, 3, 5, 7, 2], [8, 3]):
                    session.reset()
                    tokens = list(prompt)
                    with torch.no_grad():
                        expected = expected_token(tokens)
                    actual = session.extend(tokens, 0)
                    self.assertEqual(actual, expected)
                    for _ in range(3):
                        tokens.append(actual)
                        with torch.no_grad():
                            expected = expected_token(tokens)
                        actual = session.extend([actual], len(tokens) - 1)
                        self.assertEqual(actual, expected)
            finally:
                session.close()
            if weight_quantization != "none":
                baseline = Path(directory) / "uncompressed"
                prepare_loaded_qwen3(
                    model,
                    baseline,
                    max_context_length=16,
                    prefill_chunk_size=4,
                    forward_path=forward_path,
                )

                def asset_bytes(path):
                    return sum(
                        p.stat().st_size
                        for p in (path / "model.aimodel").rglob("*")
                        if p.is_file()
                    )

                self.assertLess(asset_bytes(bundle), asset_bytes(baseline))
                for name, parameter in model.named_parameters():
                    self.assertIs(parameter, originals[name])
                    torch.testing.assert_close(
                        parameter, snapshots[name], rtol=0, atol=0
                    )
                return asset_bytes(bundle)

    def test_existing_output_is_not_overwritten(self):
        from sglang.srt.hardware_backend.coreai.prepare import prepare_loaded_qwen3

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            marker = path / "keep.txt"
            marker.write_text("existing work")
            with self.assertRaises(FileExistsError):
                prepare_loaded_qwen3(tiny_qwen3(), path)
            self.assertEqual(marker.read_text(), "existing work")

    def test_invalid_forward_path_is_rejected_before_export(self):
        from sglang.srt.hardware_backend.coreai.prepare import prepare_loaded_qwen3

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qwen3"
            with self.assertRaisesRegex(ValueError, "forward_path"):
                prepare_loaded_qwen3(tiny_qwen3(), path, forward_path="invalid")
            self.assertFalse(path.exists())

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
