"""CPU-only correctness and export tests; all weights are randomly initialized."""

import unittest

import torch

from sglang.srt.hardware_backend.coreai.qwen3 import Qwen3TorchAdapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.coreai_utils import checkpoint_pair

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def tiny_model():
    return checkpoint_pair()[0]


def inputs(tokens, start=0):
    return (
        torch.tensor([tokens], dtype=torch.int32),
        torch.tensor([start], dtype=torch.int32),
    )


class TestQwen3TorchAdapter(unittest.TestCase):
    def setUp(self):
        self.model, self.oracle = checkpoint_pair()

    def test_reuses_loaded_parameters_without_changing_model(self):
        parameters = dict(self.model.named_parameters())
        config = self.model.config.to_dict()
        dispatch = {
            name: module._forward_method
            for name, module in self.model.named_modules()
            if hasattr(module, "_forward_method")
        }
        adapter = Qwen3TorchAdapter(self.model, max_context_length=16)
        adapter(*inputs([1, 2]))
        self.assertIs(adapter.model, self.model)
        self.assertEqual(self.model.config.to_dict(), config)
        self.assertFalse(self.model.training)
        self.assertEqual(len(list(adapter.parameters())), len(parameters))
        for name, parameter in parameters.items():
            self.assertIs(adapter.get_parameter("model." + name), parameter)
        for name, method in dispatch.items():
            self.assertIs(self.model.get_submodule(name)._forward_method, method)

    def test_tied_embeddings_and_attention_bias_match_independent_oracle(self):
        model, oracle = checkpoint_pair(tie_word_embeddings=True, attention_bias=True)
        adapter = Qwen3TorchAdapter(model, 16)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        adapter(*inputs([1, 3, 5]))
        actual = adapter(*inputs([7], 3))
        with torch.no_grad():
            expected = oracle(inputs([1, 3, 5, 7])[0].long()).logits[:, -1]
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-4)

    def test_prefill_and_continuations_match_hf(self):
        # Missing cache writes, wrong causal masks, and wrong positions break this.
        for dtype in (torch.float16, torch.float32):
            for chunks in ((5,), (1, 1, 1, 1, 1), (2, 2, 1), (3, 1, 1)):
                with self.subTest(dtype=dtype, chunks=chunks):
                    model = self.model.to(dtype)
                    oracle = self.oracle.to(dtype)
                    adapter = Qwen3TorchAdapter(model, max_context_length=5)
                    tokens = [2, 9, 5, 8, 3]
                    position = 0
                    with torch.no_grad():
                        for length in chunks:
                            end = position + length
                            expected = oracle(
                                input_ids=inputs(tokens[:end])[0].long(),
                                use_cache=False,
                            ).logits[:, -1, :]
                            actual = adapter(*inputs(tokens[position:end], position))
                            self.assertEqual(actual.shape, (1, 41))
                            self.assertFalse(actual.requires_grad)
                            torch.testing.assert_close(
                                actual, expected, rtol=3e-3, atol=3e-4
                            )
                            position = end

    def test_reset_zeroes_only_kv_state_and_reuses_dirty_cache(self):
        adapter = Qwen3TorchAdapter(self.model, max_context_length=16)
        rotary = self.model.model.layers[0].self_attn.rotary_emb.cos_sin_cache.clone()
        adapter(*inputs([4, 6, 8, 10, 12, 14]))
        self.assertEqual(len(adapter.state_names), 4)
        self.assertIsInstance(adapter.state_names, tuple)
        before_reset = {}
        for name in adapter.state_names:
            state = adapter.get_buffer(name)
            self.assertEqual(state.shape, (1, 2, 16, 8))
            self.assertGreater(torch.count_nonzero(state).item(), 0)
            before_reset[name] = state
        adapter.reset()
        for name in adapter.state_names:
            self.assertIs(adapter.get_buffer(name), before_reset[name])
            self.assertEqual(torch.count_nonzero(adapter.get_buffer(name)).item(), 0)
        torch.testing.assert_close(
            self.model.model.layers[0].self_attn.rotary_emb.cos_sin_cache, rotary
        )
        with torch.no_grad():
            expected = self.oracle(inputs([3, 7, 11])[0].long()).logits[:, -1, :]
        adapter(*inputs([3, 7]))
        actual = adapter(*inputs([11], 2))
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-4)

    def test_dirty_future_slots_cannot_affect_a_shorter_request(self):
        adapter = Qwen3TorchAdapter(self.model, max_context_length=16)
        adapter(*inputs([4, 6, 8, 10, 12, 14]))
        with torch.no_grad():
            expected = self.oracle(inputs([3, 7])[0].long()).logits[:, -1, :]
        actual = adapter(*inputs([3, 7]))
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-4)

    def test_rejects_invalid_context_capacity(self):
        for capacity in (0, -1, 17, True, 2.5):
            with (
                self.subTest(capacity=capacity),
                self.assertRaises((ValueError, TypeError)),
            ):
                Qwen3TorchAdapter(self.model, capacity)

    def test_rejects_unsupported_models_without_changing_them(self):
        cases = [
            torch.nn.Linear(2, 2),
            tiny_model().train(),
            tiny_model().to("meta"),
            self.oracle,
        ]
        mixed = tiny_model()
        mixed.lm_head.to(torch.float64)
        cases.append(mixed)
        partially_training = tiny_model()
        partially_training.model.layers[0].train()
        cases.append(partially_training)
        for field, value in (
            ("use_sliding_window", True),
            ("rope_parameters", {"rope_type": "linear", "factor": 2.0}),
            ("quantization_config", {"quant_method": "awq"}),
        ):
            unsupported = tiny_model()
            setattr(unsupported.config, field, value)
            cases.append(unsupported)
        for model in cases:
            with self.subTest(model=type(model), config=getattr(model, "config", None)):
                training = [module.training for module in model.modules()]
                with self.assertRaises((ValueError, TypeError)):
                    Qwen3TorchAdapter(model, 16)
                self.assertEqual(
                    [module.training for module in model.modules()], training
                )

    def test_rejects_invalid_input_metadata_before_mutating_cache(self):
        adapter = Qwen3TorchAdapter(self.model, 16)
        good_ids, good_start = inputs([1, 2])
        cases = [
            (good_ids.long(), good_start),
            (good_ids, good_start.long()),
            (good_ids.expand(2, -1), good_start),
            (good_ids[0], good_start),
            (good_ids, good_start[0]),
            (good_ids, good_start.expand(2)),
            (good_ids[:, :0], good_start),
            (good_ids.expand(1, 2).repeat(1, 9), good_start),
            (good_ids.to("meta"), good_start),
            (good_ids, good_start.to("meta")),
        ]
        for args in cases:
            with self.subTest(shapes=[x.shape for x in args]):
                with self.assertRaises((ValueError, TypeError)):
                    adapter(*args)
                for name in adapter.state_names:
                    self.assertEqual(torch.count_nonzero(adapter.get_buffer(name)), 0)

    def test_rejects_out_of_bounds_positions_before_mutating_cache(self):
        adapter = Qwen3TorchAdapter(self.model, 16)
        for start in (-1, 15, 16, 2**31 - 1):
            with self.subTest(start=start):
                with self.assertRaisesRegex(RuntimeError, "context"):
                    adapter(*inputs([1, 2], start))
                for name in adapter.state_names:
                    self.assertEqual(torch.count_nonzero(adapter.get_buffer(name)), 0)

    def test_rejects_training_after_construction(self):
        adapter = Qwen3TorchAdapter(self.model, 16)
        self.model.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            adapter(*inputs([1]))

    def test_exported_decode_and_fixed_prefill_preserve_state_mutations(self):
        # Export at zero, execute repeatedly at other positions: positions must
        # remain tensor inputs, and every KV update must survive functionalization.
        for strict in (False, True):
            for length in (1, 3):
                with self.subTest(length=length, strict=strict):
                    adapter = Qwen3TorchAdapter(self.model, 16)
                    reference = Qwen3TorchAdapter(self.model, 16)
                    example = inputs([2, 9, 5][:length])
                    program = torch.export.export(adapter, example, strict=strict)
                    functional = program.run_decompositions({})
                    self.assertEqual(
                        tuple(functional.graph_signature.buffers_to_mutate.values()),
                        adapter.state_names,
                    )
                    targets = [
                        node.target
                        for node in functional.graph.nodes
                        if node.op == "call_function"
                    ]
                    self.assertEqual(
                        targets.count(torch.ops.aten.index_copy.default), 0
                    )
                    self.assertEqual(
                        targets.count(torch.ops.aten.scatter.src),
                        len(adapter.state_names),
                    )
                    for scalar_op in (
                        torch.ops.aten.item.default,
                        torch.ops.aten._local_scalar_dense.default,
                    ):
                        self.assertEqual(targets.count(scalar_op), 0)
                    tokens = [2, 9, 5, 8, 3, 4, 11, 7, 6][: length * 3]
                    for exported in (program.module(), functional.module()):
                        for name in adapter.state_names:
                            with torch.no_grad():
                                exported.get_buffer(name).zero_()
                        reference.reset()
                        for start in range(0, len(tokens), length):
                            args = inputs(tokens[start : start + length], start)
                            expected = reference(*args)
                            with torch.no_grad():
                                actual = exported(*args)
                                hf = self.oracle(
                                    inputs(tokens[: start + length])[0].long(),
                                    use_cache=False,
                                ).logits[:, -1, :]
                            torch.testing.assert_close(
                                actual, expected, rtol=1e-5, atol=1e-6
                            )
                            torch.testing.assert_close(actual, hf, rtol=3e-3, atol=3e-4)
                            for name in adapter.state_names:
                                torch.testing.assert_close(
                                    exported.get_buffer(name),
                                    reference.get_buffer(name),
                                )
                        with (
                            torch.no_grad(),
                            self.assertRaisesRegex(RuntimeError, "context"),
                        ):
                            exported(*inputs(tokens[:length], 17 - length))
                        for name in adapter.state_names:
                            torch.testing.assert_close(
                                exported.get_buffer(name), reference.get_buffer(name)
                            )


if __name__ == "__main__":
    unittest.main()
