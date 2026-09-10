"""Real checkpoint loading: HF is an independent fixture/oracle only."""

import os
import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.coreai_utils import checkpoint_pair

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class TestNativeLoading(unittest.TestCase):
    def test_standard_loader_preserves_packed_weights_and_native_identity(self):
        from sglang.srt.hardware_backend.coreai.qwen3 import Qwen3TorchAdapter
        from sglang.srt.models.qwen3 import Qwen3ForCausalLM as NativeQwen3

        model, oracle = checkpoint_pair()
        self.assertIs(type(model), NativeQwen3)
        self.assertFalse(model.training)
        adapter = Qwen3TorchAdapter(model, 16)
        self.assertIs(adapter.model, model)
        for name, param in model.named_parameters():
            self.assertIs(adapter.get_parameter("model." + name), param)
        for actual, expected in zip(model.model.layers, oracle.model.layers):
            torch.testing.assert_close(
                actual.self_attn.qkv_proj.weight,
                torch.cat(
                    [
                        getattr(expected.self_attn, name).weight
                        for name in ("q_proj", "k_proj", "v_proj")
                    ]
                ),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                actual.mlp.gate_up_proj.weight,
                torch.cat([expected.mlp.gate_proj.weight, expected.mlp.up_proj.weight]),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            model.lm_head.weight[:41], oracle.lm_head.weight, rtol=0, atol=0
        )

    def test_load_restores_existing_runtime_and_environment(self):
        from sglang.srt.runtime_context import get_context, snapshot_context

        before = snapshot_context()
        with patch.dict(os.environ, {"SGLANG_USE_COREAI": "1"}):
            checkpoint_pair()
            self.assertEqual(os.environ["SGLANG_USE_COREAI"], "1")
        after = snapshot_context()
        self.assertIs(before["_server_args"], after["_server_args"])
        self.assertEqual(before["__parallel__"], after["__parallel__"])
        self.assertIs(get_context().flags, before["flags"][0])
        self.assertFalse(torch.distributed.is_initialized())

    def test_rejects_nondefault_rope_before_loading_weights(self):
        with self.assertRaisesRegex(ValueError, "default RoPE"):
            checkpoint_pair(rope_parameters={"rope_type": "linear", "factor": 2.0})

    def test_failed_load_restores_environment_and_all_owned_groups(self):
        from sglang.srt.distributed import parallel_state as ps

        before = ps._SELF_PP
        with patch.dict(os.environ, {"SGLANG_USE_COREAI": "1"}):
            with self.assertRaisesRegex(ValueError, "default RoPE"):
                checkpoint_pair(rope_parameters={"rope_type": "linear", "factor": 2.0})
            self.assertEqual(os.environ["SGLANG_USE_COREAI"], "1")
        self.assertIs(ps._SELF_PP, before)
        self.assertFalse(torch.distributed.is_initialized())

    def test_loading_isolates_and_restores_existing_runtime_overrides(self):
        from sglang.srt.runtime_context import get_parallel

        with get_parallel().override(tp_size=4, attn_tp_size=4):
            model, _ = checkpoint_pair()
            self.assertEqual(model.model.layers[0].self_attn.qkv_proj.tp_size, 1)
            self.assertEqual(get_parallel().tp_size, 4)

    def test_loading_reuses_existing_single_rank_groups_without_destroying_them(self):
        from sglang.srt.distributed import parallel_state as ps
        from sglang.srt.hardware_backend.coreai.prepare import _native_loading_scope
        from sglang.srt.runtime_context import get_context

        with _native_loading_scope("unused", None):
            tp, pp = ps.get_tp_group(), ps.get_pp_group()
            args = get_context()._server_args
            checkpoint_pair()
            self.assertIs(ps.get_tp_group(), tp)
            self.assertIs(ps.get_pp_group(), pp)
            self.assertIs(get_context()._server_args, args)
            value = torch.tensor(3.0)
            torch.distributed.all_reduce(value, group=tp.cpu_group)
            self.assertEqual(value.item(), 3.0)

    def test_loading_does_not_reuse_another_models_mutable_rotary_module(self):
        from sglang.srt.hardware_backend.coreai.qwen3 import Qwen3TorchAdapter

        previous, _ = checkpoint_pair()
        previous.to(torch.bfloat16)
        model, oracle = checkpoint_pair()
        adapter = Qwen3TorchAdapter(model, 16)
        ids = torch.tensor([[2, 9, 5, 8, 3, 4, 11, 7, 6]], dtype=torch.int32)
        actual = adapter(ids, torch.tensor([0], dtype=torch.int32))
        with torch.no_grad():
            expected = oracle(ids.long()).logits[:, -1]
        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-4)
        self.assertIsNot(
            model.model.layers[0].self_attn.rotary_emb,
            previous.model.layers[0].self_attn.rotary_emb,
        )


if __name__ == "__main__":
    unittest.main()
