import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class RecordingSession:
    """Stand-in only for the unavailable macOS 27 compiled runtime."""

    def __init__(self):
        self.position = 0
        self.calls = []
        self.resets = 0
        self.closed = False
        self.manifest = SimpleNamespace(prefill_chunk_size=2, context_length=16)

    def extend(self, input_ids, start_position):
        assert start_position == self.position
        self.calls.append((list(input_ids), start_position))
        self.position += len(input_ids)
        return 9

    def reset(self):
        self.resets += 1
        self.position = 0

    def close(self):
        self.closed = True


class TestCoreAITpWorker(unittest.TestCase):
    def setUp(self):
        from sglang.srt.hardware_backend.coreai.tp_worker import CoreAITpModelWorker

        self.worker = CoreAITpModelWorker.__new__(CoreAITpModelWorker)
        self.worker._coreai_session = RecordingSession()
        self.worker._coreai_rid = None

    def batch(self, ids, start=0, rid="r", mode=ForwardMode.EXTEND):
        req = SimpleNamespace(
            rid=rid,
            extend_range=SimpleNamespace(start=start, end=start + len(ids)),
        )
        return SimpleNamespace(
            reqs=[req],
            input_ids=torch.tensor(ids, dtype=torch.int64),
            forward_mode=mode,
            extend_lens=[len(ids)],
            seq_lens_cpu=torch.tensor([start + len(ids)]),
            return_logprob=False,
            return_hidden_states=False,
        )

    def test_prefill_continuation_and_decode_use_scheduled_tokens(self):
        first = self.worker.forward_batch_generation(self.batch([1, 2]))
        self.worker.forward_batch_generation(self.batch([3], start=2))
        self.worker.forward_batch_generation(
            self.batch([9], start=3, mode=ForwardMode.DECODE)
        )
        self.assertEqual(
            self.worker._coreai_session.calls, [([1, 2], 0), ([3], 2), ([9], 3)]
        )
        self.assertEqual(first.next_token_ids.tolist(), [9])
        self.assertEqual(first.next_token_ids.device.type, "cpu")
        self.assertIsNone(first.logits_output.next_token_logits)
        self.assertFalse(first.can_run_cuda_graph)

    def test_fresh_request_and_retraction_reset_state(self):
        self.worker.forward_batch_generation(self.batch([1, 2]))
        self.worker.forward_batch_generation(self.batch([4], rid="new"))
        self.worker.forward_batch_generation(self.batch([4], rid="new"))
        self.assertEqual(self.worker._coreai_session.position, 1)
        self.assertEqual(self.worker._coreai_session.resets, 3)

    def test_prefix_reuse_and_discontinuous_positions_fail(self):
        with self.assertRaisesRegex(ValueError, "prefix"):
            self.worker.forward_batch_generation(self.batch([1], start=2))
        self.worker.forward_batch_generation(self.batch([1]))
        with self.assertRaisesRegex(ValueError, "position"):
            self.worker.forward_batch_generation(self.batch([2], start=3))

    def test_unsupported_modes_never_fall_back(self):
        for mode in (
            ForwardMode.TARGET_VERIFY,
            ForwardMode.MIXED,
            ForwardMode.SPLIT_PREFILL,
        ):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.worker.forward_batch_generation(self.batch([1], mode=mode))
        with self.assertRaises(ValueError):
            self.worker.forward_batch_generation(None)
        self.assertEqual(self.worker._coreai_session.calls, [])

    def test_idle_has_no_runtime_launch(self):
        result = self.worker.forward_batch_generation(
            self.batch([], mode=ForwardMode.IDLE)
        )
        self.assertEqual(result.next_token_ids.numel(), 0)
        self.assertEqual(self.worker._coreai_session.calls, [])

    def test_batch_size_and_decode_identity_are_checked(self):
        batch = self.batch([1])
        batch.reqs *= 2
        with self.assertRaises(ValueError):
            self.worker.forward_batch_generation(batch)
        with self.assertRaises(ValueError):
            self.worker.forward_batch_generation(
                self.batch([9], mode=ForwardMode.DECODE)
            )

    def test_session_errors_propagate_and_close_is_owned(self):
        def fail(*args, **kwargs):
            raise RuntimeError("compiled function failed")

        self.worker._coreai_session.extend = fail
        with self.assertRaisesRegex(RuntimeError, "compiled function failed"):
            self.worker.forward_batch_generation(self.batch([1]))
        session = self.worker._coreai_session
        self.worker.close()
        self.worker.close()
        self.assertTrue(session.closed)

    def test_failed_close_does_not_discard_owned_session(self):
        session = self.worker._coreai_session

        def fail():
            raise RuntimeError("session is busy")

        session.close = fail
        with self.assertRaisesRegex(RuntimeError, "busy"):
            self.worker.close()
        self.assertIs(self.worker._coreai_session, session)

    def test_failed_worker_initialization_releases_its_session(self):
        from unittest.mock import patch

        from sglang.srt.hardware_backend.coreai.tp_worker import CoreAITpModelWorker
        from sglang.srt.managers.tp_worker import TpModelWorker

        session = RecordingSession()

        def fail(worker, *args, **kwargs):
            worker._coreai_session = session
            raise RuntimeError("bookkeeping initialization failed")

        with patch.object(TpModelWorker, "__init__", fail):
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                CoreAITpModelWorker()
        self.assertTrue(session.closed)

    def test_live_weight_updates_and_lora_are_truthfully_unsupported(self):
        for name in (
            "update_weights_from_disk",
            "update_weights_from_tensor",
            "update_weights_from_distributed",
            "update_weights_from_ipc",
            "init_weights_update_group",
            "destroy_weights_update_group",
        ):
            success, message = getattr(self.worker, name)(None)
            self.assertFalse(success)
            self.assertIn("Core AI", message)
        for name in (
            "load_lora_adapter",
            "unload_lora_adapter",
            "load_lora_adapter_from_tensors",
        ):
            from sglang.srt.managers.io_struct import LoadLoRAAdapterReqOutput

            response = getattr(self.worker, name)(None)
            self.assertIsInstance(response, LoadLoRAAdapterReqOutput)
            self.assertFalse(response.success)

    def test_stub_allocates_only_cpu_bookkeeping(self):
        from unittest.mock import patch

        from sglang.srt.hardware_backend.coreai.model_runner_stub import (
            CoreAIModelRunnerStub,
        )

        runner = CoreAIModelRunnerStub.__new__(CoreAIModelRunnerStub)
        runner.model_config = SimpleNamespace(num_hidden_layers=2, context_len=32)
        with (
            patch(
                "sglang.srt.hardware_backend.coreai.model_runner_stub.get_schedule",
                return_value=SimpleNamespace(max_total_tokens=32),
            ),
            patch.object(runner, "init_ngram_embedding_manager"),
        ):
            runner.initialize()
        self.assertEqual(runner.req_to_token_pool.req_to_token.device.type, "cpu")
        self.assertEqual(runner.token_to_kv_pool.get_kv_size_bytes(), (0, 0))
        self.assertEqual(runner.effective_max_total_num_tokens, 32)
        self.assertEqual(runner.req_to_token_pool.size, 1)
        with self.assertRaisesRegex(RuntimeError, "Core AI"):
            runner.token_to_kv_pool.get_key_buffer(0)
        with self.assertRaisesRegex(RuntimeError, "Core AI"):
            runner.model.forward()


if __name__ == "__main__":
    unittest.main()
