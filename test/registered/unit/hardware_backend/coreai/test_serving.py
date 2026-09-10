import asyncio
import hashlib
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3Config

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCoreAIServing(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, SGLANG_USE_CPU_ENGINE="0")
        env.start()
        self.addCleanup(env.stop)

    def test_coreai_serving_imports_without_mlx_installed(self):
        script = textwrap.dedent(
            """
            import builtins
            import importlib.util
            import os
            from unittest.mock import patch
            real_spec = importlib.util.find_spec
            real_import = builtins.__import__
            def find_spec(name, *args, **kwargs):
                return None if name == "mlx" or name.startswith("mlx.") else real_spec(name, *args, **kwargs)
            def checked_import(name, *args, **kwargs):
                if name == "mlx" or name.startswith("mlx."):
                    raise AssertionError("Core AI imported MLX")
                return real_import(name, *args, **kwargs)
            with patch("importlib.util.find_spec", find_spec), patch("builtins.__import__", checked_import):
                from sglang.srt.hardware_backend.coreai import runtime
                with patch.object(runtime, "validate_runtime"), patch.dict(os.environ, {"SGLANG_USE_COREAI": "1", "SGLANG_USE_MLX": "0"}):
                    from sglang.srt.managers.scheduler import Scheduler
                    from sglang.srt.managers.tokenizer_manager import TokenizerManager
                    from sglang.srt.hardware_backend.coreai.tp_worker import CoreAITpModelWorker
                    from sglang.srt.hardware_backend.coreai.model_runner_stub import CoreAIModelRunnerStub
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=90
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def validate(self, request=None, **kwargs):
        from sglang.srt.hardware_backend.coreai.serving import validate_request

        params = SamplingParams(**{"temperature": 0, **kwargs})
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                WordLevel({"[UNK]": 0, "done": 1}, unk_token="[UNK]")
            ),
            unk_token="[UNK]",
        )
        params.normalize(tokenizer)
        params.verify(100)
        validate_request(request or GenerateReqInput(input_ids=[1]), params)

    def test_greedy_stopping_and_streaming_are_supported(self):
        self.validate(
            GenerateReqInput(input_ids=[1], stream=True),
            stop=["done"],
            stop_token_ids={2},
        )

    def test_sampling_features_are_not_silently_ignored(self):
        for values in (
            {"temperature": 1},
            {"frequency_penalty": 0.5},
            {"presence_penalty": 0.5},
            {"repetition_penalty": 1.2},
            {"min_new_tokens": 1},
            {"json_schema": '{"type":"object"}'},
            {"regex": "a"},
            {"ebnf": 'root ::= "a"'},
            {"structural_tag": "{}"},
            {"logit_bias": {"1": 1.0}},
            {"custom_params": {"thinking_budget": 2}},
            {"n": 2},
        ):
            with (
                self.subTest(values=values),
                self.assertRaisesRegex(ValueError, "Core AI"),
            ):
                self.validate(**values)

    def test_auxiliary_and_nontext_requests_are_rejected(self):
        for field, value in (
            ("return_logprob", True),
            ("return_hidden_states", True),
            ("return_sampling_mask", True),
            ("return_routed_experts", True),
            ("return_indexer_topk", True),
            ("custom_logit_processor", "processor"),
            ("max_thinking_tokens", 0),
            ("input_embeds", [[1.0]]),
            ("image_data", "image"),
            ("audio_data", "audio"),
            ("video_data", "video"),
            ("lora_path", "adapter"),
            ("session_params", {"id": "session"}),
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "Core AI"),
            ):
                self.validate(GenerateReqInput(input_ids=[1], **{field: value}))
        with self.assertRaisesRegex(ValueError, "Core AI"):
            self.validate(EmbeddingReqInput(input_ids=[1]))

    def resolve(self, **kwargs):
        from sglang.srt.hardware_backend.coreai.serving import resolve_server_args

        args = ServerArgs(model_path="dummy", **kwargs)
        args.model_path = str(Path("bundle").resolve())
        args.coreai_artifact_path = args.model_path
        manifest = SimpleNamespace(
            context_length=128,
            prefill_chunk_size=16,
            dtype="float16",
            vocab_size=100,
            model_type="qwen3",
        )
        with patch(
            "sglang.srt.hardware_backend.coreai.serving.load_manifest",
            return_value=manifest,
        ):
            resolve_server_args(args)
        return args, resolving_view(args)

    def test_safe_profile_is_declared_without_mutating_raw_arguments(self):
        args, cfg = self.resolve()
        self.assertIsNone(args.max_running_requests)
        self.assertEqual(cfg.max_running_requests, 1)
        self.assertEqual(cfg.max_total_tokens, 128)
        self.assertEqual(cfg.context_length, 128)
        self.assertEqual(cfg.chunked_prefill_size, 16)
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.dtype, "float16")
        self.assertTrue(cfg.disable_radix_cache)
        self.assertTrue(cfg.disable_overlap_schedule)
        self.assertTrue(cfg.disable_cuda_graph)

    def test_incompatible_profile_is_rejected(self):
        for options in (
            {"tp_size": 2},
            {"pp_size": 2},
            {"dp_size": 2},
            {"max_running_requests": 2},
            {"max_total_tokens": 129},
            {"context_length": 129},
            {"device": "mps"},
            {"dtype": "bfloat16"},
            {"speculative_algorithm": "EAGLE"},
            {"enable_lora": True},
            {"is_embedding": True},
            {"quantization": "awq"},
            {"cuda_graph_backend_decode": "full"},
            {"json_model_override_args": '{"vocab_size": 101}'},
            {"mem_fraction_static": 0.5},
        ):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "Core AI"),
            ):
                self.resolve(**options)

    def test_bundle_and_model_paths_must_match(self):
        from sglang.srt.hardware_backend.coreai.serving import resolve_server_args

        args = ServerArgs(model_path="dummy")
        args.coreai_artifact_path = "different"
        with self.assertRaisesRegex(ValueError, "model-path"):
            resolve_server_args(args)

    def test_tokenizer_hook_preserves_streaming_but_rejects_sampling(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        manager = TokenizerManager.__new__(TokenizerManager)
        manager.preferred_sampling_params = None
        manager.sampling_params_class = SamplingParams
        manager.tokenizer = None
        manager.model_config = SimpleNamespace(vocab_size=100)
        request = GenerateReqInput(
            input_ids=[1], sampling_params={"temperature": 0}, stream=True
        )
        request.normalize_batch_and_arguments()
        from sglang.srt.observability.req_time_stats import APIServerReqTimeStats

        manager.rid_to_state = {
            request.rid: SimpleNamespace(time_stats=APIServerReqTimeStats())
        }
        with (
            patch(
                "sglang.srt.hardware_backend.coreai.runtime.use_coreai",
                return_value=True,
            ),
            patch(
                "sglang.srt.managers.tokenizer_manager.get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_transfer_backend="mooncake"
                ),
            ),
        ):
            tokenized = manager._create_tokenized_object(request, None, [1])
            self.assertEqual(list(tokenized.input_ids), [1])
            self.assertTrue(tokenized.stream)
            self.assertEqual(tokenized.sampling_params.top_k, 1)
            request.sampling_params = {"temperature": 1}
            with self.assertRaisesRegex(ValueError, "greedy"):
                manager._create_tokenized_object(request, None, [1])

    def test_nontext_inputs_are_rejected_before_tokenization_or_lora_loading(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        manager = TokenizerManager.__new__(TokenizerManager)
        manager.auto_create_handle_loop = lambda: None
        for request in (
            GenerateReqInput(
                input_ids=[1], image_data="image", sampling_params={"temperature": 0}
            ),
            GenerateReqInput(
                input_ids=[1], lora_path="adapter", sampling_params={"temperature": 0}
            ),
            EmbeddingReqInput(input_ids=[1]),
        ):
            with (
                patch(
                    "sglang.srt.hardware_backend.coreai.runtime.use_coreai",
                    return_value=True,
                ),
                self.assertRaisesRegex(ValueError, "Core AI"),
            ):
                asyncio.run(anext(manager.generate_request(request)))

    def test_normalized_request_lists_can_queue_without_requesting_logprobs(self):
        from sglang.srt.hardware_backend.coreai.serving import validate_request_inputs

        request = GenerateReqInput(
            input_ids=[[1], [2]], sampling_params={"temperature": 0}
        )
        request.normalize_batch_and_arguments()
        validate_request_inputs(request)
        for index in range(2):
            self.validate(request[index])

    def test_opaque_weight_and_memory_controls_fail_before_scheduler_dispatch(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        manager = TokenizerManager.__new__(TokenizerManager)
        with patch.dict(os.environ, SGLANG_USE_COREAI="1", SGLANG_USE_MLX="0"):
            for method in (
                "get_weights_by_name",
                "release_memory_occupation",
                "resume_memory_occupation",
            ):
                with (
                    self.subTest(method=method),
                    self.assertRaisesRegex(ValueError, "Core AI"),
                ):
                    asyncio.run(getattr(manager, method)(None))

    def test_weight_checks_return_explicit_failure_before_scheduler_dispatch(self):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        manager = TokenizerManager.__new__(TokenizerManager)
        with patch.dict(os.environ, SGLANG_USE_COREAI="1", SGLANG_USE_MLX="0"):
            success, message, ranks, checksum = asyncio.run(manager.check_weights(None))
        self.assertFalse(success)
        self.assertIn("Core AI", message)
        self.assertIsNone(ranks)
        self.assertIsNone(checksum)


class TestCoreAIResolutionPipeline(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, SGLANG_USE_CPU_ENGINE="0")
        env.start()
        self.addCleanup(env.stop)
        from sglang.srt.hardware_backend.coreai.artifact import (
            CoreAIManifest,
            StateSpec,
            save_manifest,
        )

        self.bundle = Path.cwd() / (".coreai-serving-test-" + uuid.uuid4().hex)
        self.bundle.mkdir()
        self.addCleanup(shutil.rmtree, self.bundle)
        (self.bundle / "model.aimodel").mkdir()
        config = Qwen3Config(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            architectures=["Qwen3ForCausalLM"],
            dtype="float16",
        )
        config_bytes = config.to_json_string().encode()
        (self.bundle / "config.json").write_bytes(config_bytes)
        save_manifest(
            self.bundle,
            CoreAIManifest(
                schema_version=1,
                model_type="qwen3",
                context_length=128,
                prefill_chunk_size=16,
                vocab_size=100,
                dtype="float16",
                model_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                states=(
                    StateSpec("kv", (1, 2, 128, 8), "float16"),
                    StateSpec("next_token", (1,), "int32"),
                ),
                packages={
                    "torch": "2.11.0",
                    "coreai-core": "1.0.0b2",
                    "coreai-torch": "0.4.2",
                },
                source_model=None,
                source_revision=None,
            ),
        )

    def test_full_resolution_keeps_safe_profile_after_ordinary_defaults(self):
        from sglang.srt.platforms.cpu import CpuSRTPlatform

        args = ServerArgs(
            model_path=str(self.bundle), coreai_artifact_path=str(self.bundle)
        )
        with (
            patch.dict(os.environ, {"SGLANG_USE_COREAI": "1", "SGLANG_USE_MLX": "0"}),
            patch("sglang.srt.hardware_backend.coreai.runtime.validate_runtime"),
            patch("sglang.srt.platforms._current_platform", CpuSRTPlatform()),
        ):
            args.resolve_once()
        cfg = resolving_view(args)
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.context_length, 128)
        self.assertEqual(cfg.chunked_prefill_size, 16)
        self.assertEqual(cfg.max_total_tokens, 128)
        self.assertEqual(cfg.max_running_requests, 1)
        self.assertTrue(cfg.disable_overlap_schedule)
        self.assertTrue(cfg.disable_radix_cache)
        self.assertEqual(cfg.cuda_graph_config.decode.backend, "disabled")
        self.assertEqual(cfg.cuda_graph_config.prefill.backend, "disabled")

    def test_platform_is_cpu_bookkeeping_and_explicit_conflicts_fail(self):
        from sglang.srt.platforms import _resolve_platform
        from sglang.srt.platforms.cpu import CpuSRTPlatform

        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_USE_COREAI": "1",
                    "SGLANG_USE_MLX": "0",
                    "SGLANG_PLATFORM": "",
                },
            ),
            patch("sglang.srt.hardware_backend.coreai.runtime.validate_runtime"),
        ):
            self.assertIsInstance(_resolve_platform(), CpuSRTPlatform)
            with patch.dict(os.environ, {"SGLANG_PLATFORM": "coreai"}):
                self.assertIsInstance(_resolve_platform(), CpuSRTPlatform)
            with patch.dict(os.environ, {"SGLANG_PLATFORM": "unrelated"}):
                with self.assertRaisesRegex(ValueError, "SGLANG_PLATFORM"):
                    _resolve_platform()

    def test_worker_startup_uses_no_linux_cpu_kernels_or_torch_model(self):
        from test_tp_worker import RecordingSession

        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        from sglang.srt.distributed.parallel_state_wrapper import ParallelState
        from sglang.srt.hardware_backend.coreai.tp_worker import CoreAITpModelWorker
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.platforms.cpu import CpuSRTPlatform
        from sglang.srt.runtime_context import (
            publish,
            restore_context,
            snapshot_context,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        context = snapshot_context()
        self.addCleanup(restore_context, context)
        args = ServerArgs(
            model_path=str(self.bundle),
            coreai_artifact_path=str(self.bundle),
            skip_tokenizer_init=True,
        )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        session = RecordingSession()
        with (
            patch.dict(os.environ, {"SGLANG_USE_COREAI": "1", "SGLANG_USE_MLX": "0"}),
            patch("sglang.srt.hardware_backend.coreai.runtime.validate_runtime"),
            patch(
                "sglang.srt.hardware_backend.coreai.session.CoreAISession",
                return_value=session,
            ),
            patch("sglang.srt.platforms._current_platform", CpuSRTPlatform()),
            patch(
                "sglang.srt.distributed.bootstrap._init_cpu_threads_env",
                side_effect=AssertionError("Linux CPU kernels must not initialize"),
            ),
        ):
            args.resolve_once()
            publish(args, role="scheduler")
            self.addCleanup(destroy_distributed_environment)
            self.addCleanup(destroy_model_parallel)
            scheduler = Scheduler.__new__(Scheduler)
            scheduler.server_args = args
            scheduler.ps = ParallelState.trivial()
            scheduler.nccl_port = port
            scheduler.model_config = ModelConfig.from_server_args(args)
            scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
            scheduler.enable_dp_attention = False
            scheduler.init_model_worker()
            worker = scheduler.tp_worker
            self.addCleanup(worker.close)
            self.assertIsInstance(worker, CoreAITpModelWorker)
            self.assertEqual(
                worker.model_runner.token_to_kv_pool.get_kv_size_bytes(), (0, 0)
            )
            self.assertEqual(worker.model_runner.req_to_token_pool.size, 1)
            self.assertEqual(worker.get_worker_info()[0], 128)
            with self.assertRaisesRegex(RuntimeError, "Core AI"):
                worker.model_runner.model.forward()
            scheduler.hisparse_coordinator = None
            scheduler.decode_offload_manager = None
            scheduler.tree_cache = SimpleNamespace(release_host_resources=lambda: None)
            scheduler.release_host_resources()
            self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
