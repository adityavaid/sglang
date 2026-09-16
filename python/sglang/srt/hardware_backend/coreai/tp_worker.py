"""Synchronous batch-one worker backed exclusively by real Core AI sessions."""

import atexit

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_device, get_schedule


class CoreAITpModelWorker(TpModelWorker):
    def __init__(self, *args, **kwargs):
        self._coreai_session = None
        self._coreai_rid = None
        initialized = False
        try:
            super().__init__(*args, **kwargs)
            initialized = True
        finally:
            if not initialized:
                self.close()
        atexit.register(self.close)

    def _init_model_runner(self):
        from .model_runner_stub import CoreAIModelRunnerStub
        from .session import CoreAISession

        self._coreai_session = CoreAISession(get_device().coreai_artifact_path)
        self._model_runner = CoreAIModelRunnerStub(
            model_config=self.model_config,
            mem_fraction_static=get_schedule().mem_fraction_static,
            gpu_id=self.gpu_id,
            ps=self.ps,
            nccl_port=self.nccl_port,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
        )

    def get_pad_input_ids_func(self):
        return None

    def close(self):
        session = self._coreai_session
        if session is not None:
            session.close()
            self._coreai_session = None
        atexit.unregister(self.close)

    @staticmethod
    def _result(token_ids):
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=torch.tensor(token_ids, dtype=torch.int64, device="cpu"),
            can_run_cuda_graph=False,
        )

    def forward_batch_generation(
        self,
        batch,
        forward_batch=None,
        pp_proxy_tensors=None,
        is_verify=False,
        skip_attn_backend_init=None,
        *,
        capture_hidden_mode=None,
    ):
        if batch is None or forward_batch is not None:
            raise ValueError(
                "Core AI requires a scheduler batch; no inference fallback."
            )
        if is_verify or pp_proxy_tensors is not None or capture_hidden_mode is not None:
            raise ValueError(
                "Core AI does not support verification, PP or hidden capture."
            )
        mode = batch.forward_mode
        if mode == ForwardMode.IDLE:
            return self._result([])
        if mode not in (ForwardMode.EXTEND, ForwardMode.DECODE):
            raise ValueError(f"Core AI does not support forward mode {mode}.")
        if len(batch.reqs) != 1:
            raise ValueError("Core AI supports exactly one active request.")
        if batch.return_logprob or batch.return_hidden_states:
            raise ValueError("Core AI does not produce logits or hidden states.")
        session = self._coreai_session
        if session is None:
            raise RuntimeError("Core AI worker is closed.")
        req = batch.reqs[0]
        tokens = batch.input_ids.cpu().tolist()
        if mode == ForwardMode.EXTEND:
            if req.extend_range is None or batch.extend_lens != [len(tokens)]:
                raise ValueError("Core AI requires an explicit scheduled extend range.")
            start = req.extend_range.start
            if req.extend_range.end != start + len(tokens) or not tokens:
                raise ValueError("Core AI received an invalid extend range.")
            if self._coreai_rid != req.rid or start == 0:
                if start != 0:
                    raise ValueError("Core AI does not support cached prefix reuse.")
                session.reset()
                self._coreai_rid = req.rid
        else:
            if req.rid != self._coreai_rid or len(tokens) != 1:
                raise ValueError("Core AI decode must continue the active request.")
            start = int(batch.seq_lens_cpu[0]) - 1
        if start != session.position:
            raise ValueError("Core AI scheduled position does not match session state.")
        token = session.extend(tokens, start_position=start)
        return self._result([token])

    @staticmethod
    def _unsupported_update(*args, **kwargs):
        return (
            False,
            "Core AI artifacts are immutable; live weight updates are unsupported.",
        )

    update_weights_from_disk = _unsupported_update
    update_weights_from_tensor = _unsupported_update
    update_weights_from_distributed = _unsupported_update
    update_weights_from_ipc = _unsupported_update
    init_weights_update_group = _unsupported_update
    destroy_weights_update_group = _unsupported_update
    init_weights_send_group_for_remote_instance = _unsupported_update
    send_weights_to_remote_instance = _unsupported_update

    @staticmethod
    def _unsupported_lora(*args, **kwargs):
        from sglang.srt.managers.io_struct import LoRAUpdateOutput

        return LoRAUpdateOutput(
            success=False, error_message="Core AI does not support LoRA."
        )

    load_lora_adapter = _unsupported_lora
    unload_lora_adapter = _unsupported_lora
    load_lora_adapter_from_tensors = _unsupported_lora

    def get_weights_by_name(self, recv_req):
        raise ValueError("Core AI does not expose Torch weights.")

    def forward_batch_embedding(self, *args, **kwargs):
        raise ValueError("Core AI does not support embedding inference.")
