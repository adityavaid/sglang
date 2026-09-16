"""Export bindings for the original, standard-loaded Qwen3 serving forward."""

from contextlib import ExitStack, contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F

from sglang.kernels.fused_op import BaseFusedOp
from sglang.srt.hardware_backend.coreai.qwen3 import Qwen3TorchAdapter
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_attn_backend,
    has_forward_context,
)


class _ExportAttentionBackend(AttentionBackend):
    def __init__(self, owner: "Qwen3NativeForward"):
        self.owner = owner

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        if not save_kv_cache:
            raise ValueError("Core AI export requires writing each layer's KV state")
        positions = forward_batch.positions
        keys = self.owner.get_buffer(f"key_cache_{layer.layer_id}")
        values = self.owner.get_buffer(f"value_cache_{layer.layer_id}")
        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim).transpose(0, 1)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        indices = positions.view(1, 1, -1, 1).expand_as(k)
        keys.scatter_(2, indices, k)
        values.scatter_(2, indices, v)
        mask = (
            torch.arange(keys.shape[2], device=q.device)[None, :] <= positions[:, None]
        )
        output = F.scaled_dot_product_attention(
            q.unsqueeze(0),
            keys,
            values,
            attn_mask=mask[None, None, :, :],
            scale=layer.scaling,
            enable_gqa=layer.tp_q_head_num != layer.tp_k_head_num,
        )
        return (
            output.squeeze(0)
            .transpose(0, 1)
            .reshape(-1, layer.tp_q_head_num * layer.head_dim)
        )


class Qwen3NativeForward(Qwen3TorchAdapter):
    """Tensor/state wrapper invoking the loaded model's unchanged forward.

    Inherits only the reference adapter's validation, KV layout and reset
    contract, not its decoder implementation. Attention is selected through
    SGLang's existing ForwardContext; model, layer, projection, normalization
    and logits-processing forwards are not patched or replaced.

    Eager calls and torch.export must run inside ``export_context()``. The
    exported graph is self-contained and needs no SGLang runtime context.
    Preparation is offline, single-threaded and single-rank, not a live
    serving-model conversion API.
    """

    def __init__(self, model: torch.nn.Module, max_context_length: int):
        super().__init__(model, max_context_length)
        self._attention = _ExportAttentionBackend(self)
        # Attribute sources let Torch 2.11 strict export trace enum methods.
        self._decode_mode = ForwardMode.DECODE
        self._extend_mode = ForwardMode.EXTEND
        self._capture_hidden_mode = CaptureHiddenMode.NULL

    @contextmanager
    def export_context(self) -> Iterator[None]:
        from sglang.srt.hardware_backend.coreai.prepare import _native_loading_scope

        with _native_loading_scope("unused", None), ExitStack() as stack:
            for module in self.model.modules():
                if isinstance(module, BaseFusedOp) and not module.is_torch_compile:
                    module.enter_torch_compile(num_tokens=1)
                    stack.callback(module.leave_torch_compile)
            with forward_context(ForwardContext(attn_backend=self._attention)):
                yield

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, start_position: torch.Tensor
    ) -> torch.Tensor:
        positions = self._positions(input_ids, start_position)
        if not has_forward_context() or get_attn_backend() is not self._attention:
            raise RuntimeError("Qwen3NativeForward requires its export_context")
        tokens = input_ids.flatten().to(torch.int64)
        length = tokens.shape[0]
        batch = ForwardBatch(
            forward_mode=self._decode_mode if length == 1 else self._extend_mode,
            batch_size=1,
            input_ids=tokens,
            req_pool_indices=torch.zeros(1, dtype=torch.int64),
            seq_lens=start_position.to(torch.int64) + length,
            out_cache_loc=positions,
            # Trace-time placeholder: this backend never consumes CPU scheduling
            # summaries. The tensor seq_lens/positions carry the actual prefix.
            seq_lens_sum=length,
            positions=positions,
            capture_hidden_mode=self._capture_hidden_mode,
            extend_seq_lens=torch.full((1,), length, dtype=torch.int64),
            extend_seq_lens_cpu=[length],
        )
        return self.model(tokens, positions, batch).next_token_logits.to(
            self.model.lm_head.weight.dtype
        )
