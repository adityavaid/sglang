"""Tensor-only execution boundary over a standard-loaded native SGLang Qwen3."""

import torch
import torch.nn.functional as F
from torch import nn


def validate_qwen3_config(config):
    if config.model_type != "qwen3" or config.architectures != ["Qwen3ForCausalLM"]:
        raise ValueError("Core AI requires a dense native Qwen3ForCausalLM checkpoint.")
    if getattr(config, "quantization_config", None):
        raise ValueError("Core AI preparation does not support quantization.")
    if (
        getattr(config, "use_sliding_window", False)
        or getattr(config, "sliding_window", None) is not None
        or any(kind != "full_attention" for kind in getattr(config, "layer_types", []))
    ):
        raise ValueError("only full-attention Qwen3 is supported")
    rope = (
        getattr(config, "rope_parameters", None)
        or getattr(config, "rope_scaling", None)
        or {}
    )
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError("only default RoPE is supported")


class _CacheView:
    def __init__(self, layers, positions):
        self.layers = layers
        self.positions = positions

    def update(self, key_states, value_states, layer_idx):
        keys, values = self.layers[layer_idx]
        indices = self.positions.view(1, 1, -1, 1).expand_as(key_states)
        keys.scatter_(2, indices, key_states)
        values.scatter_(2, indices, value_states)
        return keys, values


class Qwen3TorchAdapter(nn.Module):
    """Reuse native SGLang modules/parameters with fixed-capacity mutable KV.

    Supports CPU, eval-mode dense TP=PP=1 Qwen3 with default RoPE. Native
    normalization, rotary and activation implementations are called directly;
    packed linears use their tensor operation, bypassing collective dispatch.
    Static SDPA replaces RadixAttention's paged scheduler backend, and the
    last-token projection replaces scheduler-dependent LogitsProcessor.
    No global kernel dispatch is modified. No weights are copied,
    frozen, moved, or put into eval mode by this adapter.

    Inputs are CPU int32 ``input_ids[1, T]`` and ``start_position[1]``; output
    is last-token logits ``[1, vocab_size]`` in the model's dtype, without
    gradients. ``0 <= start_position <= max_context_length - T`` and ``T > 0``.
    Callers must populate the contiguous prefix before continuing at a nonzero
    position. One adapter serves one request at a time; reset between requests.

    ``state_names`` contains ``key_cache_i, value_cache_i`` for each layer in
    order, each shaped ``[1, num_key_value_heads, max_context_length, head_dim]``.
    Export specializes T, not start_position. Functionalize an ExportedProgram
    with ``run_decompositions({})`` to expose BUFFER_MUTATION outputs explicitly.
    """

    def __init__(self, model: nn.Module, max_context_length: int):
        super().__init__()
        from sglang.srt.models.qwen3 import Qwen3ForCausalLM

        if type(model) is not Qwen3ForCausalLM:
            raise TypeError("model must be a dense native SGLang Qwen3ForCausalLM")
        config = model.config
        if type(max_context_length) is not int:
            raise TypeError("max_context_length must be an integer")
        if not 0 < max_context_length <= config.max_position_embeddings:
            raise ValueError("context capacity must be within max_position_embeddings")
        if any(module.training for module in model.modules()):
            raise ValueError("the loaded model must already be in eval mode")
        validate_qwen3_config(config)
        if model.quant_config is not None:
            raise ValueError("quantized native models are not supported")
        if model.pp_group.world_size != 1 or any(
            getattr(module, "tp_size", 1) != 1 for module in model.modules()
        ):
            raise ValueError("Core AI export requires TP=PP=1")
        weight = model.model.embed_tokens.weight
        if weight.dtype not in (torch.float32, torch.float16):
            raise ValueError("model weights must use a supported floating dtype")
        if any(
            p.device.type != "cpu" or p.dtype != weight.dtype
            for p in model.parameters()
        ) or any(b.device.type != "cpu" for b in model.buffers()):
            raise ValueError("model must be on CPU with one weight dtype")
        self.model = model
        self.training = False
        self.max_context_length = max_context_length
        self.num_layers = config.num_hidden_layers
        shape = (
            1,
            config.num_key_value_heads,
            max_context_length,
            config.head_dim,
        )
        names = []
        for layer in range(self.num_layers):
            for kind in ("key", "value"):
                name = f"{kind}_cache_{layer}"
                self.register_buffer(name, weight.new_zeros(shape))
                names.append(name)
        self.state_names: tuple[str, ...] = tuple(names)

    @torch.no_grad()
    def reset(self) -> None:
        """Zero KV storage in place, leaving loaded weights and RoPE untouched."""
        for name in self.state_names:
            self.get_buffer(name).zero_()

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, start_position: torch.Tensor
    ) -> torch.Tensor:
        if any(module.training for module in self.modules()):
            raise ValueError("adapter and loaded model must remain in eval mode")
        if input_ids.dtype != torch.int32 or start_position.dtype != torch.int32:
            raise TypeError("input_ids and start_position must be int32 tensors")
        if input_ids.device.type != "cpu" or start_position.device.type != "cpu":
            raise ValueError("input_ids and start_position must be on CPU")
        if (
            input_ids.ndim != 2
            or input_ids.shape[0] != 1
            or not 0 < input_ids.shape[1] <= self.max_context_length
            or start_position.shape != (1,)
        ):
            raise ValueError(
                "expected input_ids[1, T] and start_position[1] within context"
            )
        torch._assert_async(
            (
                (start_position >= 0)
                & (start_position <= self.max_context_length - input_ids.shape[1])
            ).all(),
            "start_position is outside context bounds",
        )
        positions = start_position.to(torch.int64) + torch.arange(
            input_ids.shape[1], device=input_ids.device
        )
        layers = [
            (getattr(self, f"key_cache_{i}"), getattr(self, f"value_cache_{i}"))
            for i in range(self.num_layers)
        ]
        cache = _CacheView(layers, positions)
        future = torch.arange(self.max_context_length, device=input_ids.device)
        future = future.unsqueeze(0) > positions.unsqueeze(1)
        mask = torch.zeros(
            (input_ids.shape[1], self.max_context_length),
            dtype=layers[0][0].dtype,
            device=input_ids.device,
        ).masked_fill(future, torch.finfo(layers[0][0].dtype).min)
        hidden = F.embedding(input_ids.flatten(), self.model.model.embed_tokens.weight)
        residual = None
        for i, layer in enumerate(self.model.model.layers):
            if residual is None:
                residual = hidden
                hidden = layer.input_layernorm.forward_native(hidden)
            else:
                hidden, residual = layer.input_layernorm.forward_native(
                    hidden, residual
                )
            attn = layer.self_attn
            qkv = _linear(attn.qkv_proj, hidden)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            q = attn.q_norm.forward_native(q.reshape(-1, attn.num_heads, attn.head_dim))
            k = attn.k_norm.forward_native(
                k.reshape(-1, attn.num_kv_heads, attn.head_dim)
            )
            q, k = attn.rotary_emb.forward_native(positions, q, k)
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = (
                v.reshape(-1, attn.num_kv_heads, attn.head_dim)
                .transpose(0, 1)
                .unsqueeze(0)
            )
            k, v = cache.update(k, v, i)
            repeats = attn.num_heads // attn.num_kv_heads
            hidden = F.scaled_dot_product_attention(
                q,
                k.repeat_interleave(repeats, dim=1),
                v.repeat_interleave(repeats, dim=1),
                attn_mask=mask.unsqueeze(0).unsqueeze(0),
                scale=attn.attn.scaling,
            )
            hidden = hidden.squeeze(0).transpose(0, 1).reshape(-1, attn.q_size)
            hidden = _linear(attn.o_proj, hidden)
            hidden, residual = layer.post_attention_layernorm.forward_native(
                hidden, residual
            )
            hidden = layer.mlp.act_fn.forward_native(
                _linear(layer.mlp.gate_up_proj, hidden)
            )
            hidden = _linear(layer.mlp.down_proj, hidden)
        hidden, _ = self.model.model.norm.forward_native(hidden[-1:], residual[-1:])
        return F.linear(hidden, self.model.lm_head.weight)[
            :, : self.model.config.vocab_size
        ]


def _linear(layer, hidden):
    # The TP=1 dense operation, without runtime collectives or kernel selection.
    return F.linear(hidden, layer.weight, layer.bias)
