"""Batched decode attention wrapper for MLX backend."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

from sglang.srt.hardware_backend.mlx.kv_cache.contiguous_cache import ContiguousKVCache

_thread_local = threading.local()


@dataclass
class BatchedDecodeContext:
    """Context set before batched decode, read by attention wrappers."""

    batch_size: int
    seq_lens: list[int]  # per-request token count before the new token
    # layer_caches[layer_idx][req_idx] = ContiguousKVCache
    layer_caches: list[list[ContiguousKVCache]]
    # Custom RoPE kernel support (set by model_runner when available)
    cos_sin_cache: Optional[mx.array] = None
    rope_config: dict = field(default_factory=dict)


def set_context(ctx: Optional[BatchedDecodeContext]) -> None:
    _thread_local.batched_ctx = ctx


def get_context() -> Optional[BatchedDecodeContext]:
    return getattr(_thread_local, "batched_ctx", None)


def clear_context() -> None:
    _thread_local.batched_ctx = None


class MLXAttentionWrapper(nn.Module):
    """Wraps an mlx-lm Attention for batched decode (BS>1).

    When ``BatchedDecodeContext`` is set, performs per-request RoPE,
    cache writes, and batched SDPA.  Otherwise delegates to inner module.

    If the context includes a precomputed ``cos_sin_cache``, the wrapper
    uses the custom Metal RoPE kernel (single dispatch for Q+K) instead
    of two separate ``mx.fast.rope`` calls.
    """

    def __init__(self, inner: nn.Module, layer_idx: int):
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_layer_idx", layer_idx)

    def __call__(self, x: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask=mask, cache=cache)
        return self._batched_decode(x, ctx)

    def _batched_decode(self, x: mx.array, ctx: BatchedDecodeContext) -> mx.array:
        inner = self._inner
        layer_idx = self._layer_idx
        B = ctx.batch_size

        queries = inner.q_proj(x)
        keys = inner.k_proj(x)
        values = inner.v_proj(x)

        head_dim = queries.shape[-1] // inner.n_heads
        queries = queries.reshape(B, 1, inner.n_heads, head_dim)
        keys = keys.reshape(B, 1, inner.n_kv_heads, head_dim)
        values = values.reshape(B, 1, inner.n_kv_heads, head_dim)

        if hasattr(inner, "q_norm"):
            queries = inner.q_norm(queries)
        if hasattr(inner, "k_norm"):
            keys = inner.k_norm(keys)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offsets = mx.array(ctx.seq_lens, dtype=mx.int32)

        if ctx.cos_sin_cache is not None:
            queries, keys = self._rope_custom(
                queries, keys, offsets, ctx.cos_sin_cache, ctx.rope_config
            )
            if self._layer_idx == 0:
                logger.debug(
                    f"Custom Metal RoPE kernel used (BS={B}, "
                    f"positions={ctx.seq_lens})"
                )
        else:
            queries = inner.rope(queries, offset=offsets)
            keys = inner.rope(keys, offset=offsets)

        layer_caches = ctx.layer_caches[layer_idx]
        max_len = max(ctx.seq_lens) + 1

        # TODO: replace per-request loop with native batched/ragged
        # attention once mx.fast.scaled_dot_product_attention supports
        # variable-length sequences.
        all_k = []
        all_v = []

        for i in range(B):
            layer_caches[i].write_token(keys[i : i + 1], values[i : i + 1])

            k_all, v_all = layer_caches[i].get_kv()
            curr_len = layer_caches[i].offset

            if curr_len < max_len:
                pad = max_len - curr_len
                k_pad = mx.zeros(
                    (1, inner.n_kv_heads, pad, head_dim), dtype=k_all.dtype
                )
                v_pad = mx.zeros(
                    (1, inner.n_kv_heads, pad, head_dim), dtype=v_all.dtype
                )
                k_all = mx.concatenate([k_all, k_pad], axis=2)
                v_all = mx.concatenate([v_all, v_pad], axis=2)

            all_k.append(k_all)
            all_v.append(v_all)

        keys_b = mx.concatenate(all_k, axis=0)
        values_b = mx.concatenate(all_v, axis=0)

        attn_mask = None
        seq_lens_plus1 = [s + 1 for s in ctx.seq_lens]
        if min(seq_lens_plus1) < max_len:
            positions = mx.arange(max_len)
            valid_lens = mx.array(seq_lens_plus1, dtype=mx.int32)
            mask_bool = positions[None, :] >= valid_lens[:, None]
            attn_mask = mx.where(
                mask_bool[:, None, None, :],
<<<<<<< HEAD
                mx.array(mx.finfo(queries.dtype).min, dtype=queries.dtype),
=======
                mx.array(-1e9, dtype=queries.dtype),
>>>>>>> 4a8a2f7a0 ([MLX] Support radix cache)
                mx.array(0.0, dtype=queries.dtype),
            )

        output = mx.fast.scaled_dot_product_attention(
            queries, keys_b, values_b, scale=inner.scale, mask=attn_mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, 1, -1)
        return inner.o_proj(output)

    @staticmethod
    def _rope_custom(
        queries: mx.array,
        keys: mx.array,
        positions: mx.array,
        cos_sin_cache: mx.array,
        rope_config: dict,
    ) -> tuple[mx.array, mx.array]:
        """Apply RoPE using the custom Metal kernel (single dispatch for Q+K).

        Converts from attention layout (B, n_heads, 1, head_dim) to kernel
        layout (B, n_heads, head_dim) and back.
        """
        from sglang.srt.hardware_backend.mlx.kernels.rope import rope_neox

        # (B, n_heads, 1, head_dim) → (B, n_heads, head_dim)
        q_flat = queries[:, :, 0, :]
        k_flat = keys[:, :, 0, :]

        q_rot, k_rot = rope_neox(
            q_flat,
            k_flat,
            cos_sin_cache,
            positions,
            **rope_config,
        )

        # (B, n_heads, head_dim) → (B, n_heads, 1, head_dim)
        return q_rot[:, :, None, :], k_rot[:, :, None, :]

