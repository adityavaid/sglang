"""Custom Metal RoPE kernels for Apple Silicon.

Two variants:
  - rope_neox: Standalone RoPE for Q and K in a single dispatch.
  - rope_neox_with_kv: RoPE for Q and K + produces pool-formatted
    rotated K and original V slices ready for scatter into MlxKVPool.

Both use NeoX-style pairing (first half paired with second half)
and a precomputed cos/sin cache indexed by position.
"""

import mlx.core as mx

# ---------------------------------------------------------------------------
# Standalone RoPE kernel (Q + K in one dispatch)
# ---------------------------------------------------------------------------

_ROPE_NEOX_SOURCE = """
    uint elem = thread_position_in_grid.x;
    uint half_dim = ROPE_DIM / 2;
    uint total_heads = NUM_QO_HEADS + NUM_KV_HEADS;
    uint work_per_token = total_heads * half_dim;

    uint token_id = elem / work_per_token;
    uint rem = elem % work_per_token;
    uint head_id = rem / half_dim;
    uint dim_idx = rem % half_dim;

    bool is_q = (head_id < NUM_QO_HEADS);
    uint actual_head = is_q ? head_id : (head_id - NUM_QO_HEADS);
    uint heads_in_tensor = is_q ? NUM_QO_HEADS : NUM_KV_HEADS;

    int32_t pos = positions[token_id];

    float cos_val = cos_sin_cache[pos * ROPE_DIM + dim_idx];
    float sin_val = cos_sin_cache[pos * ROPE_DIM + half_dim + dim_idx];

    uint base = token_id * heads_in_tensor * HEAD_DIM
              + actual_head * HEAD_DIM;
    uint idx1 = base + dim_idx;
    uint idx2 = base + half_dim + dim_idx;

    if (is_q) {
        float x1 = static_cast<float>(q_in[idx1]);
        float x2 = static_cast<float>(q_in[idx2]);
        q_out[idx1] = static_cast<T>(x1 * cos_val - x2 * sin_val);
        q_out[idx2] = static_cast<T>(x1 * sin_val + x2 * cos_val);
    } else {
        float x1 = static_cast<float>(k_in[idx1]);
        float x2 = static_cast<float>(k_in[idx2]);
        k_out[idx1] = static_cast<T>(x1 * cos_val - x2 * sin_val);
        k_out[idx2] = static_cast<T>(x1 * sin_val + x2 * cos_val);
    }
"""

# ---------------------------------------------------------------------------
# Fused RoPE + KV preparation kernel
#
# Phase 1: RoPE on Q + K heads
# Phase 2: Copy rotated K into k_store (pool layout) and original V into v_store
#
# k_store / v_store have shape (num_tokens, n_kv_heads, head_dim) and can
# be directly scattered into MlxKVPool via pool.set_kv(layer, slots, k_store, v_store).
# ---------------------------------------------------------------------------

_ROPE_NEOX_WITH_KV_SOURCE = """
    uint elem = thread_position_in_grid.x;
    uint half_dim = ROPE_DIM / 2;
    uint total_heads = NUM_QO_HEADS + NUM_KV_HEADS;
    uint rope_work_per_token = total_heads * half_dim;
    uint rope_work = NUM_TOKENS * rope_work_per_token;

    if (elem < rope_work) {
        uint token_id = elem / rope_work_per_token;
        uint rem = elem % rope_work_per_token;
        uint head_id = rem / half_dim;
        uint dim_idx = rem % half_dim;

        bool is_q = (head_id < NUM_QO_HEADS);
        uint actual_head = is_q ? head_id : (head_id - NUM_QO_HEADS);
        uint heads_in_tensor = is_q ? NUM_QO_HEADS : NUM_KV_HEADS;

        int32_t pos = positions[token_id];

        float cos_val = cos_sin_cache[pos * ROPE_DIM + dim_idx];
        float sin_val = cos_sin_cache[pos * ROPE_DIM + half_dim + dim_idx];

        uint base = token_id * heads_in_tensor * HEAD_DIM
                  + actual_head * HEAD_DIM;
        uint idx1 = base + dim_idx;
        uint idx2 = base + half_dim + dim_idx;

        if (is_q) {
            float x1 = static_cast<float>(q_in[idx1]);
            float x2 = static_cast<float>(q_in[idx2]);
            q_out[idx1] = static_cast<T>(x1 * cos_val - x2 * sin_val);
            q_out[idx2] = static_cast<T>(x1 * sin_val + x2 * cos_val);
        } else {
            float x1 = static_cast<float>(k_in[idx1]);
            float x2 = static_cast<float>(k_in[idx2]);
            T rotated1 = static_cast<T>(x1 * cos_val - x2 * sin_val);
            T rotated2 = static_cast<T>(x1 * sin_val + x2 * cos_val);
            k_out[idx1] = rotated1;
            k_out[idx2] = rotated2;

            // Also write to k_store (same layout as k_in: num_tokens, nk, hd)
            k_store[idx1] = rotated1;
            k_store[idx2] = rotated2;
        }
        return;
    }

    // Phase 2: Copy original V to v_store
    uint v_elem = elem - rope_work;
    uint v_total = NUM_TOKENS * NUM_KV_HEADS * HEAD_DIM;
    if (v_elem < v_total) {
        v_store[v_elem] = v_in[v_elem];
    }
"""

_kernel_cache: dict[str, object] = {}


def _get_rope_kernel():
    if "neox" not in _kernel_cache:
        _kernel_cache["neox"] = mx.fast.metal_kernel(
            name="rope_neox",
            input_names=["q_in", "k_in", "cos_sin_cache", "positions"],
            output_names=["q_out", "k_out"],
            source=_ROPE_NEOX_SOURCE,
            header="#include <metal_stdlib>\nusing namespace metal;\n",
        )
    return _kernel_cache["neox"]


def _get_fused_kv_kernel():
    if "neox_kv" not in _kernel_cache:
        _kernel_cache["neox_kv"] = mx.fast.metal_kernel(
            name="rope_neox_with_kv",
            input_names=["q_in", "k_in", "v_in", "cos_sin_cache", "positions"],
            output_names=["q_out", "k_out", "k_store", "v_store"],
            source=_ROPE_NEOX_WITH_KV_SOURCE,
            header="#include <metal_stdlib>\nusing namespace metal;\n",
        )
    return _kernel_cache["neox_kv"]


def rope_neox(
    q: mx.array,
    k: mx.array,
    cos_sin_cache: mx.array,
    positions: mx.array,
    *,
    head_dim: int,
    rope_dim: int,
    num_qo_heads: int,
    num_kv_heads: int,
) -> tuple[mx.array, mx.array]:
    """Apply NeoX-style RoPE to Q and K in a single Metal dispatch.

    Args:
        q: [num_tokens, num_qo_heads, head_dim]
        k: [num_tokens, num_kv_heads, head_dim]
        cos_sin_cache: [max_pos, rope_dim], float32.
        positions: [num_tokens], int32.

    Returns:
        (q_rotated, k_rotated) with same shapes.
    """
    num_tokens = q.shape[0]
    total_heads = num_qo_heads + num_kv_heads
    half_dim = rope_dim // 2
    total_work = num_tokens * total_heads * half_dim
    threadgroup_size = min(256, total_work)

    kernel = _get_rope_kernel()
    outputs = kernel(
        inputs=[q, k, cos_sin_cache, positions],
        template=[
            ("T", q.dtype),
            ("HEAD_DIM", head_dim),
            ("ROPE_DIM", rope_dim),
            ("NUM_QO_HEADS", num_qo_heads),
            ("NUM_KV_HEADS", num_kv_heads),
        ],
        grid=(total_work, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        output_shapes=[q.shape, k.shape],
        output_dtypes=[q.dtype, k.dtype],
    )
    return outputs[0], outputs[1]


def rope_neox_with_kv(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cos_sin_cache: mx.array,
    positions: mx.array,
    *,
    head_dim: int,
    rope_dim: int,
    num_qo_heads: int,
    num_kv_heads: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Fused RoPE + KV store preparation in a single Metal dispatch.

    Applies NeoX-style RoPE to Q and K, and produces pool-layout copies
    of rotated K and original V that can be scattered into MlxKVPool.

    Args:
        q: [num_tokens, num_qo_heads, head_dim]
        k: [num_tokens, num_kv_heads, head_dim]
        v: [num_tokens, num_kv_heads, head_dim]
        cos_sin_cache: [max_pos, rope_dim], float32.
        positions: [num_tokens], int32.

    Returns:
        (q_rotated, k_rotated, k_store, v_store)
        k_store: [num_tokens, num_kv_heads, head_dim] -- rotated K for pool scatter
        v_store: [num_tokens, num_kv_heads, head_dim] -- original V for pool scatter
    """
    num_tokens = q.shape[0]
    total_heads = num_qo_heads + num_kv_heads
    half_dim = rope_dim // 2

    rope_work = num_tokens * total_heads * half_dim
    v_copy_work = num_tokens * num_kv_heads * head_dim
    total_work = rope_work + v_copy_work
    threadgroup_size = min(256, total_work)

    kernel = _get_fused_kv_kernel()
    kv_shape = (num_tokens, num_kv_heads, head_dim)
    outputs = kernel(
        inputs=[q, k, v, cos_sin_cache, positions],
        template=[
            ("T", q.dtype),
            ("HEAD_DIM", head_dim),
            ("ROPE_DIM", rope_dim),
            ("NUM_QO_HEADS", num_qo_heads),
            ("NUM_KV_HEADS", num_kv_heads),
            ("NUM_TOKENS", num_tokens),
        ],
        grid=(total_work, 1, 1),
        threadgroup=(threadgroup_size, 1, 1),
        output_shapes=[q.shape, k.shape, kv_shape, kv_shape],
        output_dtypes=[q.dtype, k.dtype, k.dtype, v.dtype],
    )
    return outputs[0], outputs[1], outputs[2], outputs[3]
