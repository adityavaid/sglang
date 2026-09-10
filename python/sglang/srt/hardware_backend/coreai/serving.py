"""Strict admission for the experimental, greedy, batch-one Core AI backend."""

import json
from pathlib import Path

from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.hardware_backend.coreai.artifact import load_manifest


def resolve_server_args(server_args):
    """Declare bookkeeping defaults before ordinary device/model resolution."""
    cfg = resolving_view(server_args)
    if not cfg.coreai_artifact_path:
        raise ValueError(
            "Core AI requires --coreai-artifact-path (a bundle directory)."
        )
    bundle = Path(cfg.coreai_artifact_path).expanduser().resolve()
    if Path(cfg.model_path).expanduser().resolve() != bundle:
        raise ValueError("Core AI --model-path must match --coreai-artifact-path.")
    if cfg.tokenizer_path and Path(cfg.tokenizer_path).expanduser().resolve() != bundle:
        raise ValueError("Core AI --tokenizer-path must match --coreai-artifact-path.")
    manifest = load_manifest(bundle)
    if manifest.model_type != "qwen3" or manifest.dtype != "float16":
        raise ValueError("Core AI serving supports dense Qwen3 float16 artifacts only.")
    if json.loads(cfg.json_model_override_args or "{}"):
        raise ValueError("Core AI does not support model config overrides.")
    if cfg.mem_fraction_static is not None:
        raise ValueError(
            "Core AI uses fixed artifact memory; mem_fraction_static is unsupported."
        )

    for name in (
        "tp_size",
        "pp_size",
        "dp_size",
        "dcp_size",
        "attn_cp_size",
        "moe_dp_size",
        "ep_size",
        "dwdp_size",
    ):
        if getattr(cfg, name) != 1:
            raise ValueError(f"Core AI requires {name}=1.")
    for name, allowed in (
        ("device", (None, "cpu")),
        ("dtype", ("auto", "float16")),
        ("max_running_requests", (None, 1)),
        ("page_size", (None, 1)),
        ("kv_cache_dtype", ("auto", "float16")),
        ("load_format", ("auto",)),
        ("startup_weight_load_mode", ("serial",)),
        ("disaggregation_mode", ("null",)),
        ("weight_cache_mode", ("off",)),
        ("model_impl", ("auto",)),
    ):
        if getattr(cfg, name) not in allowed:
            raise ValueError(f"Core AI does not support {name}={getattr(cfg, name)!r}.")
    for name in (
        "speculative_algorithm",
        "speculative_draft_model_path",
        "quantization",
        "quantization_param_path",
        "enable_lora",
        "lora_paths",
        "is_embedding",
        "enable_dp_attention",
        "enable_hierarchical_cache",
        "enable_lmcache",
        "enable_hisparse",
        "enable_two_batch_overlap",
        "enable_single_batch_overlap",
        "enable_pdmux",
        "dllm_algorithm",
        "cpu_offload_gb",
        "enable_memory_saver",
        "enable_return_hidden_states",
        "enable_return_routed_experts",
        "enable_return_indexer_topk",
        "enable_custom_logit_processor",
        "enable_session_radix_cache",
        "enable_prefix_mm_cache",
        "enable_unified_cache_external_linker",
        "prefill_only_disable_kv_cache",
        "elastic_ep_backend",
        "mlx_enable_sampling",
    ):
        if getattr(cfg, name, None):
            raise ValueError(f"Core AI does not support {name}.")
    for name in (
        "attention_backend",
        "prefill_attention_backend",
        "decode_attention_backend",
    ):
        if getattr(cfg, name) is not None:
            raise ValueError(
                f"Core AI owns attention execution; {name} is unsupported."
            )
    for name in ("cuda_graph_backend_decode", "cuda_graph_backend_prefill"):
        if getattr(cfg, name) not in (None, "disabled"):
            raise ValueError("Core AI does not support CUDA graphs.")
    if cfg.cuda_graph_tc_compiler is not None:
        raise ValueError("Core AI does not support Torch compilation.")
    graph = cfg.cuda_graph_config
    if graph:
        if hasattr(graph, "to_dict"):
            graph = graph.to_dict()
        if any(
            phase.get("backend", "disabled") != "disabled" for phase in graph.values()
        ):
            raise ValueError("Core AI does not support CUDA graphs.")

    for name in ("context_length", "max_total_tokens"):
        value = getattr(cfg, name)
        if value is not None and not 8 <= value <= manifest.context_length:
            raise ValueError(
                f"Core AI {name} must be between 8 and the artifact context length."
            )
    context = cfg.context_length or manifest.context_length
    capacity = cfg.max_total_tokens or context
    chunk = cfg.chunked_prefill_size
    if chunk is not None and not 1 <= chunk <= manifest.prefill_chunk_size:
        raise ValueError(
            "Core AI chunked_prefill_size exceeds the artifact prefill chunk."
        )
    declare_resolution(
        server_args,
        "_handle_coreai_backend",
        coreai_artifact_path=str(bundle),
        device="cpu",
        dtype="float16",
        max_running_requests=1,
        max_total_tokens=capacity,
        context_length=min(context, capacity),
        chunked_prefill_size=min(chunk or manifest.prefill_chunk_size, capacity),
        page_size=1,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
    )


def _present(value):
    if isinstance(value, (list, tuple)):
        return any(_present(item) for item in value)
    return value is not None and value is not False


def validate_request(request, sampling_params, *, input_embeds=None, mm_inputs=None):
    """Validate normalized sampling semantics before scheduler admission."""
    validate_request_inputs(request)
    if input_embeds is not None or mm_inputs is not None:
        raise ValueError(
            "Core AI does not support input embeddings or multimodal inputs."
        )
    if sampling_params.top_k != 1:
        raise ValueError("Core AI supports greedy sampling only (temperature=0).")
    for name, default in (
        ("frequency_penalty", 0),
        ("presence_penalty", 0),
        ("repetition_penalty", 1),
        ("min_new_tokens", 0),
        ("n", 1),
    ):
        if getattr(sampling_params, name) != default:
            raise ValueError(f"Core AI does not support sampling parameter {name}.")
    for name in (
        "json_schema",
        "regex",
        "ebnf",
        "structural_tag",
        "logit_bias",
        "custom_params",
        "beam_width",
    ):
        if getattr(sampling_params, name):
            raise ValueError(f"Core AI does not support sampling parameter {name}.")


def validate_request_inputs(request):
    """Reject unsupported inputs before preprocessing or loading an adapter."""
    from sglang.srt.managers.io_struct import GenerateReqInput

    if not isinstance(request, GenerateReqInput):
        raise ValueError("Core AI supports text generation, not embedding requests.")
    for name in (
        "return_logprob",
        "return_hidden_states",
        "return_routed_experts",
        "return_sampling_mask",
        "return_indexer_topk",
        "return_flat_raw_top_logprobs",
        "return_entropy",
        "custom_logit_processor",
        "input_embeds",
        "image_data",
        "audio_data",
        "video_data",
        "lora_path",
        "lora_id",
        "session_params",
        "positional_embed_overrides",
        "encoder_urls",
    ):
        if _present(getattr(request, name, None)):
            raise ValueError(f"Core AI does not support request field {name}.")
    if request.max_thinking_tokens is not None:
        raise ValueError("Core AI does not support a thinking budget.")
    top_logprobs = request.top_logprobs_num
    has_top_logprobs = (
        any(top_logprobs) if isinstance(top_logprobs, list) else bool(top_logprobs)
    )
    if has_top_logprobs or _present(request.token_ids_logprob):
        raise ValueError("Core AI does not produce log probabilities.")
