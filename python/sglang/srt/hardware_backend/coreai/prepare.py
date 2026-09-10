"""Prepare a serving bundle from a loaded Torch Qwen3, without Core AI Models."""

import argparse
import copy
import hashlib
import importlib.metadata
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.export.graph_signature import InputKind

from sglang.srt.hardware_backend.coreai.artifact import (
    CoreAIManifest,
    StateSpec,
    save_manifest,
)
from sglang.srt.hardware_backend.coreai.export import (
    CoreAIExportSpec,
    _export_program,
    _load_coreai_dependencies,
)


@contextmanager
def _native_loading_scope(model_path, revision):
    """Own only preparation's runtime state; never tear down a caller's groups."""
    from sglang.srt.runtime_context import (
        get_context,
        get_parallel,
        reset_context,
        restore_context,
        snapshot_context,
    )
    from sglang.srt.utils.common import temp_set_env

    if os.environ.get("SGLANG_USE_CPU_ENGINE") == "1":
        raise ValueError(
            "Preparation requires SGLANG_USE_CPU_ENGINE unset (no CPU kernels)."
        )
    with temp_set_env(allow_sglang=True, SGLANG_USE_COREAI="0"):
        from sglang.srt.distributed import parallel_state as ps
        from sglang.srt.server_args import ServerArgs

        existing = torch.distributed.is_initialized()
        if existing and (
            not ps.model_parallel_is_initialized()
            or ps.get_tp_group().world_size != 1
            or ps.get_pp_group().world_size != 1
            or torch.distributed.get_world_size() != 1
        ):
            raise ValueError(
                "Core AI preparation requires a standalone TP=PP=1 runtime."
            )
        state = snapshot_context()
        state["__parallel__"] = {
            key: value.copy() if isinstance(value, dict) else value
            for key, value in state["__parallel__"].items()
        }
        self_pp = ps._SELF_PP
        try:
            reset_context()
            get_parallel()._overrides = {}
            args = ServerArgs(
                model_path=str(model_path),
                revision=revision,
                device="cpu",
                dtype="float16",
                trust_remote_code=False,
                attention_backend="torch_native",
                disable_cuda_graph=True,
            )
            # Loading needs configuration bags, not server resolution (which
            # probes Linux NUMA/CPU serving kernels even on macOS).
            get_context().set_server_args(args)
            if existing:
                yield
            else:
                with tempfile.TemporaryDirectory(
                    prefix=".coreai-load-", dir=Path.cwd()
                ) as directory:
                    ps.init_distributed_environment(
                        world_size=1,
                        rank=0,
                        local_rank=0,
                        backend="gloo",
                        distributed_init_method=(
                            Path(directory) / "rendezvous"
                        ).as_uri(),
                    )
                    ps.initialize_model_parallel(backend="gloo")
                    yield
        finally:
            try:
                if not existing:
                    ps.destroy_model_parallel()
                    # Standard teardown currently omits the singleton draft
                    # PP group; dispose only the one this scope created.
                    if ps._SELF_PP is not self_pp:
                        ps._SELF_PP.destroy()
                        ps._SELF_PP = self_pp
                    ps.destroy_distributed_environment()
            finally:
                restore_context(state)


def load_native_qwen3(model_path: str | Path, *, revision: str | None = None):
    """Load CPU FP16 native SGLang weights via the standard DefaultModelLoader."""
    with _native_loading_scope(model_path, revision):
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.configs.load_config import LoadConfig
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.hardware_backend.coreai.qwen3 import validate_qwen3_config
        from sglang.srt.layers.rotary_embedding import factory as rope_factory
        from sglang.srt.model_loader import get_model

        config = ModelConfig(
            str(model_path),
            revision=revision,
            trust_remote_code=False,
            dtype="float16",
            model_impl="sglang",
        )
        validate_qwen3_config(config.hf_config)
        if config.quantization is not None:
            raise ValueError("Core AI preparation does not support quantization.")
        # Native RoPE's cache shares mutable modules across models and its key
        # omits device. Keep sharing within this load, not with another runtime.
        rope_cache = rope_factory._ROPE_DICT
        rope_factory._ROPE_DICT = {}
        try:
            return get_model(
                model_config=config,
                load_config=LoadConfig(),
                device_config=DeviceConfig("cpu"),
            )
        finally:
            rope_factory._ROPE_DICT = rope_cache


class _GreedyStep(torch.nn.Module):
    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self.register_buffer("next_token", torch.zeros(1, dtype=torch.int32))

    def forward(self, input_ids, start_position):
        logits = self.adapter(input_ids, start_position)
        self.next_token.copy_(logits.argmax(dim=-1).to(torch.int32))
        return ()


def prepare_loaded_qwen3(
    model,
    bundle_path: str | Path,
    *,
    max_context_length: int = 2048,
    prefill_chunk_size: int = 64,
    tokenizer=None,
    source_model: str | None = None,
    source_revision: str | None = None,
) -> Path:
    """Export two whole-forward functions sharing loaded parameters and state.

    The caller retains its Torch model. The CLI exits after preparation so the
    serving process need not retain a second, eager weight representation.
    """
    output = Path(bundle_path).resolve()
    if output.exists():
        raise FileExistsError(f"Core AI bundle already exists: {output}")
    if (
        type(max_context_length) is not int
        or max_context_length <= 0
        or type(prefill_chunk_size) is not int
        or not 1 <= prefill_chunk_size <= max_context_length
    ):
        raise ValueError("Core AI prefill size must fit a positive context length.")
    dtype = next(model.parameters()).dtype
    if dtype not in (torch.float16, torch.float32):
        raise ValueError("Core AI export requires loaded float16 or float32 weights.")

    from sglang.srt.hardware_backend.coreai.qwen3 import Qwen3TorchAdapter

    adapter = Qwen3TorchAdapter(model, max_context_length)
    step = _GreedyStep(adapter).eval()
    _, converter_type = _load_coreai_dependencies()
    converter = converter_type()
    targets = None
    state_specs: tuple[StateSpec, ...] = ()
    for name, length in (("decode", 1), ("prefill", prefill_chunk_size)):
        adapter.reset()
        exported = _export_program(
            step,
            CoreAIExportSpec(
                entrypoint_name=name,
                input_names=("input_ids", "start_position"),
                output_names=(),
                example_args=(
                    torch.zeros((1, length), dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32),
                ),
            ),
        )
        mutated = set(exported.graph_signature.buffers_to_mutate.values())
        current_targets = tuple(
            spec.target
            for spec in exported.graph_signature.input_specs
            if spec.kind == InputKind.BUFFER and spec.target in mutated
        )
        expected = {"next_token", *(f"adapter.{name}" for name in adapter.state_names)}
        if set(current_targets) != expected:
            raise ValueError(
                "Torch export did not preserve the complete Core AI state ABI."
            )
        if targets is not None and targets != current_targets:
            raise ValueError("Core AI prefill/decode state ordering differs.")
        targets = current_targets
        state_specs = tuple(
            StateSpec(
                name="next_token" if target == "next_token" else f"kv_{index}",
                shape=tuple(exported.state_dict[target].shape),
                dtype=str(exported.state_dict[target].dtype).removeprefix("torch."),
            )
            for index, target in enumerate(targets)
        )
        converter.add_exported_program(
            exported,
            input_names=("input_ids", "start_position"),
            output_names=(),
            state_names=tuple(spec.name for spec in state_specs),
            entrypoint_name=name,
        )

    program = converter.to_coreai()
    program.optimize()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".coreai-", dir=output.parent) as directory:
        temporary = Path(directory) / "bundle"
        temporary.mkdir()
        program.save_asset(temporary / "model.aimodel")
        config = copy.deepcopy(model.config)
        config.architectures = ["Qwen3ForCausalLM"]
        config.dtype = dtype
        config.save_pretrained(temporary)
        if tokenizer is not None:
            tokenizer.save_pretrained(temporary)
        save_manifest(
            temporary,
            CoreAIManifest(
                schema_version=1,
                model_type="qwen3",
                context_length=max_context_length,
                prefill_chunk_size=prefill_chunk_size,
                vocab_size=config.vocab_size,
                dtype=str(dtype).removeprefix("torch."),
                model_config_sha256=hashlib.sha256(
                    (temporary / "config.json").read_bytes()
                ).hexdigest(),
                states=state_specs,
                packages={
                    name: importlib.metadata.version(name)
                    for name in ("torch", "transformers", "coreai-core", "coreai-torch")
                },
                source_model=source_model,
                source_revision=source_revision,
            ),
        )
        if output.exists():
            raise FileExistsError(f"Core AI bundle already exists: {output}")
        temporary.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--prefill-chunk-size", type=int, default=64)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new bundle path")
    from transformers import AutoTokenizer

    model = load_native_qwen3(args.model, revision=args.revision)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False
    )
    bundle = prepare_loaded_qwen3(
        model,
        args.output_dir,
        max_context_length=args.context_length,
        prefill_chunk_size=args.prefill_chunk_size,
        tokenizer=tokenizer,
        source_model=args.model,
        source_revision=getattr(model.config, "_commit_hash", None) or args.revision,
    )
    print(bundle)


if __name__ == "__main__":
    main()
