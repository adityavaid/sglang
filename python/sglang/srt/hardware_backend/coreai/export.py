"""Convert an already-loaded Torch module into a Core AI artifact.

The exporter is intentionally tensor-only: a caller supplies the exact export
module and example tensors.  SGLang model forwards consume scheduler metadata
and paged-attention state, so exporting a live ``ModelRunner.model`` directly
would silently capture invalid serving semantics.  A model-specific adapter
must first expose a tensor-only prefill/decode entry point with explicit state.
"""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class CoreAIExportSpec:
    """The explicit contract for one Torch-to-Core-AI entry point."""

    entrypoint_name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    example_args: tuple[Any, ...] = ()
    example_kwargs: Mapping[str, Any] | None = None
    dynamic_shapes: Any | None = None
    state_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.entrypoint_name:
            raise ValueError("entrypoint_name must not be empty")
        if not self.input_names:
            raise ValueError("input_names must not be empty")
        if len(set(self.input_names)) != len(self.input_names):
            raise ValueError("input_names must be unique")
        if len(set(self.output_names)) != len(self.output_names):
            raise ValueError("output_names must be unique")
        if set(self.state_names) & set(self.input_names):
            raise ValueError("state_names and input_names must be disjoint")
        if len(set(self.state_names)) != len(self.state_names):
            raise ValueError("state_names must be unique")
        if bool(self.example_args) == bool(self.example_kwargs):
            raise ValueError("provide exactly one of example_args or example_kwargs")


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_coreai_dependencies() -> tuple[Any, Any]:
    try:
        import coreai_torch
        from coreai_torch import TorchConverter
    except ImportError as exc:
        raise RuntimeError(
            "Core AI export requires coreai-torch. Install the isolated Core AI "
            "environment; do not add Core AI as an unconditional SGLang dependency."
        ) from exc
    return coreai_torch, TorchConverter


def _export_program(module: torch.nn.Module, spec: CoreAIExportSpec) -> torch.export.ExportedProgram:
    module.eval()
    with torch.no_grad():
        if spec.example_kwargs is not None:
            program = torch.export.export(
                module,
                args=(),
                kwargs=dict(spec.example_kwargs),
                dynamic_shapes=spec.dynamic_shapes,
            )
        else:
            program = torch.export.export(
                module,
                args=spec.example_args,
                dynamic_shapes=spec.dynamic_shapes,
            )
    coreai_torch, _ = _load_coreai_dependencies()
    return program.run_decompositions(coreai_torch.get_decomp_table())


def _write_manifest(
    artifact_dir: Path, spec: CoreAIExportSpec, source_model: str | None
) -> None:
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model": source_model,
        "entrypoint": asdict(spec),
        "packages": {
            name: _version(name)
            for name in ("torch", "coreai-core", "coreai-torch")
        },
        "serving_status": "artifact-only; no SGLang Core AI serving worker is enabled",
    }
    (artifact_dir / "sglang-coreai-manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_loaded_torch_model(
    module: torch.nn.Module,
    spec: CoreAIExportSpec,
    artifact_dir: str | Path,
    *,
    source_model: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Export a loaded Torch module to an optimized Core AI ``.aimodel``.

    ``artifact_dir`` is the final asset directory.  The function never mutates
    the module, nor does it claim that its Torch parameter storage can be
    reused at runtime: Core AI compiles its own executable weight asset.
    """
    _coreai_torch, converter_type = _load_coreai_dependencies()
    output = Path(artifact_dir)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Core AI artifact already exists: {output}")
        if not output.is_dir():
            raise ValueError(f"Core AI artifact path is not a directory: {output}")
        import shutil

        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    exported_program = _export_program(module, spec)
    converter = converter_type()
    converter.add_exported_program(
        exported_program,
        entrypoint_name=spec.entrypoint_name,
        input_names=spec.input_names,
        output_names=spec.output_names,
        state_names=spec.state_names or None,
    )
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(output)
    _write_manifest(output, spec, source_model)
    return output
