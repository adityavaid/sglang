#!/usr/bin/env python3
"""Export-gate tooling for the Core AI Qwen3 proof of concept.

This module intentionally has no import from ``sglang``.  It is usable from a
small, pinned export environment and cannot accidentally activate a serving
backend.  Serving integration begins only after this experiment proves that an
accelerated, long-lived Core AI runtime can own its KV state safely.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
COREAI_MODELS_REVISION = "7afb40821654fa8bc3e5049ea9bf7edc72df80d5"


@dataclass(frozen=True)
class RuntimeProbe:
    """The information needed to decide whether an export is worth attempting."""

    macos_version: str
    machine: str
    python_version: str
    packages: dict[str, str | None]
    specialization_supported: bool | None
    errors: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.errors and self.specialization_supported is True


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _macos_version() -> str:
    if platform.system() != "Darwin":
        return "not-macos"
    result = subprocess.run(
        ["sw_vers", "-productVersion"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def probe_runtime() -> RuntimeProbe:
    """Check the accelerated runtime contract without creating a model or artifact."""
    packages = {
        name: _package_version(name)
        for name in ("coreai-core", "coreai-torch", "coreai-models", "torch")
    }
    errors: list[str] = []
    macos_version = _macos_version()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        errors.append("Core AI POC requires Apple Silicon macOS.")
    elif macos_version != "unknown" and int(macos_version.split(".", 1)[0]) < 27:
        errors.append("Core AI acceleration requires macOS 27 or newer.")

    missing = [name for name, version in packages.items() if version is None]
    if missing:
        errors.append(f"Missing required package(s): {', '.join(missing)}.")

    specialization_supported: bool | None = None
    if not missing:
        try:
            from coreai.runtime import SpecializationOptions

            specialization_supported = bool(SpecializationOptions.is_supported())
            if not specialization_supported:
                errors.append("System Core AI specialization is unavailable.")
        except Exception as exc:  # Runtime imports may fail before specialization.
            errors.append(f"Unable to query Core AI specialization: {exc}")

    return RuntimeProbe(
        macos_version=macos_version,
        machine=platform.machine(),
        python_version=platform.python_version(),
        packages=packages,
        specialization_supported=specialization_supported,
        errors=tuple(errors),
    )


def _artifact_path(output_dir: Path) -> Path:
    artifacts = sorted(output_dir.glob("*.aimodel"))
    if len(artifacts) != 1:
        raise RuntimeError(
            f"Expected exactly one .aimodel in {output_dir}, found {len(artifacts)}."
        )
    return artifacts[0]


def export_qwen3(
    *,
    model: str,
    output_dir: Path,
    max_context_length: int,
    compression: str,
    overwrite: bool,
) -> Path:
    """Export the upstream Qwen3 Core AI graph and persist a reproducibility manifest."""
    probe = probe_runtime()
    if not probe.ready:
        raise RuntimeError("Runtime gate failed:\n- " + "\n- ".join(probe.errors))
    from coreai_models.export import ExportConfig, export_model

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = Path(
        export_model(
            ExportConfig(
                hf_model_id=model,
                variant="macOS",
                max_context_length=max_context_length,
                compute_precision="float16",
                compression=compression,
                output_dir=str(output_dir),
                overwrite=overwrite,
            )
        )
    )
    # The upstream exporter returns a bundle directory, not an individual
    # .aimodel.  Validate its content rather than guessing the generated name.
    artifact = _artifact_path(bundle_path)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "max_context_length": max_context_length,
        "compute_precision": "float16",
        "compression": compression,
        "coreai_models_revision": COREAI_MODELS_REVISION,
        "artifact": artifact.name,
        "runtime_probe": asdict(probe),
        "entrypoints_expected": ["prefill", "main"],
        "serving_contract": {
            "kv_owner": "Core AI native runtime (not yet implemented in SGLang)",
            "scheduler_owner": "SGLang (not connected in this POC)",
        },
    }
    (bundle_path / "artifact-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("probe", help="validate the accelerated Core AI runtime")
    export = commands.add_parser("export", help="export Qwen3 as a Core AI artifact")
    export.add_argument("--model", default=DEFAULT_MODEL)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--max-context-length", type=int, default=2048)
    export.add_argument(
        "--compression", default="none", help="Core AI Models compression preset; default is FP16."
    )
    export.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.command == "probe":
        probe = probe_runtime()
        print(json.dumps(asdict(probe), indent=2, sort_keys=True))
        return 0 if probe.ready else 1
    if args.max_context_length < 4:
        parser.error("--max-context-length must be at least 4.")
    artifact = export_qwen3(
        model=args.model,
        output_dir=args.output_dir,
        max_context_length=args.max_context_length,
        compression=args.compression,
        overwrite=args.overwrite,
    )
    print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
