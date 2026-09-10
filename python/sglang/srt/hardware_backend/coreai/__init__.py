"""Opt-in Torch export and experimental Core AI execution.

Serving modules and optional Core AI dependencies are imported only on demand.
Generic exports below are not serving bundles; use ``prepare_loaded_qwen3``
from ``coreai.prepare`` for the versioned Qwen3 serving contract.
"""

from sglang.srt.hardware_backend.coreai.export import (
    CoreAIExportSpec,
    export_loaded_torch_model,
)

__all__ = ["CoreAIExportSpec", "export_loaded_torch_model"]
