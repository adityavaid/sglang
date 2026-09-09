"""Opt-in Core AI artifact preparation for Torch-authored models.

This package deliberately prepares artifacts only.  It must not be selected as
a serving backend until a Core AI worker owns persistent KV state and honours
SGLang's scheduler/cache contracts.
"""

from sglang.srt.hardware_backend.coreai.export import (
    CoreAIExportSpec,
    export_loaded_torch_model,
)

__all__ = ["CoreAIExportSpec", "export_loaded_torch_model"]
