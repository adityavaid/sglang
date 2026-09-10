"""Opt-in Core AI selection; never substitute the bundled CPU interpreter."""

import os
import platform

from sglang.srt.environ import envs


def use_coreai() -> bool:
    enabled = bool(envs.SGLANG_USE_COREAI.get())
    if enabled and envs.SGLANG_USE_MLX.get():
        raise ValueError("Core AI and MLX cannot both be selected.")
    if enabled and os.environ.get("SGLANG_USE_CPU_ENGINE") == "1":
        raise ValueError(
            "Core AI and the Torch CPU engine cannot both be selected; "
            "unset SGLANG_USE_CPU_ENGINE."
        )
    return enabled


def reject_torch_model_control(operation: str) -> None:
    if use_coreai():
        raise ValueError(
            f"Core AI does not support {operation} on opaque compiled model state."
        )


def validate_runtime() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Core AI serving requires Apple Silicon.")
    version = platform.mac_ver()[0]
    if not version or int(version.split(".", 1)[0]) < 27:
        raise RuntimeError(
            "Core AI serving requires macOS 27 or newer. The bundled CPU "
            "interpreter is not a serving fallback."
        )
    try:
        from coreai.runtime import SpecializationOptions
    except ImportError as exc:
        raise RuntimeError("Core AI serving requires coreai-core==1.0.0b2.") from exc
    if not SpecializationOptions.is_supported():
        raise RuntimeError("System Core AI specialization is unavailable.")
