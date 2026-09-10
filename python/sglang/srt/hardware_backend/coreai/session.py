"""Synchronous, single-owner execution of compiled Core AI functions."""

import asyncio
import importlib.metadata
import logging
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sglang.srt.hardware_backend.coreai.artifact import (
    COREAI_RUNTIME_VERSION,
    load_manifest,
)
from sglang.srt.hardware_backend.coreai.runtime import validate_runtime

logger = logging.getLogger(__name__)


class CoreAISession:
    """Own the executable and persistent KV/token state; never execute Torch.

    ``next_token`` is a mutable scalar state, not an allocated function output.
    This avoids requesting a new runtime output NDArray on every decode step.
    macOS 27 endurance/placement qualification is still required.
    """

    def __init__(self, artifact_path: str | Path):
        validate_runtime()
        self._bundle = Path(artifact_path).resolve()
        self.manifest = load_manifest(self._bundle)
        if importlib.metadata.version("coreai-core") != COREAI_RUNTIME_VERSION:
            raise RuntimeError("Core AI runtime and compiled bundle versions differ.")
        self.position = 0
        self._owner = threading.get_ident()
        self._closed = False
        self._failed = False
        self._busy = False
        self._model = None
        self._functions = {}
        self._state = {}
        self._loop = asyncio.new_event_loop()
        ready = False
        try:
            self._model = self._loop.run_until_complete(self._load_model())
            self._functions = {
                name: self._model.load_function(name) for name in ("decode", "prefill")
            }
            self._validate_descriptors()
            self._state = self._new_state()
            self.extend([0] * self.manifest.prefill_chunk_size, 0)
            self.reset()
            self.extend([0], 0)
            self.reset()
            ready = True
        finally:
            if not ready:
                self.close()
        logger.info(
            "Core AI executable ready: GPU preferred, persistent KV/token state, "
            "no per-step output NDArrays. GPU placement is not guaranteed by preference."
        )

    async def _load_model(self):
        from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions

        options = SpecializationOptions.from_preferred_compute_unit_kind(
            ComputeUnitKind.gpu()
        ).with_debug(enabled=True)
        return await AIModel.load(
            self._bundle / "model.aimodel", specialization_options=options
        )

    def _new_state(self):
        from coreai.runtime import NDArray, StorageKind

        return {
            spec.name: NDArray(
                np.zeros(spec.shape, dtype=spec.dtype), backing=StorageKind.METAL
            )
            for spec in self.manifest.states
        }

    def _validate_descriptors(self):
        for name, function in self._functions.items():
            desc = function.desc
            if set(desc.input_names) != {"input_ids", "start_position"}:
                raise ValueError(f"Unexpected Core AI {name} input ABI.")
            if desc.output_names:
                raise ValueError(
                    "Core AI serving functions must use persistent token state."
                )
            if set(desc.state_names) != {s.name for s in self.manifest.states}:
                raise ValueError(
                    f"Core AI {name} state names differ from the manifest."
                )
            length = 1 if name == "decode" else self.manifest.prefill_chunk_size
            for input_name, shape in (
                ("input_ids", (1, length)),
                ("start_position", (1,)),
            ):
                actual = desc.input_descriptor(input_name)
                if tuple(actual.shape) != shape or actual.dtype != "int32":
                    raise ValueError(
                        f"Core AI {name}.{input_name} has an incompatible descriptor."
                    )
            for spec in self.manifest.states:
                actual = desc.state_descriptor(spec.name)
                if tuple(actual.shape) != spec.shape or actual.dtype != spec.dtype:
                    raise ValueError(f"Core AI state descriptor mismatch: {spec.name}")

    def _check_ready(self):
        if self._closed:
            raise RuntimeError("Core AI session is closed.")
        if self._failed:
            raise RuntimeError(
                "Core AI session failed; its mutable state cannot be reused."
            )
        if self._busy or threading.get_ident() != self._owner:
            raise RuntimeError(
                "Core AI session requires serial execution by its owner thread."
            )

    def extend(self, input_ids: Sequence[int], start_position: int) -> int:
        self._check_ready()
        tokens = list(input_ids)
        if type(start_position) is not int or start_position != self.position:
            raise ValueError(
                "Core AI requires contiguous positions in the current request."
            )
        if not tokens or any(
            type(token) is not int or not 0 <= token < self.manifest.vocab_size
            for token in tokens
        ):
            raise ValueError(
                "Core AI requires nonempty, vocabulary-bounded integer tokens."
            )
        if self.position + len(tokens) > self.manifest.context_length:
            raise ValueError("Core AI request exceeds the compiled context length.")

        from coreai.runtime import NDArray

        succeeded = False
        self._busy = True
        try:
            offset = 0
            while offset < len(tokens):
                length = (
                    self.manifest.prefill_chunk_size
                    if len(tokens) - offset >= self.manifest.prefill_chunk_size
                    else 1
                )
                name = (
                    "prefill"
                    if length == self.manifest.prefill_chunk_size
                    else "decode"
                )
                inputs = {
                    "input_ids": NDArray(
                        np.asarray([tokens[offset : offset + length]], dtype=np.int32)
                    ),
                    "start_position": NDArray(
                        np.asarray([self.position], dtype=np.int32)
                    ),
                }
                outputs = self._loop.run_until_complete(
                    self._functions[name](inputs=inputs, state=self._state)
                )
                if outputs:
                    raise RuntimeError("Core AI returned unexpected allocated outputs.")
                self.position += length
                offset += length
            token = int(self._state["next_token"].numpy().item())
            if not 0 <= token < self.manifest.vocab_size:
                raise RuntimeError("Core AI produced an invalid token ID.")
            succeeded = True
            return token
        finally:
            self._busy = False
            if not succeeded:
                self._failed = True

    def reset(self) -> None:
        self._check_ready()
        # The exported causal mask hides the old suffix; the new request
        # overwrites every visible row. No full KV copy/zero is needed.
        self.position = 0

    def close(self) -> None:
        if self._closed:
            return
        if self._busy or threading.get_ident() != self._owner:
            raise RuntimeError("Cannot close Core AI while another execution owns it.")
        self._closed = True
        self._functions.clear()
        self._state.clear()
        self._model = None
        self._loop.close()
