"""Versioned contract for a Torch-exported Core AI serving bundle."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

COREAI_RUNTIME_VERSION = "1.0.0b2"
COREAI_CONVERTER_VERSION = "0.4.2"


@dataclass(frozen=True)
class StateSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Core AI state names must be nonempty strings.")
        if not self.shape or any(type(n) is not int or n <= 0 for n in self.shape):
            raise ValueError("Core AI states require positive, static dimensions.")
        if self.dtype not in ("float16", "float32", "int32"):
            raise ValueError(f"Unsupported Core AI state dtype: {self.dtype}")


@dataclass(frozen=True)
class CoreAIManifest:
    schema_version: int
    model_type: str
    context_length: int
    prefill_chunk_size: int
    vocab_size: int
    dtype: str
    model_config_sha256: str
    states: tuple[StateSpec, ...]
    packages: dict[str, str]
    source_model: str | None
    source_revision: str | None

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported Core AI manifest schema.")
        if self.model_type != "qwen3":
            raise ValueError("Core AI serving currently supports dense Qwen3 only.")
        for name in ("context_length", "prefill_chunk_size", "vocab_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Core AI {name} must be a positive integer.")
        if self.prefill_chunk_size > self.context_length:
            raise ValueError("Core AI prefill chunk exceeds the context length.")
        if self.dtype not in ("float16", "float32"):
            raise ValueError(f"Unsupported Core AI model dtype: {self.dtype}")
        if len(self.states) < 2 or len({s.name for s in self.states}) != len(
            self.states
        ):
            raise ValueError(
                "Core AI requires uniquely named persistent KV/token states."
            )
        token_states = [s for s in self.states if s.name == "next_token"]
        if token_states != [StateSpec("next_token", (1,), "int32")]:
            raise ValueError("Core AI requires a persistent int32 next_token state.")
        if any(s.dtype != self.dtype for s in self.states if s.name != "next_token"):
            raise ValueError("Core AI KV state/model dtypes must match.")
        if (
            not isinstance(self.packages, dict)
            or self.packages.get("coreai-core") != COREAI_RUNTIME_VERSION
            or self.packages.get("coreai-torch") != COREAI_CONVERTER_VERSION
            or not self.packages.get("torch")
        ):
            raise ValueError(
                "Re-export this bundle with the supported Core AI versions."
            )


def save_manifest(bundle_path: str | Path, manifest: CoreAIManifest) -> None:
    path = Path(bundle_path) / "coreai-manifest.json"
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")


def load_manifest(bundle_path: str | Path) -> CoreAIManifest:
    bundle = Path(bundle_path)
    document = json.loads((bundle / "coreai-manifest.json").read_text())
    if not isinstance(document, dict):
        raise ValueError("Core AI manifest must be a JSON object.")
    document["states"] = tuple(
        StateSpec(name=s["name"], shape=tuple(s["shape"]), dtype=s["dtype"])
        for s in document["states"]
    )
    manifest = CoreAIManifest(**document)
    config_bytes = (bundle / "config.json").read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != manifest.model_config_sha256:
        raise ValueError("The model config does not match the Core AI artifact.")
    config = json.loads(config_bytes)
    if (
        config.get("model_type") != manifest.model_type
        or config.get("vocab_size") != manifest.vocab_size
    ):
        raise ValueError("Core AI manifest and model config disagree.")
    if not (bundle / "model.aimodel").is_dir():
        raise ValueError(f"Missing compiled Core AI asset: {bundle / 'model.aimodel'}")
    return manifest
