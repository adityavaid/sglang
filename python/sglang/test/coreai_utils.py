"""Tiny native checkpoints with independent Transformers test oracles."""

import tempfile
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def checkpoint_pair(**config_overrides):
    from sglang.srt.hardware_backend.coreai.prepare import load_native_qwen3

    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=41,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=16,
        **config_overrides,
    )
    config._attn_implementation = "eager"
    oracle = Qwen3ForCausalLM(config).half().eval()
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
        oracle.save_pretrained(directory)
        model = load_native_qwen3(directory)
    return model, oracle
