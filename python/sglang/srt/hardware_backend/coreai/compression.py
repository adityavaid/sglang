"""Weight-only Core AI compression without modifying the standard-loaded model."""

import torch
from torch import nn


class Qwen3QuantizedWeights(nn.Module):
    """Compress independent projection copies; bind them only during export.

    Embeddings, the (possibly tied) LM head, biases and normalization stay in
    their original precision. Core AI owns compressed constants in the asset;
    eager execution reconstructs floating weights and is not a fast serving path.
    """

    def __init__(self, model: nn.Module, mode: str):
        super().__init__()
        if mode not in ("int4", "int8"):
            raise ValueError("weight_quantization must be int4 or int8")
        try:
            from coreai_opt.base_model_compressor import ExportBackend
            from coreai_opt.quantization import Quantizer, QuantizerConfig
        except ImportError as exc:
            raise RuntimeError(
                "Quantized Core AI export requires requirements-quantization.txt "
                "(coreai-opt==0.2.1 and Torch 2.11)."
            ) from exc

        config = QuantizerConfig.from_dict(
            {
                "quantization_config": {
                    "execution_mode": "eager",
                    "global_config": {
                        "op_state_spec": {
                            "weight": {
                                "dtype": mode,
                                "qscheme": "symmetric_with_clipping",
                                "granularity": {
                                    "type": "per_block",
                                    "block_size": 32,
                                    "axis": 1,
                                },
                            }
                        },
                        "op_input_spec": None,
                        "op_output_spec": None,
                    },
                }
            }
        )
        self.projections = nn.ModuleList()
        names = []
        for index, layer in enumerate(model.model.layers):
            for path in (
                "self_attn.qkv_proj",
                "self_attn.o_proj",
                "mlp.gate_up_proj",
                "mlp.down_proj",
            ):
                weight = layer.get_submodule(path).weight
                name = f"model.model.layers.{index}.{path}.weight"
                if weight.ndim != 2 or weight.shape[1] % 32:
                    raise ValueError(
                        f"Block-32 compression requires a divisible input width: {name}"
                    )
                if not torch.isfinite(weight).all():
                    raise ValueError(f"Cannot quantize nonfinite weights: {name}")
                # Only one floating-point projection copy is live at a time.
                projection = nn.Linear(
                    weight.shape[1], weight.shape[0], bias=False, device="meta"
                )
                projection.weight = nn.Parameter(
                    weight.detach().clone(), requires_grad=False
                )
                projection.eval()
                quantizer = Quantizer(projection, config)
                prepared = quantizer.prepare((weight.new_zeros((1, weight.shape[1])),))
                compressed = quantizer.finalize(prepared, backend=ExportBackend.CoreAI)
                if not nn.utils.parametrize.is_parametrized(
                    compressed, "weight"
                ) or any(p.numel() for p in compressed.parameters()):
                    raise RuntimeError(f"Core AI did not compress projection: {name}")
                self.projections.append(compressed)
                names.append(name)
        if not names:
            raise ValueError("Core AI compression requires Qwen3 decoder projections")
        self.names = tuple(names)
        self.eval()

    def forward(self) -> dict[str, torch.Tensor]:
        return {
            name: projection.weight
            for name, projection in zip(self.names, self.projections)
        }
