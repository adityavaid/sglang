"""Exercise real Core AI state/output allocation without a serving fallback."""

import argparse
import asyncio
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from sglang.srt.hardware_backend.coreai.export import (
    CoreAIExportSpec,
    export_loaded_torch_model,
)
from sglang.srt.hardware_backend.coreai.runtime import validate_runtime


class _CounterStep(torch.nn.Module):
    def __init__(self, output_count: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(32) / 1024, requires_grad=False)
        self.register_buffer("counter", torch.zeros(1))
        self.register_buffer("next_token", torch.zeros(1, dtype=torch.int32))
        self.output_count = output_count

    def forward(self, x):
        self.counter.add_((x @ self.weight).sum())
        self.next_token.copy_(self.counter.to(torch.int32))
        return tuple(self.counter + float(i) for i in range(self.output_count))


def run_probe(
    output_dir: Path, *, iterations: int, output_count: int = 0, reference: bool = False
) -> dict:
    if type(iterations) is not int or not 1 <= iterations <= 16_000_000:
        raise ValueError("iterations must be between 1 and 16000000.")
    if type(output_count) is not int or output_count not in (0, 1, 2, 4):
        raise ValueError("output_count must be 0, 1, 2, or 4.")
    if not reference:
        validate_runtime()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    asset = export_loaded_torch_model(
        _CounterStep(output_count).eval(),
        CoreAIExportSpec(
            entrypoint_name="step",
            input_names=("x",),
            output_names=tuple(f"output_{i}" for i in range(output_count)),
            example_args=(torch.ones(32, 32),),
            state_names=("counter", "next_token"),
        ),
        output_dir / "state-probe.aimodel",
        source_model="stateful-matmul-allocation-probe",
    )
    report = {
        "status": "failed",
        "requested_iterations": iterations,
        "completed_iterations": 0,
        "ordinary_outputs_per_call": output_count,
        "token_readback_per_call": True,
        "preparation_seconds": time.perf_counter() - started,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "coreai-core", "coreai-torch")
        },
        "memory_scope": "sampled process RSS, not exclusive GPU/unified memory",
        "bounded_memory_verified": False,
        "gpu_placement_verified": False,
    }

    async def execute():
        from coreai.runtime import (
            AIModel,
            ComputeUnitKind,
            NDArray,
            SpecializationOptions,
            StorageKind,
        )

        if reference:
            supported = SpecializationOptions.is_supported()
            options = SpecializationOptions.cpu_only() if supported else None
            report["execution_profile"] = (
                "os-cpu-reference" if supported else "bundled-reference"
            )
        else:
            options = SpecializationOptions.from_preferred_compute_unit_kind(
                ComputeUnitKind.gpu()
            ).with_debug(enabled=True)
            report["execution_profile"] = "os-runtime-gpu-preferred"
        model = await AIModel.load(asset, specialization_options=options)
        if not reference:
            # Pinned 1.0.0b2 diagnostic API; GPU preference alone is not placement proof.
            (output_dir / "compute-debug.json").write_bytes(model._debug_infos)
            report["placement_evidence"] = "compute-debug.json (requires inspection)"
        function = model.load_function("step")
        if len(function.desc.output_names) != output_count:
            raise RuntimeError("Compiled output ABI differs from the requested probe.")
        backing = StorageKind.BYTES if reference else StorageKind.METAL
        inputs = {"x": NDArray(np.ones((32, 32), dtype=np.float32), backing=backing)}
        state = {
            "counter": NDArray(np.zeros(1, dtype=np.float32), backing=backing),
            "next_token": NDArray(np.zeros(1, dtype=np.int32), backing=backing),
        }
        process = psutil.Process()
        samples = [{"iteration": 0, "rss_bytes": process.memory_info().rss}]
        report["rss_samples"] = samples
        started = time.perf_counter()
        for i in range(iterations):
            outputs = await function(inputs=inputs, state=state)
            if len(outputs) != output_count:
                raise RuntimeError("Runtime output count changed during the probe.")
            del outputs
            report["completed_iterations"] = i + 1
            report["final_counter"] = int(state["next_token"].numpy().item())
            if report["final_counter"] != i + 1:
                raise RuntimeError("Persistent state updates were lost or reordered.")
            if (i + 1) % 1000 == 0:
                samples.append(
                    {"iteration": i + 1, "rss_bytes": process.memory_info().rss}
                )
        samples.append(
            {"iteration": iterations, "rss_bytes": process.memory_info().rss}
        )
        report["execution_seconds"] = time.perf_counter() - started
        report["max_sampled_rss_bytes"] = max(sample["rss_bytes"] for sample in samples)
        report["status"] = "passed"

    try:
        asyncio.run(execute())
    finally:
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--outputs", type=int, choices=(0, 1, 2, 4), default=0)
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Explicit CPU/reference diagnosis only; never GPU qualification.",
    )
    args = parser.parse_args()
    report = run_probe(
        args.output_dir,
        iterations=args.iterations,
        output_count=args.outputs,
        reference=args.reference,
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "rss_samples"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
