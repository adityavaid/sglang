# Running SGLang Core AI on this Mac

## Machine and current limits

Recorded on 2026-09-16:

| Item | Local configuration |
| --- | --- |
| Hardware | Apple M4 Pro, 48 GB unified memory, arm64 |
| Operating system | macOS 26.6.2, build 25G83 |
| Repository | `/Users/aditshar/personal/sglang-core-ai` |
| Branch | `feat/core-ai-serving` |
| Python environment | `/Users/aditshar/personal/sglang-fork diff/.venv-mlx-dev` |
| Supported model family | Dense Qwen3, initially `Qwen/Qwen3-0.6B` |

**On this machine's current macOS version, you can prepare quantized assets and
run Core AI CPU-reference tests. Native Core AI serving requires macOS 27.**
Torch MPS availability does not establish Core AI runtime availability.
Do not bypass the serving version check or interpret CPU-reference timing as GPU
performance. This implementation does not support Qwen3.5-0.8B.

## 1. Activate the existing environment

Run this in each terminal used for preparation or serving:

```bash
cd /Users/aditshar/personal/sglang-core-ai
source "/Users/aditshar/personal/sglang-fork diff/.venv-mlx-dev/bin/activate"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
unset SGLANG_USE_CPU_ENGINE SGLANG_PLATFORM
export SGLANG_USE_MLX=0
export SGLANG_USE_COREAI=0
```

The quoted activation path is intentional: the directory name contains a space.
The FFmpeg library path lets TorchCodec find the Homebrew FFmpeg 7 libraries.
Use this MLX development environment, not `.venv-coreai`: the latter retains
Torch 2.13 and does not have the optional compressor installed.

Check the interpreter and dependencies:

```bash
python - <<'PY'
import sys
from importlib.metadata import version
import torch
import torchcodec
import coreai_torch
import coreai_opt

print("Python:", sys.executable)
for package in (
    "torch", "torchvision", "torchaudio", "torchcodec",
    "coreai-core", "coreai-torch", "coreai-opt", "torchao",
):
    print(f"{package}: {version(package)}")
print("Torch MPS available:", torch.backends.mps.is_available())
PY
```

The prepared environment uses Torch 2.11.0, torchvision 0.26.0, torchaudio
2.11.0, TorchCodec 0.11.1, coreai-core 1.0.0b2, coreai-torch 0.4.2,
coreai-opt 0.2.1 and torchao 0.17.0.

The Core AI export dependencies are already installed. The rebase updates the
requirements files but does not reinstall packages in this shared environment.
The expanded upstream serving profile contains additional packages and pins.
To inspect the synchronization plan without changing the environment:

```bash
uv pip install --dry-run --python "$VIRTUAL_ENV/bin/python" \
  -r experimental/coreai/requirements-quantization.txt
```

Only if provisioning that complete profile or restoring missing dependencies in
a compatible SGLang environment, use:

```bash
uv pip install --python "$VIRTUAL_ENV/bin/python" \
  -r experimental/coreai/requirements-quantization.txt \
  "torchvision==0.26.0" "torchaudio==2.11.0" "torchcodec==0.11.1" \
  "setuptools==81.0.0"
```

Do not install `coreai-models` for this path or mix these pins with SGLang extras
requiring a different torchao version. The existing shared environment has
unrelated dependency conflicts; this is not a full environment repair command.
Warnings from coremltools about its tested Torch/scikit-learn versions are
distinct from failures to import or execute the Core AI compressor.

## 2. Export dense Qwen3 with INT4 weights

This downloads the pinned floating-point checkpoint and tokenizer if not cached.
SGLang's normal PyTorch loader loads the model on CPU, then the original-forward
export path creates compressed Core AI prefill and decode functions.

```bash
python -m sglang.srt.hardware_backend.coreai.prepare \
  --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --output-dir "$HOME/models/qwen3-0.6b-coreai-int4" \
  --context-length 2048 \
  --prefill-chunk-size 64 \
  --forward-path native \
  --weight-quantization int4
```

Choose a new output path on each export; existing directories are rejected, not
overwritten. For INT8, change both the output directory suffix and
`--weight-quantization` to `int8`. For a floating-point comparison, use `none`
and a separate output directory. Keep context and prefill sizes equal when
comparing these bundles.

Rebuild any bundles exported before this rebase. PR #1 fixes a macOS 27 native
load crash by replacing a write-only token-state update with an integer
read-modify-write. This fix is retained for both quantized and floating exports;
updating Python alone cannot repair an older compiled asset.

The quantizer uses clipped-symmetric, linear block-32 weights, not palettization.
It compresses packed QKV, attention output, packed gate/up and down projections.
Embeddings, the LM head, biases, norms, activations and KV remain floating-point.
It does not mutate the caller's original Torch parameters.

Preparation still holds the original floating model and temporary export data:
INT4 bundle size is not a preparation peak-memory estimate. The serving process
loads the compiled asset without retaining that eager model.

After successful export, inspect the bundle:

```bash
ls "$HOME/models/qwen3-0.6b-coreai-int4"
python -m json.tool "$HOME/models/qwen3-0.6b-coreai-int4/coreai-manifest.json"
du -sh "$HOME/models/qwen3-0.6b-coreai-int4/model.aimodel"
```

Expect `model.aimodel/`, `config.json`, tokenizer files and
`coreai-manifest.json`. The manifest should say `weight_quantization: "int4"`
and `dtype: "float16"`: the latter describes compute and KV, not compressed
weight storage. Both entrypoints share persistent KV and token state.

## 3. Exercise reference inference on macOS 26

The following runs tiny local checkpoints through actual export and Core AI's
CPU-reference runtime. It does not download the full Qwen3 checkpoint or start
an HTTP server:

```bash
python -m pytest -q \
  test/registered/unit/hardware_backend/coreai/test_prepare.py
```

This covers both quantization modes and forward paths, prefill, advancing decode
positions, request reuse, graph bit widths, serialized asset size ordering and
preservation of original weights. Quantized runtime tokens are compared with a
precision-matched eager reference; separate logit checks compare with floating
weights. Tiny-model results do not establish full-checkpoint generation quality.

For the full Core AI backend test set, isolate each file in its own process:

```bash
for test_file in test/registered/unit/hardware_backend/coreai/test_*.py; do
  python -m pytest -q "$test_file" || exit
done
```

## 4. Serve the compiled bundle on macOS 27 only

**Stop here on macOS 26.6.2.** The following commands are for a Mac with the
macOS 27 system Core AI runtime and the compatible environment above.
Recheck dependencies and qualify the asset on that machine; exporting an asset
on macOS 26 does not prove GPU placement or performance on macOS 27.

After running the activation block from section 1:

```bash
BUNDLE="$HOME/models/qwen3-0.6b-coreai-int4"
SGLANG_USE_MLX=0 SGLANG_USE_COREAI=1 \
python -m sglang.launch_server \
  --model-path "$BUNDLE" \
  --coreai-artifact-path "$BUNDLE" \
  --served-model-name qwen3-coreai \
  --host 127.0.0.1 \
  --port 30000
```

Leave this command running in the foreground. Both model-path arguments must
point to the same bundle. Wait for startup and warmup to finish.

From another terminal, send a greedy streaming request:

```bash
curl --fail-with-body -N http://127.0.0.1:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-coreai",
    "prompt": "The capital of France is",
    "temperature": 0,
    "max_tokens": 32,
    "stream": true
  }'
```

Stop the foreground server with Ctrl-C.

This is batch-one, serial-request, greedy serving. Continuous batching, shared
radix prefixes, speculative decoding, LoRA, embeddings and non-greedy sampling
are not supported. The host still reads and resubmits each generated token;
GPU-resident pipelined token feedback is not implemented. No native tok/s or
speedup over MLX has been established for this M4 Pro.

For placement and sustained-memory qualification, see
[experimental/coreai/README.md](experimental/coreai/README.md#runtime-ownership-and-qualification).

## 5. What the accompanying patch contains

`coreai-macos-diff.txt` contains the remaining local changes and new Core AI
files, including this guide, after rebasing onto PR #1's fetched head:

```text
8ae55d9eb40c06e9336b2b55201842bf69fb3ef7
Add tabulate for one-batch benchmark
```

It includes the original-forward wrapper, INT4/INT8 preparation, compatible
dependency pins, documentation and tests. The serving implementation and native
load fix already present in PR #1 are part of the base, not duplicated additions.
It excludes the previous `coreai-unstaged-vs-pr1.txt` and `coreai-macos.patch`,
the diff itself, virtual environments, downloaded models and session logs.
Environment installations are not captured by a Git patch.

**Do not apply this patch to the current working directory: it already contains
these changes.** To apply it elsewhere, use a clean checkout with the base commit
available, adjusting the patch path if it was copied to another machine:

```bash
cd /path/to/clean/sglang-checkout
git switch -c coreai-macos-import 8ae55d9eb40c06e9336b2b55201842bf69fb3ef7
git apply --check /Users/aditshar/personal/sglang-core-ai/coreai-macos-diff.txt
git apply /Users/aditshar/personal/sglang-core-ai/coreai-macos-diff.txt
```

The machine-specific paths in section 1 must be adjusted for a different checkout
or user account.

## 6. macOS 27 validation on a second Mac (2026-09-16)

The instructions above record an M4 Pro running macOS 26.6.2. A separate Apple
M5 with 16 GB memory, macOS 27.0 (build 26A5425a), and Python 3.13.15
completed the native serving steps after this branch was rebased onto
`sgl-project/sglang` main at `e7f7447333`. Its `.venv-coreai` has Torch 2.13.0,
`coreai-core` 1.0.0b2, `coreai-torch` 0.4.2, and `coreai-opt` 0.2.1. These
results are specific to that environment; the macOS 26 version gate remains.

The native-forward INT4 export used the section 2 command with
`--output-dir artifacts/qwen3-coreai-native-int4-20260916`. Its `model.aimodel`
was 544 MB. The section 4 server command, pointed at that bundle, reached
readiness and answered `/v1/completions` with a continuation beginning
` Paris.` The earlier floating-point adapter bundle was also served and
answered ` Paris. The capital of Italy is Rome.`

Using `sglang.benchmark.one_batch_server` with batch size 1, 128 input
tokens, and 128 output tokens, one warmed run per configuration produced:

| Backend and weight format | Latency (s) | Input tok/s | Output tok/s | TTFT (s) | ITL (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core AI floating FP16 | 2.85 | 1536.72 | 46.34 | 0.08 | 21.58 |
| MLX default BF16 | 1.58 | 563.54 | 94.35 | 0.23 | 10.60 |
| Core AI native-forward INT4 | 2.07 | 2298.86 | 63.67 | 0.06 | 15.71 |
| MLX `mlx_q4` | 0.55 | 1473.54 | 273.99 | 0.09 | 3.65 |

MLX `mlx_q4` uses a group size of 64; the Core AI INT4 export uses block 32.
Their outputs and weight formats differ, so the quantized rows measure two
serving configurations rather than identical numerical models. These are
single-run measurements, not a confidence interval. Native Core AI compilation
and HTTP serving succeeded, but a GPU preference alone does not prove every
operation ran on GPU or establish sustained memory behavior.
