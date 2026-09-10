# Experimental Core AI Qwen3 serving

Preparation uses SGLang's **standard model loader** to load the native PyTorch
`sglang.srt.models.qwen3.Qwen3ForCausalLM`, exports tensor-only prefill/decode
functions over that model's modules and parameters, and executes those functions
through Core AI under SGLang's existing worker/scheduler and HTTP/SSE endpoints.
It is not a new Torch device, a Core ML conversion, or an MPS/MLX fallback.

**This is a batch-one experimental profile, not full MLX parity or a
production-qualified backend.** It supports dense full-attention Qwen3 and
greedy generation. It does not support continuous batching, shared radix
prefixes, speculative decoding, LoRA, live weight updates, multimodal inputs,
logprobs, or constrained/non-greedy sampling. Unsupported requests/settings
are rejected rather than silently ignored.

## Serving environment and preparation

Use Python 3.13 (tested with 3.13.15). Python 3.14 does not currently have the
required Core AI wheels. This macOS profile installs the serving dependencies
without SGLang's Linux/CUDA kernels; MLX is not required. **Do not install
`coreai-models`**, whose Torch 2.9 pin conflicts with this branch.
SGLang's package metadata still lists its general GPU/media dependencies,
so pip may warn about those absent optional paths; this profile does not
claim that `pip check` passes for the complete CUDA-oriented distribution.

```bash
cd /path/to/sglang
# Only if the environment does not already exist:
uv venv --python 3.13 --seed .venv-coreai
.venv-coreai/bin/python -m pip install \
  -r experimental/coreai/requirements-serving.txt
SGLANG_BUILD_RUST_EXTS=none .venv-coreai/bin/python -m pip install \
  --no-deps -e python

unset SGLANG_USE_CPU_ENGINE
SGLANG_USE_COREAI=0 SGLANG_USE_MLX=0 \
.venv-coreai/bin/python -m sglang.srt.hardware_backend.coreai.prepare \
  --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --output-dir artifacts/qwen3-coreai-serving-delta \
  --context-length 2048 --prefill-chunk-size 64
```

Preparation can run with the bundled authoring/reference runtime on macOS 26.
**Serving requires Apple Silicon and the macOS 27 system Core AI runtime.**
The preparation process loads Torch weights on CPU and exits; the serving
process loads the compiled asset, not a second eager model. The bundle contains
`model.aimodel`, the tokenizer/config, and `coreai-manifest.json`. Existing
output directories are never overwritten.

The Python API accepts a standard-loaded **native SGLang** model, not a
Transformers `AutoModelForCausalLM`:

```python
from sglang.srt.hardware_backend.coreai.prepare import (
    load_native_qwen3,
    prepare_loaded_qwen3,
)

loaded_torch_qwen3 = load_native_qwen3(
    "Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
)

prepare_loaded_qwen3(
    loaded_torch_qwen3,
    "/new/path/qwen3-coreai",
    tokenizer=loaded_tokenizer,
    max_context_length=2048,
    prefill_chunk_size=64,
    source_model="Qwen/Qwen3-0.6B",
    source_revision="c1899de289a04d12100db370d81485cdf75e47ca",
)
```

The loaded model must already be on CPU in eval mode, using float16 or
float32 weights. Float32 is available for conversion/reference work; the
current SGLang serving profile requires float16. Preparation does not silently
move, dequantize, or cast the loaded model.

### Native loading and export boundary

`load_native_qwen3` constructs `ModelConfig(dtype="float16",
model_impl="sglang", trust_remote_code=False)`, `LoadConfig()` and
`DeviceConfig("cpu")`, then calls `sglang.srt.model_loader.get_model`.
This follows `get_model_loader` → `DefaultModelLoader.load_model` →
native construction and normal weight loading/postprocessing, including packed
QKV and gate/up mappings. There is no production HF model construction, weight
transposition into another model, or eager serving alternative.

Construction scopes SGLang configuration bags and single-rank Gloo groups.
It restores prior configuration/environment on success and failure, reuses
compatible existing single-rank groups, and destroys only groups it owns.
The native RoPE module cache is also scoped to the load, so prior models'
mutable/device-specific rotary buffers are neither reused nor changed.
It does not run server argument resolution or Linux CPU-engine kernels.
Preparation is an offline operation, not safe to run concurrently with another
runtime in the same process; incompatible distributed topologies are rejected.

The export adapter retains native model/parameter identity and calls native
RMSNorm, RoPE and activation tensor implementations. Dense packed projections
use their tensor linear operation without distributed/kernel dispatch.
Static causal SDPA with scatter-updated persistent KV replaces paged
`RadixAttention`/`ForwardBatch` scheduling; a last-token LM-head projection
replaces scheduler-dependent `LogitsProcessor`. The full scheduler-facing
model `forward` is therefore **not** the export boundary. No global native
kernel dispatch is patched.

This boundary supports only dense, unquantized, full-attention, default-RoPE
Qwen3 with TP=PP=1. It specializes token length, not position. Both compiled
entrypoints retain the same ABI: CPU int32 `input_ids[1,T]`,
`start_position[1]`, mutable floating KV and int32 `next_token[1]`, no ordinary
outputs. Tiny standard-loaded checkpoints are checked against independent
Transformers test oracles, strict Torch export and actual Core AI CPU-reference
execution. CPU-reference results are **not** macOS 27 native GPU qualification.

## Launch on macOS 27

```bash
BUNDLE=artifacts/qwen3-coreai-serving-delta
unset SGLANG_USE_CPU_ENGINE
SGLANG_USE_MLX=0 SGLANG_USE_COREAI=1 \
.venv-coreai/bin/python -m sglang.launch_server \
  --model-path "$BUNDLE" \
  --coreai-artifact-path "$BUNDLE" \
  --served-model-name qwen3-coreai \
  --host 127.0.0.1 --port 30000
```

Both paths must identify the same bundle, preventing tokenizer/config drift.
Torch CPU tensors provide scheduler bookkeeping only; Core AI owns the model
execution and persistent KV state. Startup specializes and warms the compiled
functions before readiness. Selecting Core AI on macOS 26 fails explicitly.
The separate Torch CPU engine flag is incompatible with Core AI; CPU
bookkeeping does not mean Torch CPU model execution.

```bash
curl --fail http://127.0.0.1:30000/health
curl --fail http://127.0.0.1:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-coreai","prompt":"The capital of France is","temperature":0,"max_tokens":16}'

curl -N http://127.0.0.1:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-coreai","prompt":"The capital of France is","temperature":0,"max_tokens":32,"stream":true}'
```

Queued requests execute serially. Each request starts at position zero; the
compiled causal mask hides the previous request's suffix, so logical reset
reuses physical state rather than copying KV to the host. An execution failure
poisons the session instead of retrying with partially updated cache or a
different backend.

Torch weight inspection/checks and memory-release/resume controls are rejected
before dispatch; they cannot operate on opaque Core AI resources.

## Runtime ownership and qualification

Both compiled functions write greedy argmax into an explicit persistent
`int32[1]` state named `next_token` and return **zero ordinary outputs**.
The host reads only this scalar after completion. This avoids requesting new
Core AI output NDArrays on every decode call, rather than relying on a
nonexistent Python `outputs=` API or periodically recycling workers.

With `coreai-core==1.0.0b2` on the tested macOS 27 runtime, a write-only
`next_token.copy_(token)` crashes native model loading in
`ReadHandleOpPattern::matchAndRewrite`. The exporter uses the equivalent
integer read-modify-write `next_token.add_(token - next_token)` to keep the
handle read live. Old bundles must be rebuilt into a **new directory**;
updating Python alone does not repair their compiled graph.

Local verification on 2026-09-10: the rebuilt Qwen3-0.6B bundle reached
SGLang readiness, `/health` returned 200, and `/v1/completions` returned
"Paris. The capital of Italy is Rome...". Streaming chat reached `[DONE]`;
repeating a prompt after another request reproduced its completion. The
zero-output native probe also completed 1,000 state-checked iterations.
Eight-token completions for short, chat-template, and 185-token prompts
matched an independently loaded Transformers Qwen3-0.6B Torch reference.
These are functional checks, not a production or sustained-memory claim.

Run the native regression alongside the reference/unit tests explicitly:

```bash
SGLANG_TEST_COREAI_NATIVE=1 .venv-coreai/bin/python -m pytest -q \
  test/registered/unit/hardware_backend/coreai experimental/coreai/tests
```

The opt-in test prepares a tiny native Qwen3, executes real Metal-backed
prefill/decode, and compares multiple requests against a Torch test oracle.

This is a candidate mitigation for the output-allocation ceiling reported in
[apple/coreai-torch#75](https://github.com/apple/coreai-torch/issues/75), not
proof that every macOS 27 runtime build is unaffected. Qualify the actual
target machine:

```bash
python -m experimental.coreai.qualify \
  --iterations 100000 --outputs 0 \
  --output-dir "$HOME/coreai-qualification/zero-output"
```

For the reported one/two/four-output controls, run the same command with
`--outputs 1`, `2`, and `4`, each in a fresh process and output directory.
The report records completed calls, exact versions, state correctness, and
sampled process RSS (not exclusive GPU/unified memory). On macOS 27 it also
retains raw specialization diagnostics through the pinned runtime's private
`AIModel._debug_infos` accessor.

A probe status of `passed` means call-count and state-correctness checks passed,
not that memory is bounded. Local macOS 26 reference runs completed 100,000
calls for all four output counts, but RSS grew even in the zero-output case.
Consequently they do not clear the sustained-memory gate or establish that
the macOS 27 allocation problem is fixed.

**GPU preference is not GPU-only enforcement.** Inspect placement evidence;
neither successful conversion nor the presence of a GPU preference establishes
all-GPU execution. The published 0.4.2 wheel does not include the newer source
tree's `ComputePlan` helper.

`--reference` is an explicit CPU/reference diagnostic option for this probe,
never a serving fallback. Local reference tests exercise real conversion and
Core AI execution, but do not qualify macOS 27 acceleration, Metal-backed
state access, sustained native memory behavior, or performance against MLX:

```bash
python -m pytest -q test/registered/unit/hardware_backend/coreai \
  experimental/coreai/tests
```

## Original PR export oracle

`poc.py` and the original `requirements.txt` remain isolated export-oracle
tools from the base PR. They do not enable the serving backend or establish
runtime/endurance qualification.

The experiment uses the upstream Core AI reauthored Qwen3 model rather than
trying to export SGLang's scheduler-aware forward directly.  That model emits
two graph entrypoints: `prefill` (KV writes only) and `main` (decode logits).
It is an export oracle, not a second SGLang model implementation.

### Legacy oracle environment

Run the exporter in a clean Python 3.11+ environment on Apple Silicon with
macOS 27 or later.  The pinned source revisions make an artifact reproducible;
they are intentionally not added to SGLang's base dependency set.

```bash
python -m venv .venv-coreai
.venv-coreai/bin/pip install -r experimental/coreai/requirements.txt
```

Before exporting, confirm that the system Core AI runtime is available:

```bash
.venv-coreai/bin/python experimental/coreai/poc.py probe
```

`probe` failing is a stop condition.  Do not interpret the bundled compatibility
runtime as evidence of GPU execution.

### Legacy oracle export

The default is FP16 so an MLX comparison can hold precision constant.  INT4 is
a separate deployment comparison, not a backend comparison.

```bash
.venv-coreai/bin/python experimental/coreai/poc.py export \
  --output-dir artifacts/coreai-qwen3-0.6b-fp16 \
  --max-context-length 2048
```

The command writes `artifact-manifest.json` next to the `.aimodel`.  It refuses
to overwrite an existing artifact unless `--overwrite` is specified.

## Benchmarking Core AI and MLX

Start two servers that expose the OpenAI-compatible `/v1/completions` endpoint,
one for each candidate.  The benchmark sends the *same deterministic prompt and
generation parameters* to both endpoints and stores raw request samples plus
percentile summaries.

```bash
.venv-coreai/bin/python experimental/coreai/benchmark_openai.py \
  --coreai-url http://127.0.0.1:30000 \
  --mlx-url http://127.0.0.1:30001 \
  --model Qwen/Qwen3-0.6B \
  --prompt-tokens 128 --output-tokens 128 \
  --concurrency 1 --requests 20 \
  --output artifacts/qwen3-0.6b-c1.json
```

The script measures complete-request latency and output throughput. Use
concurrency 1 for the current Core AI profile and identical precision/context
limits for both servers. Higher client concurrency only measures queueing, not
continuous-batching parity. Performance claims require target-machine placement
and endurance qualification first.
