# Core AI Qwen3 proof of concept

This directory is deliberately outside SGLang's serving backend selection.  It
answers the first two gates in `core-ai-implementation-plan.md` without making
an experimental runtime responsible for scheduler state or paged KV ownership.

The experiment uses the upstream Core AI reauthored Qwen3 model rather than
trying to export SGLang's scheduler-aware forward directly.  That model emits
two graph entrypoints: `prefill` (KV writes only) and `main` (decode logits).
It is an export oracle, not a second SGLang model implementation.

## Environment

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

## Export Qwen3-0.6B

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

The script measures complete-request latency and output throughput.  Run the
matrix from the proposal (concurrency 1/4/8/16, prompt 128/2K/8K, output
128/512) in interleaved order, with a fresh output file for each point.  This
benchmark is intentionally not a claim of Core AI performance until its server
uses native-owned persistent KV state and has passed the runtime/endurance gate.
