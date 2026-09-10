#!/usr/bin/env python3
"""Fair, dependency-free comparison of two OpenAI-compatible completion servers."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sample:
    backend: str
    latency_s: float
    output_tokens: int
    error: str | None = None


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _request(url: str, payload: dict[str, Any], timeout_s: float, backend: str) -> Sample:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            result = json.loads(response.read())
        usage = result.get("usage", {})
        tokens = int(usage.get("completion_tokens", payload["max_tokens"]))
        return Sample(backend=backend, latency_s=time.perf_counter() - start, output_tokens=tokens)
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
        return Sample(
            backend=backend,
            latency_s=time.perf_counter() - start,
            output_tokens=0,
            error=str(exc),
        )


def benchmark_backend(
    backend: str,
    url: str,
    payload: dict[str, Any],
    requests: int,
    concurrency: int,
    timeout_s: float,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        samples = list(
            executor.map(
                lambda _: _request(url, payload, timeout_s, backend), range(requests)
            )
        )
    wall_time_s = time.perf_counter() - started_at
    successful = [sample for sample in samples if sample.error is None]
    latencies = [sample.latency_s for sample in successful]
    return {
        "backend": backend,
        "requests": requests,
        "successful_requests": len(successful),
        "errors": [sample.error for sample in samples if sample.error],
        "latency_ms": {
            "p50": _percentile(latencies, 0.50) * 1000,
            "p90": _percentile(latencies, 0.90) * 1000,
            "p99": _percentile(latencies, 0.99) * 1000,
            "mean": statistics.fmean(latencies) * 1000 if latencies else 0.0,
        },
        "aggregate_output_tokens_per_second": (
            sum(sample.output_tokens for sample in successful) / wall_time_s
            if wall_time_s
            else 0.0
        ),
        "wall_time_s": wall_time_s,
        "samples": [asdict(sample) for sample in samples],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coreai-url", required=True)
    parser.add_argument("--mlx-url", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.output_tokens, args.concurrency, args.requests) < 1:
        parser.error("token counts, concurrency, and requests must be positive.")

    # A whitespace-separated synthetic prompt avoids tokenizer-specific text
    # content while still giving both servers an identical request body.
    prompt = " ".join(["benchmark"] * args.prompt_tokens)
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "stream": False,
    }
    started_at = time.time()
    results = {
        "schema_version": 1,
        "started_at_unix_s": started_at,
        "request": payload,
        "concurrency": args.concurrency,
        "requests_per_backend": args.requests,
        "results": [
            benchmark_backend(
                "coreai",
                args.coreai_url,
                payload,
                args.requests,
                args.concurrency,
                args.timeout_s,
            ),
            benchmark_backend(
                "mlx",
                args.mlx_url,
                payload,
                args.requests,
                args.concurrency,
                args.timeout_s,
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
