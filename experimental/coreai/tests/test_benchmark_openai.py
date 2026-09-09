import unittest
from unittest import mock

from experimental.coreai.benchmark_openai import Sample, _percentile, benchmark_backend


class TestBenchmarkOpenAI(unittest.TestCase):
    def test_percentile_interpolates_between_samples(self):
        self.assertEqual(_percentile([1.0, 3.0], 0.5), 2.0)

    def test_benchmark_keeps_failed_requests_out_of_latency_and_throughput(self):
        samples = iter(
            [
                Sample(backend="coreai", latency_s=0.1, output_tokens=5),
                Sample(
                    backend="coreai", latency_s=0.2, output_tokens=0, error="failed"
                ),
            ]
        )
        with mock.patch(
            "experimental.coreai.benchmark_openai._request",
            side_effect=lambda *_: next(samples),
        ):
            result = benchmark_backend(
                "coreai", "http://unused", {"max_tokens": 5}, 2, 1, 1.0
            )

        self.assertEqual(result["successful_requests"], 1)
        self.assertEqual(result["latency_ms"]["p50"], 100.0)
        self.assertGreater(result["aggregate_output_tokens_per_second"], 0)
