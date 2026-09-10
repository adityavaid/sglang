import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class TestCoreAIQualification(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("coreai_torch"), "Core AI extra is not installed"
    )
    def test_zero_output_probe_executes_the_real_state_updates(self):
        from experimental.coreai.qualify import run_probe

        with tempfile.TemporaryDirectory() as directory:
            report = run_probe(
                Path(directory) / "probe", iterations=3, output_count=0, reference=True
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["completed_iterations"], 3)
            self.assertEqual(report["final_counter"], 3)
            self.assertEqual(report["ordinary_outputs_per_call"], 0)
            self.assertIn("reference", report["execution_profile"])

    @unittest.skipUnless(
        importlib.util.find_spec("coreai_torch"), "Core AI extra is not installed"
    )
    def test_intermediate_token_corruption_stops_the_probe(self):
        from experimental.coreai.qualify import _CounterStep, run_probe

        class CorruptSecondToken(_CounterStep):
            def forward(self, x):
                outputs = super().forward(x)
                self.next_token.copy_(
                    torch.where(
                        self.counter == 2,
                        -torch.ones_like(self.next_token),
                        self.next_token,
                    )
                )
                return outputs

        with tempfile.TemporaryDirectory() as directory:
            with patch("experimental.coreai.qualify._CounterStep", CorruptSecondToken):
                with self.assertRaisesRegex(RuntimeError, "state updates"):
                    run_probe(
                        Path(directory) / "probe",
                        iterations=3,
                        output_count=0,
                        reference=True,
                    )

    def test_invalid_run_is_rejected_without_writing(self):
        from experimental.coreai.qualify import run_probe

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe"
            with self.assertRaises(ValueError):
                run_probe(path, iterations=0, output_count=0, reference=True)
            self.assertFalse(path.exists())
