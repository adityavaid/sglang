import unittest

from sglang.srt.hardware_backend.coreai.export import CoreAIExportSpec
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCoreAIExportSpec(unittest.TestCase):
    def test_rejects_ambiguous_examples(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            CoreAIExportSpec(
                entrypoint_name="main",
                input_names=("x",),
                output_names=("y",),
            )

    def test_state_must_not_be_an_input(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            CoreAIExportSpec(
                entrypoint_name="main",
                input_names=("x", "cache"),
                output_names=("y",),
                example_args=(object(),),
                state_names=("cache",),
            )

    def test_valid_tensor_contract(self):
        spec = CoreAIExportSpec(
            entrypoint_name="decode",
            input_names=("token_ids",),
            output_names=("logits",),
            example_args=(object(), object()),
            state_names=("cache",),
        )
        self.assertEqual(spec.entrypoint_name, "decode")


if __name__ == "__main__":
    unittest.main()
