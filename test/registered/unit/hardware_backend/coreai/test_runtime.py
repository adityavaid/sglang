import os
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCoreAIRuntimeGate(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, SGLANG_USE_CPU_ENGINE="0")
        env.start()
        self.addCleanup(env.stop)

    def test_disabled_backend_does_not_require_coreai(self):
        from sglang.srt.hardware_backend.coreai.runtime import use_coreai

        with patch.dict(os.environ, SGLANG_USE_COREAI="0", SGLANG_USE_MLX="0"):
            self.assertFalse(use_coreai())

    def test_backend_selection_is_exclusive(self):
        from sglang.srt.hardware_backend.coreai.runtime import use_coreai

        with patch.dict(os.environ, SGLANG_USE_COREAI="1", SGLANG_USE_MLX="1"):
            with self.assertRaisesRegex(ValueError, "MLX"):
                use_coreai()

    def test_cpu_model_execution_cannot_be_selected_with_coreai(self):
        from sglang.srt.hardware_backend.coreai.runtime import use_coreai

        with patch.dict(
            os.environ,
            SGLANG_USE_COREAI="1",
            SGLANG_USE_MLX="0",
            SGLANG_USE_CPU_ENGINE="1",
        ):
            with self.assertRaisesRegex(ValueError, "CPU"):
                use_coreai()

    def test_old_macos_cannot_silently_use_the_interpreter(self):
        from sglang.srt.hardware_backend.coreai.runtime import validate_runtime

        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch("platform.mac_ver", return_value=("26.6.2", (), "")),
        ):
            with self.assertRaisesRegex(RuntimeError, "macOS 27"):
                validate_runtime()

    def test_non_apple_host_is_rejected(self):
        from sglang.srt.hardware_backend.coreai.runtime import validate_runtime

        with patch("platform.system", return_value="Linux"):
            with self.assertRaisesRegex(RuntimeError, "Apple Silicon"):
                validate_runtime()


if __name__ == "__main__":
    unittest.main()
