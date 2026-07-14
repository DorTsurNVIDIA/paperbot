import io
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from agent import preflight


class PreflightTests(unittest.TestCase):
    def test_reports_models_when_configured_model_is_not_available(self):
        stderr = io.StringIO()
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai_compatible",
                "LLM_BASE_URL": "https://example.com/v1",
                "LLM_MODEL": "wanted-model",
                "LLM_API_KEY": "secret",
            },
            clear=True,
        ), patch.object(
            preflight, "_available_model_ids", return_value=["allowed-a", "allowed-b"]
        ), redirect_stderr(stderr):
            result = preflight.main()

        self.assertEqual(result, 1)
        self.assertIn("allowed-a", stderr.getvalue())
        self.assertNotIn("secret", stderr.getvalue())

    def test_passes_when_configured_model_is_available(self):
        stdout = io.StringIO()
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai_compatible",
                "LLM_BASE_URL": "https://example.com/v1",
                "LLM_MODEL": "allowed-model",
                "LLM_API_KEY": "secret",
            },
            clear=True,
        ), patch.object(
            preflight, "_available_model_ids", return_value=["allowed-model"]
        ), redirect_stdout(stdout):
            result = preflight.main()

        self.assertEqual(result, 0)
        self.assertIn("configured model is available", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
