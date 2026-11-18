import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

MODELS_DIR = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


models_pkg = types.ModuleType("models")
models_pkg.__path__ = [str(MODELS_DIR)]
sys.modules.setdefault("models", models_pkg)

openai_pkg = types.ModuleType("openai")
openai_pkg.OpenAI = object
sys.modules.setdefault("openai", openai_pkg)

groq_pkg = types.ModuleType("groq")
groq_pkg.Groq = object
sys.modules.setdefault("groq", groq_pkg)

load_module("models.base_llm", MODELS_DIR / "base_llm.py")
api_llm = load_module("models.api_llm", MODELS_DIR / "api_llm.py")
APIBasedLLM = api_llm.APIBasedLLM


def build_api_error(status_code, error_type, message):
    class FakeAPIError(Exception):
        __module__ = "openai._exceptions"

        def __init__(self):
            super().__init__(message)
            self.status_code = status_code
            self.body = {
                "error": {
                    "type": error_type,
                    "message": message,
                }
            }

    return FakeAPIError()


class FakeCompletions:
    def __init__(self, side_effects):
        self.side_effects = list(side_effects)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        side_effect = self.side_effects[min(self.calls - 1, len(self.side_effects) - 1)]
        raise side_effect


class FakeChat:
    def __init__(self, side_effects):
        self.completions = FakeCompletions(side_effects)


class FakeClient:
    def __init__(self, side_effects):
        self.chat = FakeChat(side_effects)


class APIBasedLLMTests(unittest.TestCase):
    @staticmethod
    def build_model(*side_effects):
        model = object.__new__(APIBasedLLM)
        model.provider = "openai"
        model.client = FakeClient(side_effects)
        model.model = "gpt-4o-mini"
        model.model_name = "gpt-4o-mini"
        model.temperature = 0.8
        model.max_tokens = 64
        model.top_p = 0.8
        model.repetition_penalty = 1.05
        model.use_cache = False
        model.model_loaded = True
        model.config = {}
        model.is_cache_loaded = True
        return model

    def test_content_policy_failures_return_empty_prediction(self):
        model = self.build_model(
            build_api_error(
                400,
                "content_policy_violation_error",
                "Content validation failed.",
            )
        )

        with mock.patch.object(api_llm.time, "sleep", return_value=None):
            response = model.inference({"query": "unsafe prompt"})

        self.assertEqual("", response["completion"])
        self.assertIsNone(response["prediction"])
        self.assertEqual(400, response["error"]["status_code"])
        self.assertEqual(
            "content_policy_violation_error",
            response["error"]["type"],
        )
        self.assertEqual(1, model.client.chat.completions.calls)

    def test_retryable_failures_are_retried_before_returning_empty_prediction(self):
        model = self.build_model(
            build_api_error(503, "server_error", "Service temporarily unavailable."),
        )

        with mock.patch.object(api_llm.time, "sleep", return_value=None) as sleep_mock:
            response = model.inference({"query": "retry me"})

        self.assertEqual("", response["completion"])
        self.assertIsNone(response["prediction"])
        self.assertEqual(503, response["error"]["status_code"])
        self.assertEqual(3, model.client.chat.completions.calls)
        self.assertEqual(2, sleep_mock.call_count)

    def test_invalid_requests_still_fail_fast(self):
        model = self.build_model(
            build_api_error(
                400,
                "invalid_request_error",
                "The configured model does not exist.",
            )
        )

        with self.assertRaises(RuntimeError):
            with mock.patch.object(api_llm.time, "sleep", return_value=None):
                model.inference({"query": "hello"})

        self.assertEqual(1, model.client.chat.completions.calls)


if __name__ == "__main__":
    unittest.main()
