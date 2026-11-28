# Copyright 2024 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Cloud-Edge Collaborative Inference for LLM example."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_EXAMPLE_DIR = REPO_ROOT / "examples" / "cloud-edge-collaborative-inference-for-llm"
QUERY_ROUTING_DIR = LLM_EXAMPLE_DIR / "testalgorithms" / "query-routing"


class TestCloudEdgeLLMRequirements(unittest.TestCase):
    """Test dependency specifications in requirements.txt."""

    def test_retry_in_requirements(self):
        req_file = LLM_EXAMPLE_DIR / "requirements.txt"
        self.assertTrue(req_file.exists(), "requirements.txt must exist")
        content = req_file.read_text(encoding="utf-8")
        deps = [
            line.strip().split("==")[0].split(">=")[0]
            for line in content.splitlines()
            if line.strip()
        ]
        self.assertIn("retry", deps, "retry must be listed in requirements.txt")


class TestEdgeModelModule(unittest.TestCase):
    """Test EdgeModel backend loading and error handling."""

    @classmethod
    def setUpClass(cls):
        # Mock external dependencies not installed in basic test environment
        cls._mocked_modules = {}
        for mod_name in [
            "sedna",
            "sedna.common",
            "sedna.common.class_factory",
            "retry",
            "groq",
            "openai",
            "vllm",
            "vllm.distributed",
            "vllm.distributed.parallel_state",
            "transformers",
            "torch",
        ]:
            if mod_name not in sys.modules:
                mock_mod = MagicMock()
                cls._mocked_modules[mod_name] = mock_mod
                sys.modules[mod_name] = mock_mod

        # Ensure @ClassFactory.register acts as pass-through decorator
        sedna_cf = sys.modules.get("sedna.common.class_factory")
        if sedna_cf and hasattr(sedna_cf, "ClassFactory"):
            sedna_cf.ClassFactory.register.return_value = lambda c: c

        if str(QUERY_ROUTING_DIR) not in sys.path:
            sys.path.insert(0, str(QUERY_ROUTING_DIR))

    @classmethod
    def tearDownClass(cls):
        for mod_name in cls._mocked_modules:
            sys.modules.pop(mod_name, None)
        if str(QUERY_ROUTING_DIR) in sys.path:
            sys.path.remove(str(QUERY_ROUTING_DIR))

    def test_import_edge_model_succeeds_without_ladespecdec(self):
        """edge_model should import cleanly even if LadeSpecDecLLM is missing."""
        import edge_model
        self.assertTrue(hasattr(edge_model, "EdgeModel"))
        self.assertIsNone(edge_model.LadeSpecDecLLM)

    def test_edge_model_supported_backends(self):
        import edge_model
        for backend in ["huggingface", "vllm", "api", "EagleSpecDec", "LadeSpecDec"]:
            with patch.object(edge_model.EdgeModel, "_set_config"):
                model = edge_model.EdgeModel(backend=backend, model="dummy")
                self.assertEqual(model.backend, backend)

    def test_edge_model_unsupported_backend_raises_value_error(self):
        import edge_model
        with self.assertRaises(ValueError) as ctx:
            edge_model.EdgeModel(backend="unsupported_backend", model="dummy")
        self.assertIn("Unsupported backend", str(ctx.exception))

    def test_edge_model_lade_spec_dec_raises_not_implemented(self):
        """When LadeSpecDec backend is loaded but class is unavailable, raise RuntimeError."""
        import edge_model
        with patch.object(edge_model.EdgeModel, "_set_config"):
            model = edge_model.EdgeModel(backend="LadeSpecDec", model="dummy")
            with self.assertRaises(RuntimeError) as ctx:
                model.load()
            self.assertIn("Model loading failed for backend `LadeSpecDec`", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
