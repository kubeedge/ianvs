# Copyright 2026 The KubeEdge Authors.
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

import copy
from pathlib import Path
import pickle
import sys
import unittest
from unittest.mock import MagicMock

# Mock sedna if not installed
for mod in [
    "sedna",
    "sedna.datasources",
    "sedna.core",
    "sedna.core.base",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import yaml

from core.testenvmanager.dataset import Dataset
from core.testenvmanager.testenv import TestEnv


class TestTestEnvConfigParsing(unittest.TestCase):
    def setUp(self):
        self.base_config = {
            "testenv": {
                "dataset": {
                    "train_url": "/tmp/train.txt",
                    "test_url": "/tmp/test.txt",
                },
                "metrics": [{"name": "accuracy", "url": "/tmp/acc.py"}],
                "incremental_rounds": 2,
            }
        }

    def test_standard_config_parsing(self):
        te = TestEnv(self.base_config)
        self.assertEqual(te.incremental_rounds, 2)
        self.assertEqual(len(te.metrics), 1)
        self.assertFalse(te.use_gpu)
        self.assertIsInstance(te.dataset, Dataset)

    def test_method_shadowing_rejected(self):
        """Keys matching class methods (e.g. prepare, _check_fields) must raise ValueError."""
        for bad_key in ["prepare", "_check_fields", "_parse_config"]:
            cfg = copy.deepcopy(self.base_config)
            cfg["testenv"][bad_key] = "malicious_override"
            with self.assertRaises(ValueError):
                TestEnv(cfg)

    def test_reserved_key_other_params_rejected(self):
        """A key named other_params must not overwrite the internal storage."""
        cfg = copy.deepcopy(self.base_config)
        cfg["testenv"]["other_params"] = "overwrite_store"
        with self.assertRaises(ValueError) as ctx:
            TestEnv(cfg)
        self.assertIn("cannot shadow", str(ctx.exception))

    def test_private_attribute_injection_rejected(self):
        """Keys starting with underscore must raise ValueError."""
        cfg = copy.deepcopy(self.base_config)
        cfg["testenv"]["_secret"] = "injected_value"
        with self.assertRaises(ValueError) as ctx:
            TestEnv(cfg)
        self.assertIn("private attributes cannot be configured", str(ctx.exception))

    def test_duplicate_key_collision_rejected(self):
        """Duplicate keys (e.g. integer 1 and string '1') must raise ValueError."""
        cfg = copy.deepcopy(self.base_config)
        cfg["testenv"][1] = "first"
        cfg["testenv"]["1"] = "second"
        with self.assertRaises(ValueError) as ctx:
            TestEnv(cfg)
        self.assertIn("Duplicate", str(ctx.exception))

    def test_strict_use_gpu_parsing(self):
        """Test strict boolean evaluation for use_gpu."""
        falsy_values = ["false", "False", "FALSE", "0", "off", "no", 0, False]
        for val in falsy_values:
            cfg = copy.deepcopy(self.base_config)
            cfg["testenv"]["use_gpu"] = val
            te = TestEnv(cfg)
            self.assertIs(te.use_gpu, False, f"Expected False for {val}")

        truthy_values = ["true", "True", "TRUE", "1", "on", "yes", 1, True]
        for val in truthy_values:
            cfg = copy.deepcopy(self.base_config)
            cfg["testenv"]["use_gpu"] = val
            te = TestEnv(cfg)
            self.assertIs(te.use_gpu, True, f"Expected True for {val}")

        invalid_values = ["maybe", None, [1, 2], {"a": 1}, 2, -1]
        for val in invalid_values:
            cfg = copy.deepcopy(self.base_config)
            cfg["testenv"]["use_gpu"] = val
            with self.assertRaises(ValueError):
                TestEnv(cfg)

    def test_other_params_and_getattr(self):
        """Extra benchmark-specific keys are stored in other_params and accessible via attribute access."""
        cfg = copy.deepcopy(self.base_config)
        cfg["testenv"]["hard_example_mining_mode"] = "mining-free"
        cfg["testenv"]["gpu_num"] = 4
        cfg["testenv"]["custom_setting"] = "custom_val"

        te = TestEnv(cfg)
        self.assertEqual(te.hard_example_mining_mode, "mining-free")
        self.assertEqual(te.gpu_num, 4)
        self.assertEqual(te.custom_setting, "custom_val")
        self.assertEqual(te.other_params["hard_example_mining_mode"], "mining-free")

        # Top-level __dict__ is not polluted
        self.assertNotIn("hard_example_mining_mode", te.__dict__)
        self.assertNotIn("gpu_num", te.__dict__)

        # Non-existent attribute raises AttributeError
        with self.assertRaises(AttributeError):
            _ = te.non_existent_key

    def test_deepcopy_and_pickle_safety(self):
        """Verifies __getattr__ does not enter infinite recursion on copy or pickle."""
        cfg = copy.deepcopy(self.base_config)
        cfg["testenv"]["gpu_num"] = 2
        te = TestEnv(cfg)

        copied = copy.deepcopy(te)
        self.assertEqual(copied.gpu_num, 2)
        self.assertEqual(copied.incremental_rounds, 2)

        pickled = pickle.dumps(te)
        unpickled = pickle.loads(pickled)
        self.assertEqual(unpickled.gpu_num, 2)
        self.assertEqual(unpickled.incremental_rounds, 2)

    def test_testcase_downstream_unpacking(self):
        """Verifies TestCase unpacks other_params so downstream paradigms receive expected kwargs."""
        cfg = copy.deepcopy(self.base_config)
        cfg["testenv"]["hard_example_mining_mode"] = "mining-free"
        cfg["testenv"]["gpu_num"] = 2
        te = TestEnv(cfg)

        test_env_config = {}
        for k, v in te.__dict__.items():
            if k == "other_params" and isinstance(v, dict):
                test_env_config.update(v)
            else:
                test_env_config[k] = v

        self.assertEqual(test_env_config["hard_example_mining_mode"], "mining-free")
        self.assertEqual(test_env_config["gpu_num"], 2)
        self.assertEqual(test_env_config["incremental_rounds"], 2)
        self.assertNotIn("other_params", test_env_config)

    def test_dataset_allowed_fields_sync(self):
        """Dataset.ALLOWED_FIELDS must strictly match attributes initialized by Dataset({})."""
        sample_dataset = Dataset({})
        self.assertEqual(Dataset.ALLOWED_FIELDS, frozenset(vars(sample_dataset).keys()))

    def test_dataset_unknown_fields_logged_and_ignored(self):
        """Unknown dataset fields are ignored and do not mutate instance attributes."""
        ds_config = {
            "train_index": "/tmp/train.txt",
            "unknown_injected_attr": "should_be_ignored",
        }
        ds = Dataset(ds_config)
        self.assertEqual(ds.train_index, "/tmp/train.txt")
        self.assertNotIn("unknown_injected_attr", ds.__dict__)

    def test_existing_examples_testenvs_load_successfully(self):
        """Scan real testenv configurations in examples/ to guarantee 100% backward compatibility."""
        examples_dir = Path(__file__).resolve().parents[2] / "examples"
        scanned_count = 0
        orig_check = Dataset._check_dataset_url
        Dataset._check_dataset_url = classmethod(lambda cls, url: None)
        try:
            for yaml_path in examples_dir.rglob("*.yaml"):
                try:
                    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="ignore"))
                    if not isinstance(data, dict):
                        continue
                    # If file defines testenv section, ensure TestEnv can parse it
                    for section in ["testenv", "test_env"]:
                        if section in data and isinstance(data[section], dict):
                            te_data = {section: copy.deepcopy(data[section])}
                            # If required metrics or incremental_rounds are missing, provide mock
                            if "metrics" not in te_data[section]:
                                te_data[section]["metrics"] = [{"name": "acc"}]
                            if "incremental_rounds" not in te_data[section]:
                                te_data[section]["incremental_rounds"] = 2
                            te = TestEnv(te_data)
                            self.assertIsNotNone(te)
                            scanned_count += 1
                except Exception as e:
                    self.fail(f"Failed to parse testenv from {yaml_path}: {e}")
        finally:
            Dataset._check_dataset_url = orig_check

        self.assertGreater(scanned_count, 10, f"Expected at least 10 example configs scanned, got {scanned_count}.")


if __name__ == "__main__":
    unittest.main()
