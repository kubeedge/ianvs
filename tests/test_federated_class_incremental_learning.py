# Copyright 2022 The KubeEdge Authors.
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

"""Unit tests for FederatedClassIncrementalLearning paradigm."""

# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods,wrong-import-position,line-too-long,import-outside-toplevel,protected-access,unnecessary-dunder-call

import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.testcasecontroller.algorithm.paradigm.federated_learning.federated_class_incremental_learning import (
    FederatedClassIncrementalLearning,
)


class MockClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.received_helper_info = None

    def helper_function(self, helper_info):
        self.received_helper_info = helper_info


class MockAggregator:
    def helper_function(self, train_info):
        # Generate helper info explicitly tagged with the source client_id
        client_id = train_info.get("client_id")
        return {"source_client_id": client_id, "server_payload": f"payload_for_{client_id}"}


class TestFederatedClassIncrementalLearning(unittest.TestCase):
    """Test suite for FederatedClassIncrementalLearning fixes."""

    def test_init_without_model_eval(self):
        """FCIL __init__ should gracefully handle missing or empty model_eval without KeyError."""
        paradigm = object.__new__(FederatedClassIncrementalLearning)
        # Mock minimal attributes from ParadigmBase
        paradigm.modules = {}
        paradigm.workspace = "/tmp/workspace"
        paradigm.clients_number = 2
        paradigm.fl_data_setting = {}
        paradigm.rounds = 1

        # Test empty kwargs
        FederatedClassIncrementalLearning.__init__(paradigm, "/tmp/workspace", modules={})
        self.assertIsNone(paradigm.accuracy_func)
        self.assertEqual(paradigm.train_infos, {})

    def test_init_with_empty_model_metric_name(self):
        """FCIL __init__ should gracefully handle default model_eval with empty name."""
        paradigm = object.__new__(FederatedClassIncrementalLearning)
        kwargs = {
            "modules": {},
            "model_eval": {
                "model_metric": {
                    "mode": "",
                    "name": "",
                    "url": "",
                }
            },
        }
        FederatedClassIncrementalLearning.__init__(paradigm, "/tmp/workspace", **kwargs)
        self.assertIsNone(paradigm.accuracy_func)

    def test_helper_function_routing_out_of_order_list(self):
        """helper_function must route helper state to the correct client even if train_infos is out of order."""
        paradigm = object.__new__(FederatedClassIncrementalLearning)
        paradigm.clients_number = 3
        paradigm.clients = [MockClient(i) for i in range(3)]
        paradigm.aggregator = MockAggregator()

        # Simulate out-of-order completion list: client 2 finished first, then 0, then 1
        out_of_order_train_infos = [
            {"client_id": 2, "num_samples": 50},
            {"client_id": 0, "num_samples": 100},
            {"client_id": 1, "num_samples": 80},
        ]

        paradigm.helper_function(out_of_order_train_infos)

        for i in range(3):
            received = paradigm.clients[i].received_helper_info
            self.assertIsNotNone(received)
            self.assertEqual(
                received["source_client_id"],
                i,
                f"Client {i} received helper info from client {received['source_client_id']}",
            )

    def test_helper_function_routing_dict(self):
        """helper_function must route helper state correctly when train_infos is a dict."""
        paradigm = object.__new__(FederatedClassIncrementalLearning)
        paradigm.clients_number = 3
        paradigm.clients = [MockClient(i) for i in range(3)]
        paradigm.aggregator = MockAggregator()

        train_infos_dict = {
            0: {"client_id": 0, "num_samples": 100},
            1: {"client_id": 1, "num_samples": 80},
            2: {"client_id": 2, "num_samples": 50},
        }

        paradigm.helper_function(train_infos_dict)

        for i in range(3):
            received = paradigm.clients[i].received_helper_info
            self.assertIsNotNone(received)
            self.assertEqual(received["source_client_id"], i)

    def test_concurrent_client_train_population(self):
        """Concurrent client_train execution must populate train_infos without race condition."""
        paradigm = object.__new__(FederatedClassIncrementalLearning)
        paradigm.clients_number = 4
        paradigm.clients = [MockClient(i) for i in range(4)]
        paradigm.lock = threading.RLock()
        paradigm.train_infos = {}

        def mock_super_client_train(client_idx, *_args, **_kwargs):
            # Introduce non-deterministic timing
            delay = 0.05 * (4 - client_idx)
            time.sleep(delay)
            return {"client_id": client_idx, "samples": 100 * client_idx}

        # Override super().client_train call via mocking
        class DummySuper:
            def client_train(self, client_idx, *args, **kwargs):
                return mock_super_client_train(client_idx, *args, **kwargs)

        with patch(
            "core.testcasecontroller.algorithm.paradigm.federated_learning.federated_class_incremental_learning.super",
            return_value=DummySuper(),
        ):
            threads = [
                threading.Thread(target=paradigm.client_train, args=(i, None, None))
                for i in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertEqual(len(paradigm.train_infos), 4)
        for i in range(4):
            self.assertIn(i, paradigm.train_infos)
            self.assertEqual(paradigm.train_infos[i]["client_id"], i)


    def test_helper_function_incomplete_train_infos_raises(self):
        """helper_function must raise RuntimeError if any client train_info is missing."""
        paradigm = object.__new__(FederatedClassIncrementalLearning)
        paradigm.clients_number = 3
        paradigm.clients = [MockClient(i) for i in range(3)]
        paradigm.aggregator = MockAggregator()

        # Incomplete map missing client 1
        incomplete_train_infos = {
            0: {"client_id": 0, "num_samples": 100},
            2: {"client_id": 2, "num_samples": 50},
        }

        with self.assertRaises(RuntimeError) as ctx:
            paradigm.helper_function(incomplete_train_infos)
        self.assertIn("missing client(s) [1]", str(ctx.exception))

    def test_helper_function_unexpected_train_infos_raises(self):
        """helper_function must raise RuntimeError if unexpected client IDs are passed."""
        paradigm = object.__new__(FederatedClassIncrementalLearning)
        paradigm.clients_number = 2
        paradigm.clients = [MockClient(0), MockClient(1)]
        paradigm.aggregator = MockAggregator()

        # Unexpected client 5
        unexpected_train_infos = {
            0: {"client_id": 0, "num_samples": 100},
            1: {"client_id": 1, "num_samples": 80},
            5: {"client_id": 5, "num_samples": 50},
        }

        with self.assertRaises(RuntimeError) as ctx:
            paradigm.helper_function(unexpected_train_infos)
        self.assertIn("unexpected client(s) [5]", str(ctx.exception))

    def test_train_worker_failure_raises_runtime_error(self):
        """FederatedLearning.train must capture and re-raise worker exceptions."""
        from core.testcasecontroller.algorithm.paradigm.federated_learning.federated_learning import (
            FederatedLearning,
        )

        paradigm = object.__new__(FederatedLearning)
        paradigm.clients_number = 2
        paradigm.clients = [MockClient(0), MockClient(1)]
        paradigm.lock = threading.RLock()
        paradigm.aggregate_clients = []

        def failing_client_train(client_idx, *_args, **_kwargs):
            if client_idx == 1:
                raise ValueError("Client 1 OOM during training")

        paradigm.client_train = failing_client_train

        with self.assertRaises(RuntimeError) as ctx:
            paradigm.train([None, None])
        self.assertIn("client training failed for client(s)", str(ctx.exception))
        self.assertIn("Client 1 OOM during training", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
