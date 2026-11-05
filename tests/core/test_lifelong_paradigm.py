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

"""Pins S3: the initial-training flag set by LifelongLearning._train
(core/testcasecontroller/algorithm/paradigm/lifelong_learning/lifelong_learning.py).

Sedna builds the knowledge base when HAS_COMPLETED_INITIAL_TRAINING is
'False' and updates an existing one when it is 'True'. The default mode trains
first in round 1; no-inference and hard-example-mining train first in round 0.

Preservation tests pass on the unmodified code and must still pass after S3.
Change tests fail on the unmodified code and pass after S3.
"""

import os

import pytest

from core.testcasecontroller.algorithm.paradigm.lifelong_learning.lifelong_learning import (
    LifelongLearning,
)


class _RecordingJob:
    """Stands in for Sedna's LifelongLearning; records the flag at train time."""

    def __init__(self, seen):
        self.seen = seen

    cloud_task_index = "cloud/index.pkl"

    def train(self, train_dataset):
        self.seen.append(os.environ["HAS_COMPLETED_INITIAL_TRAINING"])
        # Sedna 0.6 train() returns the per-task model results, not the index.
        return {"real_model": "checkpoint.pth"}


def _flag_for(tmp_path, monkeypatch, mode, rounds):
    monkeypatch.delenv("HAS_COMPLETED_INITIAL_TRAINING", raising=False)
    paradigm = LifelongLearning.__new__(LifelongLearning)
    paradigm.workspace = str(tmp_path)
    metric = {"name": "accuracy"}
    if mode is not None:
        metric["mode"] = mode
    paradigm.model_eval_config = {"model_metric": metric}
    seen = []
    paradigm.build_paradigm_job = lambda paradigm_type: _RecordingJob(seen)

    paradigm._train("cloud_index.pkl", object(), rounds)  # pylint: disable=protected-access

    return seen[0]


# Preservation tests


@pytest.mark.parametrize("mode, rounds, expected", [
    (None, 2, "True"),
    (None, 3, "True"),
    ("no-inference", 0, "False"),
    ("no-inference", 1, "True"),
    ("no-inference", 2, "True"),
    ("hard-example-mining", 0, "False"),
    ("hard-example-mining", 1, "True"),
])
def test_flag_unchanged_where_already_correct(tmp_path, monkeypatch, mode, rounds, expected):
    assert _flag_for(tmp_path, monkeypatch, mode, rounds) == expected


# Change tests


def test_default_mode_first_round_is_initial_training(tmp_path, monkeypatch):
    assert _flag_for(tmp_path, monkeypatch, None, 1) == "False"


class _InferenceJob:
    """Stands in for IanvsLifelongLearning in _inference() and _eval()."""

    cloud_task_index = "cloud/index.pkl"

    def __init__(self):
        self.calls = []

    def inference_lifelong(self, data, **kwargs):
        self.calls.append(kwargs)
        return ["pred"], False, []

    def evaluate(self, data, **kwargs):
        return {"accuracy": 0.5}


class _Dataset:
    def load_data(self, *args, **kwargs):
        data = type("Data", (), {})()
        data.x, data.y = [["a.png"], ["b.png"]], [["a_y.png"], ["b_y.png"]]
        return data


def _paradigm_with(tmp_path, job, mode=None):
    paradigm = LifelongLearning.__new__(LifelongLearning)
    paradigm.workspace = str(tmp_path)
    metric = {"name": "accuracy"}
    if mode is not None:
        metric["mode"] = mode
    paradigm.model_eval_config = {"model_metric": metric, "threshold": 0, "operator": "<"}
    paradigm.dataset = _Dataset()
    paradigm.build_paradigm_job = lambda paradigm_type: job
    return paradigm


def test_inference_uses_inference_lifelong(tmp_path):
    job = _InferenceJob()
    paradigm = _paradigm_with(tmp_path, job)

    results, unseen = paradigm._inference("edge/index.pkl", "index.txt", 2)  # pylint: disable=protected-access

    assert results == [["pred"], ["pred"]]
    assert len(unseen.x) == 0
    assert job.calls == [{}, {}]


def test_eval_returns_the_cloud_task_index(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.testcasecontroller.algorithm.paradigm.lifelong_learning."
        "lifelong_learning.get_metric_func", lambda metric: ("accuracy", None))
    paradigm = _paradigm_with(tmp_path, _InferenceJob())

    index = paradigm._eval("cloud/index.pkl", "index.txt", 1)  # pylint: disable=protected-access

    assert index == "cloud/index.pkl"


def test_train_returns_the_cloud_task_index(tmp_path):
    paradigm = LifelongLearning.__new__(LifelongLearning)
    paradigm.workspace = str(tmp_path)
    paradigm.model_eval_config = {"model_metric": {"name": "accuracy"}}
    paradigm.build_paradigm_job = lambda paradigm_type: _RecordingJob([])

    index = paradigm._train("previous/index.pkl", object(), 1)  # pylint: disable=protected-access

    assert index == "cloud/index.pkl"
