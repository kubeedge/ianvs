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

"""Tests for IanvsLifelongLearning
(core/testcasecontroller/algorithm/paradigm/sedna_lifelong_learning.py)."""

import itertools
from types import SimpleNamespace

import pytest
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.datasources import BaseDataSource

from core.testcasecontroller.algorithm.paradigm.sedna_lifelong_learning import (
    IanvsLifelongLearning,
)

_COUNTER = itertools.count()


class _SeenEstimator:
    def __init__(self):
        self.calls = []
        self.estimator = SimpleNamespace(base_model=object())

    def predict(self, data, post_process=None, task_index=None, task_type=None, **kwargs):
        self.calls.append({"task_index": task_index, "task_type": task_type, "kwargs": kwargs})
        return [f"pred-{x[0]}" for x in data.x], ["task"]


def _register_recognizer(call):
    """Register a UTD recognizer whose __call__ is `call`; return its name."""
    name = f"IanvsTestRecognizer{next(_COUNTER)}"

    def __init__(self, task_index, **kwargs):
        self.task_index = task_index

    cls = type(name, (), {"__init__": __init__, "__call__": call})
    ClassFactory.register_cls(cls, ClassType.UTD, alias=name)
    return name


def _all_unseen(self, samples):
    seen, unseen = BaseDataSource(data_type="test"), BaseDataSource(data_type="test")
    seen.x, unseen.x = [], samples.x
    return seen, unseen


def _first_unseen_with_kwargs(self, samples, **kwargs):
    self.kwargs = kwargs
    seen, unseen = BaseDataSource(data_type="test"), BaseDataSource(data_type="test")
    seen.x, unseen.x = samples.x[1:], samples.x[:1]
    return seen, unseen


def _needs_scores(self, samples, scores, **kwargs):
    raise AssertionError("must not be called")


@pytest.fixture(name="job")
def fixture_job(tmp_path, monkeypatch):
    index = tmp_path / "index.pkl"
    index.write_bytes(b"")
    monkeypatch.setenv("MODEL_URLS", str(index))
    job = IanvsLifelongLearning.__new__(IanvsLifelongLearning)
    job.cloud_knowledge_management = SimpleNamespace(
        seen_estimator=_SeenEstimator(), task_index=str(tmp_path / "cloud" / "index.pkl"))
    job.unseen_sample_recognition_param = {}
    job._unseen_recognizer = None  # pylint: disable=protected-access
    job.index_path = str(index)
    return job


def _data(*names):
    data = BaseDataSource(data_type="test")
    data.x = [[name, f"{name}-depth"] for name in names]
    return data


def test_seen_only_skips_detection(job):
    job.unseen_sample_recognition = {"method": _register_recognizer(_needs_scores)}

    res, is_unseen, tasks = job.inference_lifelong(_data("a", "b"), detect_unseen=False)

    assert res == ["pred-a", "pred-b"]
    assert is_unseen is False
    assert tasks == ["task"]
    assert job._unseen_recognizer is None  # pylint: disable=protected-access


def test_uses_the_index_from_model_urls(job):
    job.unseen_sample_recognition = {"method": _register_recognizer(_all_unseen)}

    job.inference_lifelong(_data("a"))

    call = job.cloud_knowledge_management.seen_estimator.calls[0]
    assert call["task_index"] == job.index_path
    assert call["task_type"] == "seen_task"


def test_single_unseen_sample_gives_bool(job):
    job.unseen_sample_recognition = {"method": _register_recognizer(_all_unseen)}

    res, is_unseen, _ = job.inference_lifelong(_data("a"))

    assert res == ["pred-a"]
    assert is_unseen is True


def test_batch_gives_dict_and_passes_kwargs(job):
    job.unseen_sample_recognition = {"method": _register_recognizer(_first_unseen_with_kwargs)}

    _, is_unseen, _ = job.inference_lifelong(_data("a", "b", "c"), mode="multi")

    assert is_unseen == {"a": True, "b": False, "c": False}
    assert job._unseen_recognizer.kwargs == {"mode": "multi"}  # pylint: disable=protected-access


def test_kwargs_not_passed_to_recognizer_without_kwargs(job):
    job.unseen_sample_recognition = {"method": _register_recognizer(_all_unseen)}

    _, is_unseen, _ = job.inference_lifelong(_data("a"), mode="multi")

    assert is_unseen is True


def test_score_based_recognizer_is_rejected_clearly(job):
    job.unseen_sample_recognition = {"method": _register_recognizer(_needs_scores)}

    with pytest.raises(NotImplementedError, match="seen-task scores"):
        job.inference_lifelong(_data("a"))


def test_cloud_task_index(job):
    assert job.cloud_task_index == job.cloud_knowledge_management.task_index


def test_update_strategy_default_is_registered():
    assert ClassFactory.is_exists(ClassType.KM, "UpdateStrategyDefault")


def test_update_strategy_default_retrains_each_task_group():
    task_a, task_b = SimpleNamespace(entry="a"), SimpleNamespace(entry="b")
    index = {
        "seen_task": {"task_groups": [SimpleNamespace(tasks=[task_a]), None,
                                      SimpleNamespace(tasks=[task_b])]},
        "unseen_task": {"task_groups": []},
    }
    samples = object()

    strategy = ClassFactory.get_cls(ClassType.KM, "UpdateStrategyDefault")(index)
    tasks, strategies = strategy(samples, task_type="seen_task")

    assert tasks == [task_a, task_b]
    assert all(task.samples is samples for task in tasks)
    assert strategies["a"]["target_model_update"] is True
    assert strategy(samples, task_type="unseen_task") == ([], {})


def _capture_prefix_during_update(monkeypatch):
    from sedna.common.config import BaseConfig
    from sedna.core.lifelong_learning import LifelongLearning
    seen = []

    def fake_update(self, train_data, valid_data=None, post_process=None, **kwargs):
        seen.append(BaseConfig.data_path_prefix)
        return "updated"

    monkeypatch.setattr(LifelongLearning, "update", fake_update)
    return seen, BaseConfig


def test_update_ignores_data_path_prefix_for_absolute_local_index(tmp_path, monkeypatch):
    seen, base_config = _capture_prefix_during_update(monkeypatch)
    index = tmp_path / "index.pkl"
    index.write_bytes(b"")
    monkeypatch.setenv("CLOUD_KB_INDEX", str(index))
    before = base_config.data_path_prefix
    job = IanvsLifelongLearning.__new__(IanvsLifelongLearning)

    assert job.update("train-data") == "updated"
    assert seen == ["/"]
    assert base_config.data_path_prefix == before


def test_update_keeps_data_path_prefix_for_other_indexes(monkeypatch):
    seen, base_config = _capture_prefix_during_update(monkeypatch)
    monkeypatch.setenv("CLOUD_KB_INDEX", "relative/index.pkl")
    job = IanvsLifelongLearning.__new__(IanvsLifelongLearning)

    job.update("train-data")

    assert seen == [base_config.data_path_prefix]
