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

"""Tests for module ③, the Lifelong Smoke Harness (services/lifelong_smoke.py)."""

import yaml

from services import lifelong_smoke as ls


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _benchmark(tmp_path, paradigm="lifelonglearning", rounds=2, ratio=0.8, mode=None,
               dataset=None):
    metric = {"name": "accuracy"}
    if mode:
        metric["mode"] = mode
    _write_yaml(tmp_path / "ex/testenv.yaml", {"testenv": {
        "dataset": dataset or {"train_url": "/data/train.txt", "test_url": "/data/test.txt"},
        "model_eval": {"model_metric": metric}, "incremental_rounds": rounds}})
    _write_yaml(tmp_path / "ex/algo.yaml", {"algorithm": {
        "paradigm_type": paradigm,
        "lifelong_learning_data_setting": {"train_ratio": ratio, "splitting_method": "default"}}})
    _write_yaml(tmp_path / "ex/benchmarkingjob.yaml", {"benchmarkingjob": {
        "name": "benchmarkingjob", "workspace": "./workspace", "testenv": "./ex/testenv.yaml",
        "test_object": {"type": "algorithms", "algorithms": [{"name": "a", "url": "./ex/algo.yaml"}]}}})
    return ls.load_lifelong_settings(tmp_path, "ex/benchmarkingjob.yaml")


def _index(path, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"./images/{i}.png ./images/{i}_y.png\n" for i in range(n)), encoding="utf-8")
    return path


def _round_dirs(output, *names, unseen_round=None):
    for name in names:
        (output / name).mkdir(parents=True, exist_ok=True)
    if unseen_round:
        unseen = output / "inference" / "unseen_task_samples" / str(unseen_round)
        unseen.mkdir(parents=True, exist_ok=True)
        (unseen / "sample.png").write_bytes(b"")


def _workspace(tmp_path, rank_rows=1):
    workspace = tmp_path / "workspace"
    output = workspace / "benchmarkingjob" / "algo" / "case-id" / "output"
    rank = workspace / "benchmarkingjob" / "rank" / "all_rank.csv"
    rank.parent.mkdir(parents=True)
    rank.write_text("rank,algorithm\n" + "1,algo\n" * rank_rows, encoding="utf-8")
    return workspace, output


def test_settings_read_from_benchmark(tmp_path):
    settings = _benchmark(tmp_path, rounds=3, ratio=0.9, mode="no-inference")

    assert (settings.rounds, settings.train_ratio, settings.mode) == (3, 0.9, "no-inference")


def test_non_lifelong_benchmark_gives_none(tmp_path):
    assert _benchmark(tmp_path, paradigm="singletasklearning") is None


def test_min_samples_per_round():
    assert ls.min_samples_per_round(0.8) == 2
    assert ls.min_samples_per_round(0.95) == 2
    assert ls.min_samples_per_round(0.5) == 2
    assert ls.min_samples_per_round(0.3) == 4


def test_sizing_passes_at_the_minimum_and_fails_below(tmp_path):
    settings = _benchmark(tmp_path, rounds=2, ratio=0.8)

    assert ls.check_sizing(_index(tmp_path / "ok.txt", 4), settings).status == "PASS"
    assert ls.check_sizing(_index(tmp_path / "small.txt", 3), settings).status == "FAIL"
    assert ls.check_sizing(tmp_path / "missing.txt", settings).status == "FAIL"


def test_materialize_keeps_key_names_and_uses_smoke_root(tmp_path):
    settings = _benchmark(tmp_path)
    temp = tmp_path / "tmp"
    temp.mkdir()

    benchmark_file, train_index, workspace = ls.materialize_benchmark(settings, tmp_path / "smoke", temp)

    job = yaml.safe_load(open(benchmark_file))["benchmarkingjob"]
    dataset = yaml.safe_load(open(job["testenv"]))["testenv"]["dataset"]
    assert dataset == {"train_url": str(tmp_path / "smoke/train_data/index.txt"),
                       "test_url": str(tmp_path / "smoke/test_data/index.txt")}
    assert train_index == tmp_path / "smoke/train_data/index.txt"
    assert job["workspace"] == str(workspace) == str(temp / "workspace")


def test_materialize_with_modern_keys(tmp_path):
    settings = _benchmark(tmp_path, dataset={"train_index": "/a.txt", "test_index": "/b.txt"})
    temp = tmp_path / "tmp"
    temp.mkdir()

    benchmark_file, _, _ = ls.materialize_benchmark(settings, tmp_path / "smoke", temp)

    job = yaml.safe_load(open(benchmark_file))["benchmarkingjob"]
    dataset = yaml.safe_load(open(job["testenv"]))["testenv"]["dataset"]
    assert set(dataset) == {"train_index", "test_index"}


def test_complete_rounds_pass(tmp_path):
    settings = _benchmark(tmp_path, rounds=2)
    workspace, output = _workspace(tmp_path)
    _round_dirs(output, "train/1", "eval/1", "inference/results/2", "train/2", "eval/2", unseen_round=2)

    assert ls.check_round_completion(workspace, settings).status == "PASS"


def test_round_without_unseen_samples_need_not_retrain(tmp_path):
    settings = _benchmark(tmp_path, rounds=2)
    workspace, output = _workspace(tmp_path)
    _round_dirs(output, "train/1", "eval/1", "inference/results/2")

    assert ls.check_round_completion(workspace, settings).status == "PASS"


def test_missing_round_outputs_fail_with_details(tmp_path):
    settings = _benchmark(tmp_path, rounds=3)
    workspace, output = _workspace(tmp_path)
    _round_dirs(output, "train/1", "inference/results/2", unseen_round=2)

    outcome = ls.check_round_completion(workspace, settings)

    assert outcome.status == "FAIL"
    assert "round 1: output/eval/1 is missing" in outcome.details
    assert "round 2: output/train/2 is missing" in outcome.details
    assert "round 3: output/inference/results/3 is missing" in outcome.details


def test_missing_rank_rows_fail(tmp_path):
    settings = _benchmark(tmp_path, rounds=2)
    workspace, output = _workspace(tmp_path, rank_rows=0)
    _round_dirs(output, "train/1", "eval/1", "inference/results/2")

    outcome = ls.check_round_completion(workspace, settings)

    assert outcome.status == "FAIL"
    assert any("all_rank.csv" in detail for detail in outcome.details)


def test_no_output_at_all_fails(tmp_path):
    settings = _benchmark(tmp_path)

    assert ls.check_round_completion(tmp_path / "empty", settings).status == "FAIL"


def test_other_modes_are_skipped(tmp_path):
    settings = _benchmark(tmp_path, mode="no-inference")

    assert ls.check_round_completion(tmp_path, settings).status == "SKIP"
