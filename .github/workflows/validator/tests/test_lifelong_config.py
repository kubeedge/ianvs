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

"""Tests for module ①, the Lifelong Config Guard in static_validator.py."""

import textwrap

import static_validator as sv

LIFELONG_ALGORITHM = """
algorithm:
  paradigm_type: "lifelonglearning"
  lifelong_learning_data_setting:
    train_ratio: {ratio}
    splitting_method: "{method}"
"""

TESTENV = """
testenv:
  dataset:
{dataset}
  incremental_rounds: {rounds}
"""


def _write(root, relpath, text):
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def _example(tmp_path, dataset='    train_index: "a.txt"\n    test_index: "b.txt"',
             rounds=2, ratio=0.8, method="default", lifelong=True):
    _write(tmp_path, "ex/testenv/testenv.yaml", TESTENV.format(dataset=dataset, rounds=rounds))
    if lifelong:
        _write(tmp_path, "ex/testalgorithms/algo.yaml",
               LIFELONG_ALGORITHM.format(ratio=ratio, method=method))
    else:
        _write(tmp_path, "ex/testalgorithms/algo.yaml",
               'algorithm:\n  paradigm_type: "singletasklearning"\n')
    report = sv.ExampleReport(name="ex", path="ex")
    sv._check_lifelong_config(report, tmp_path, sv._example_files(tmp_path / "ex"))  # pylint: disable=protected-access
    return {check.name: check for check in report.checks}


def test_modern_keys_pass(tmp_path):
    checks = _example(tmp_path)

    assert checks["Legacy dataset keys"].status == sv.PASS
    assert checks["Lifelong learning configuration"].status == sv.PASS


def test_legacy_keys_warn_and_name_the_replacement(tmp_path):
    checks = _example(tmp_path, dataset='    train_url: "a.txt"\n    test_url: "b.txt"')

    check = checks["Legacy dataset keys"]
    assert check.status == sv.WARNING
    assert len(check.details) == 2
    assert any("train_url" in d and "train_index" in d for d in check.details)
    assert all("testenv/testenv.yaml" in d for d in check.details)


def test_legacy_keys_are_checked_for_every_paradigm(tmp_path):
    checks = _example(tmp_path, dataset='    train_url: "a.txt"', lifelong=False)

    assert checks["Legacy dataset keys"].status == sv.WARNING
    assert "Lifelong learning configuration" not in checks


def test_too_few_rounds_is_an_error(tmp_path):
    check = _example(tmp_path, rounds=1)["Lifelong learning configuration"]

    assert check.status == sv.ERROR
    assert any("incremental_rounds" in d for d in check.details)


def test_non_integer_rounds_is_an_error(tmp_path):
    check = _example(tmp_path, rounds='"2"')["Lifelong learning configuration"]

    assert check.status == sv.ERROR


def test_unknown_splitting_method_is_an_error(tmp_path):
    check = _example(tmp_path, method="random_split")["Lifelong learning configuration"]

    assert check.status == sv.ERROR
    assert any("random_split" in d for d in check.details)


def test_train_ratio_out_of_range_is_an_error(tmp_path):
    for ratio in (0, 1, 1.5, '"0.8"'):
        check = _example(tmp_path, ratio=ratio)["Lifelong learning configuration"]
        assert check.status == sv.ERROR, ratio


def test_all_supported_splitting_methods_pass(tmp_path):
    for method in ("default", "city_splitting", "fwt_splitting", "hard-example_splitting"):
        check = _example(tmp_path, method=method)["Lifelong learning configuration"]
        assert check.status == sv.PASS, method


def test_lifelong_check_skipped_for_other_paradigms(tmp_path):
    checks = _example(tmp_path, lifelong=False)

    assert "Lifelong learning configuration" not in checks
