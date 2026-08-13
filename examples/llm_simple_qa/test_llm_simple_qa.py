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

import os
import yaml
import pytest


def test_llm_simple_qa_config_paths():
    base_dir = "examples/llm_simple_qa"
    bench_job_path = os.path.join(base_dir, "benchmarkingjob.yaml")
    
    assert os.path.exists(bench_job_path)
    with open(bench_job_path, "r", encoding="utf-8") as f:
        bench_job = yaml.safe_load(f)

    job_cfg = bench_job["benchmarkingjob"]
    testenv_path = job_cfg["testenv"].lstrip("./")
    assert os.path.exists(testenv_path), f"Testenv path {testenv_path} does not exist"

    algo_url = job_cfg["test_object"]["algorithms"][0]["url"].lstrip("./")
    assert os.path.exists(algo_url), f"Algorithm URL {algo_url} does not exist"

    with open(testenv_path, "r", encoding="utf-8") as f:
        testenv_cfg = yaml.safe_load(f)["testenv"]

    train_data = testenv_cfg["dataset"]["train_data"].lstrip("./")
    test_data = testenv_cfg["dataset"]["test_data"].lstrip("./")
    metric_url = testenv_cfg["metrics"][0]["url"].lstrip("./")

    assert os.path.exists(train_data), f"Train dataset {train_data} does not exist"
    assert os.path.exists(test_data), f"Test dataset {test_data} does not exist"
    assert os.path.exists(metric_url), f"Metric URL {metric_url} does not exist"

    with open(algo_url, "r", encoding="utf-8") as f:
        algo_cfg = yaml.safe_load(f)["algorithm"]

    module_url = algo_cfg["modules"][0]["url"].lstrip("./")
    assert os.path.exists(module_url), f"Module URL {module_url} does not exist"


def test_get_last_letter_and_acc():
    from examples.llm_simple_qa.testenv.acc import acc, get_last_letter

    assert get_last_letter("Answer is A") == "A"
    assert get_last_letter("Choice B.") == "B"
    assert get_last_letter("none") is None

    assert acc(["A", "C", "B"], ["The answer is A", "C", "D"]) == 2.0 / 3.0
    assert acc([], []) == 0.0
