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

"""Check that prepare_dataset.py output keeps what the benchmark needs.

Runs prepare_dataset.py on a few fixture rows and loads the result with the
same Sedna parser Ianvs uses for test_data_info, so no model or GPU is needed.

    python -m pytest examples/government/singletask_learning_bench/test_data_contract.py
"""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("sedna")
from sedna.datasources import JSONMetaDataParse  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "subjective" / "testenv"))

import prepare_dataset  # noqa: E402
from llm_judgement import extract_comprehensive_score  # noqa: E402

OBJECTIVE_ROW = {
    "question": "Which body approves the annual budget?",
    "options": ["The court", "The people's congress", "The police", "A bank"],
    "answer": "B",
}
SUBJECTIVE_METADATA = {
    "dataset": "fixture",
    "description": "fixture",
    "level_1_dim": "single-modal",
    "level_2_dim": "text",
}
JUDGE_PROMPT = "Score the answer from 1 to 10 for accuracy and give an Overall Score.\nAnswer: "


def _source(tmp_path, subjective_rows):
    source = tmp_path / "government"
    objective = source / "multi-choice questions"
    subjective = source / "subjective questions"
    objective.mkdir(parents=True)
    subjective.mkdir(parents=True)
    (objective / "data.jsonl").write_text(json.dumps(OBJECTIVE_ROW) + "\n", encoding="utf-8")
    (subjective / "data.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in subjective_rows), encoding="utf-8"
    )
    (subjective / "metadata.json").write_text(json.dumps(SUBJECTIVE_METADATA), encoding="utf-8")
    return source


def _prepare(tmp_path, subjective_rows):
    output = tmp_path / "out"
    prepare_dataset._prepare_objective(_source(tmp_path, subjective_rows), output)
    prepare_dataset._prepare_subjective(tmp_path / "government", output)
    return output


def _load(metadata_path):
    data = JSONMetaDataParse(data_type="test")
    data.parse(str(metadata_path))
    return data


def test_objective_options_reach_the_prompt(tmp_path):
    output = _prepare(tmp_path, [{"query": "q", "response": "r", "judge_prompt": JUDGE_PROMPT}])

    data = _load(output / "objective" / "test_data" / "metadata.json")

    assert len(data.x) == 1
    assert OBJECTIVE_ROW["question"] in data.x[0]
    for label, option in zip("ABCD", OBJECTIVE_ROW["options"]):
        assert f"{label}. {option}" in data.x[0]
    assert data.y[0] == "B"


@pytest.mark.parametrize(
    "row",
    [
        {"query": "How do I renew a permit?", "response": "At the office.",
         "judge_prompt": JUDGE_PROMPT, "level_3_dim": "Q&A", "level_4_dim": "permits"},
        {"question": "How do I renew a permit?", "answer": "At the office.",
         "judge_prompt": JUDGE_PROMPT},
    ],
)
def test_subjective_judge_prompt_and_dimensions_survive_loading(tmp_path, row):
    output = _prepare(tmp_path, [row])

    data = _load(output / "subjective" / "test_data" / "metadata.json")

    assert list(data.x) == ["How do I renew a permit?"]
    assert list(data.y) == ["At the office."]
    assert list(data.judge_prompts) == [JUDGE_PROMPT]
    assert data.level_3[0] and data.level_4[0]


def test_subjective_row_without_a_judge_prompt_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="judge_prompt"):
        _prepare(tmp_path, [{"query": "q", "response": "r"}])


@pytest.mark.parametrize(
    ("judgement", "score"),
    [
        ("Overall Score: 7", 7),
        ("overall score: 7/10", 7),
        ('{"Overall Points": 8}', 8),
        ('{"综合得分": 6}', 6),
        ("综合得分：9", 9),
        ("no score here", None),
    ],
)
def test_judge_scores_are_read_in_english_and_chinese(judgement, score):
    assert extract_comprehensive_score(judgement) == score
