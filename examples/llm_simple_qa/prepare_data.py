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

"""Generate the simple-qa dataset used by the llm_simple_qa benchmark.

The Sedna ``JsonlDataParse`` parser expects exactly one valid JSON object per
line. This script writes the dataset in that format so users do not have to
hand-format the pretty-printed objects shown in the README.

It creates the following layout, relative to this file::

    examples/llm_simple_qa/dataset/
    ├── test_data
    │   └── data.jsonl
    └── train_data
        └── data.jsonl

``train_data/data.jsonl`` is intentionally left empty because this example
performs single-task evaluation only.

Run from anywhere::

    python examples/llm_simple_qa/prepare_data.py
"""

import json
import os

# Each item is written as a single line (one JSON object per line) so that the
# JSONL parser can read it without choking on multi-line, pretty-printed JSON.
TEST_DATA = [
    {
        "question": "If Xiao Ming has 5 apples, and he gives 3 to Xiao Hua, "
                    "how many apples does Xiao Ming have left?\n"
                    "A. 2\nB. 3\nC. 4\nD. 5",
        "answer": "A",
    },
    {
        "question": "Which of the following numbers is the smallest prime "
                    "number?\nA. 0\nB. 1\nC. 2\nD. 4",
        "answer": "C",
    },
    {
        "question": "A rectangle has a length of 10 centimeters and a width of "
                    "5 centimeters, what is its perimeter in centimeters?\n"
                    "A. 20 centimeters\nB. 30 centimeters\n"
                    "C. 40 centimeters\nD. 50 centimeters",
        "answer": "B",
    },
    {
        "question": "Which of the following fractions is closest to 1?\n"
                    "A. 1/2\nB. 3/4\nC. 4/5\nD. 5/6",
        "answer": "D",
    },
    {
        "question": "If a number plus 10 equals 30, what is the number?\n"
                    "A. 20\nB. 21\nC. 22\nD. 23",
        "answer": "A",
    },
    {
        "question": "Which of the following expressions has the largest "
                    "result?\nA. 3 + 4\nB. 5 - 2\nC. 6 * 2\nD. 7 ÷ 2",
        "answer": "C",
    },
    {
        "question": "A class has 24 students, and if each student brings 2 "
                    "books, how many books are there in total?\n"
                    "A. 48\nB. 36\nC. 24\nD. 12",
        "answer": "A",
    },
    {
        "question": "Which of the following is the correct multiplication "
                    "rhyme?\nA. Three threes are seven\n"
                    "B. Four fours are sixteen\nC. Five fives are twenty-five\n"
                    "D. Six sixes are thirty-six",
        "answer": "B",
    },
    {
        "question": "If one number is three times another number, and this "
                    "number is 15, what is the other number?\n"
                    "A. 5\nB. 10\nC. 15\nD. 45",
        "answer": "A",
    },
    {
        "question": "Which of the following shapes has the longest perimeter?\n"
                    "A. Square\nB. Rectangle\nC. Circle\nD. Triangle",
        "answer": "C",
    },
]


def write_jsonl(path, records):
    """Write ``records`` to ``path`` as JSONL (one JSON object per line)."""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "dataset")
    train_path = os.path.join(base_dir, "train_data", "data.jsonl")
    test_path = os.path.join(base_dir, "test_data", "data.jsonl")

    # train_data is empty for this single-task evaluation example.
    write_jsonl(train_path, [])
    write_jsonl(test_path, TEST_DATA)

    print(f"Wrote {len(TEST_DATA)} records to {test_path}")
    print(f"Wrote empty train set to {train_path}")


if __name__ == "__main__":
    main()
