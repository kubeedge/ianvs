#!/usr/bin/env python3
"""Standard ordered preparation entry point for the Simple QA dataset."""

import argparse
import json
import sys
from pathlib import Path


SAMPLE_ROWS = [
    {
        "question": "If Xiao Ming has 5 apples, and he gives 3 to Xiao Hua, how many apples does Xiao Ming have left?\nA. 2\nB. 3\nC. 4\nD. 5",
        "answer": "A",
    },
    {
        "question": "Which of the following numbers is the smallest prime number?\nA. 0\nB. 1\nC. 2\nD. 4",
        "answer": "C",
    },
    {
        "question": "A rectangle has a length of 10 centimeters and a width of 5 centimeters, what is its perimeter in centimeters?\nA. 20 centimeters\nB. 30 centimeters\nC. 40 centimeters\nD. 50 centimeters",
        "answer": "B",
    },
    {
        "question": "Which of the following fractions is closest to 1?\nA. 1/2\nB. 3/4\nC. 4/5\nD. 5/6",
        "answer": "D",
    },
    {
        "question": "If a number plus 10 equals 30, what is the number?\nA. 20\nB. 21\nC. 22\nD. 23",
        "answer": "A",
    },
    {
        "question": "Which of the following expressions has the largest result?\nA. 3 + 4\nB. 5 - 2\nC. 6 * 2\nD. 7 ÷ 2",
        "answer": "C",
    },
    {
        "question": "A class has 24 students, and if each student brings 2 books, how many books are there in total?\nA. 48\nB. 36\nC. 24\nD. 12",
        "answer": "A",
    },
    {
        "question": "Which of the following is the correct multiplication rhyme?\nA. Three threes are seven\nB. Four fours are sixteen\nC. Five fives are twenty-five\nD. Six sixes are thirty-six",
        "answer": "B",
    },
    {
        "question": "If one number is three times another number, and this number is 15, what is the other number?\nA. 5\nB. 10\nC. 15\nD. 45",
        "answer": "A",
    },
    {
        "question": "Which of the following shapes has the longest perimeter?\nA. Square\nB. Rectangle\nC. Circle\nD. Triangle",
        "answer": "C",
    },
]


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the llm_simple_qa dataset in Ianvs example layout."
    )
    parser.add_argument(
        "--dataset-root",
        "--output-dir",
        dest="dataset_root",
        type=Path,
        default=Path("./dataset/llm_simple_qa"),
        help="Destination dataset root. Defaults to ./dataset/llm_simple_qa.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing generated files.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Generate a single test sample for smoke-test runs.",
    )
    args = parser.parse_args()
    rows = SAMPLE_ROWS[:1] if args.smoke else SAMPLE_ROWS

    train_dir = args.dataset_root / "train_data"
    test_dir = args.dataset_root / "test_data"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_path = train_dir / "data.jsonl"
    test_path = test_dir / "data.jsonl"
    train_path.write_text("", encoding="utf-8")
    _write_jsonl(test_path, rows)

    json.dump(
        {
            "dataset_root": str(args.dataset_root),
            "train_data": str(train_path),
            "test_data": str(test_path),
            "test_size": len(rows),
            "smoke": args.smoke,
        },
        sys.stdout,
        ensure_ascii=False,
        indent=2,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
