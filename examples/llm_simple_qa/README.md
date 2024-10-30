# README

## Simple QA

### Prepare Data

The data of simple-qa example structure is:

```
.
├── test_data
│   └── data.jsonl
└── train_data
    └── data.jsonl
```

`train_data/data.jsonl` is empty, and the `test_data/data.jsonl` is as follows:

```
{
  "question": "If Xiao Ming has 5 apples, and he gives 3 to Xiao Hua, how many apples does Xiao Ming have left?\nA. 2\nB. 3\nC. 4\nD. 5",
  "answer": "A"
}
{
  "question": "Which of the following numbers is the smallest prime number?\nA. 0\nB. 1\nC. 2\nD. 4",
  "answer": "C"
}
{
  "question": "A rectangle has a length of 10 centimeters and a width of 5 centimeters, what is its perimeter in centimeters?\nA. 20 centimeters\nB. 30 centimeters\nC. 40 centimeters\nD. 50 centimeters",
  "answer": "B"
}
{
  "question": "Which of the following fractions is closest to 1?\nA. 1/2\nB. 3/4\nC. 4/5\nD. 5/6",
  "answer": "D"
}
{
  "question": "If a number plus 10 equals 30, what is the number?\nA. 20\nB. 21\nC. 22\nD. 23",
  "answer": "A"
}
{
  "question": "Which of the following expressions has the largest result?\nA. 3 + 4\nB. 5 - 2\nC. 6 * 2\nD. 7 ÷ 2",
  "answer": "C"
}
{
  "question": "A class has 24 students, and if each student brings 2 books, how many books are there in total?\nA. 48\nB. 36\nC. 24\nD. 12",
  "answer": "A"
}
{
  "question": "Which of the following is the correct multiplication rhyme?\nA. Three threes are seven\nB. Four fours are sixteen\nC. Five fives are twenty-five\nD. Six sixes are thirty-six",
  "answer": "B"
}
{
  "question": "If one number is three times another number, and this number is 15, what is the other number?\nA. 5\nB. 10\nC. 15\nD. 45",
  "answer": "A"
}
{
  "question": "Which of the following shapes has the longest perimeter?\nA. Square\nB. Rectangle\nC. Circle\nD. Triangle",
  "answer": "C"
}
```

### Prepare Environment

You need to install the changed-sedna package, which added `JsonlDataParse` in `sedna.datasources`

Replace the file in `yourpath/anaconda3/envs/ianvs/lib/python3.x/site-packages/sedna` with `examples/resources/sedna-with-jsonl.zip`


### Run Ianvs

Run the following command:

`ianvs -f examples/llm/singletask_learning_bench/simple_qa/benchmarkingjob.yaml`

## OpenCompass Evaluation

### Prepare Environment

`pip install examples/resources/opencompass-0.2.5-py3-none-any.whl`

### Run Evaluation

`python run_op.py examples/llm/singletask_learning_bench/simple_qa/testalgorithms/gen/op_eval.py`

