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
{"question": "如果小明有5个苹果，他给了小华3个，那么小明还剩下多少个苹果？\nA. 2个\nB. 3个\nC. 4个\nD. 5个", "answer": "A"}
{"question": "下列哪个数是最小的质数？\nA. 0\nB. 1\nC. 2\nD. 4", "answer": "C"}
{"question": "一个长方形的长是10厘米，宽是5厘米，它的周长是多少厘米？\nA. 20厘米\nB. 30厘米\nC. 40厘米\nD. 50厘米", "answer": "B"}
{"question": "下列哪个分数是最接近1的？\nA. 1/2\nB. 3/4\nC. 4/5\nD. 5/6", "answer": "D"}
{"question": "如果一个数加上10等于30，那么这个数是多少？\nA. 20\nB. 21\nC. 22\nD. 23", "answer": "A"}
{"question": "下列哪个算式的结果最大？\nA. 3 + 4\nB. 5 - 2\nC. 6 * 2\nD. 7 ÷ 2", "answer": "C"}
{"question": "一个班级有24个学生，如果每个学生都带了2本书，那么总共有多少本书？\nA. 48本\nB. 36本\nC. 24本\nD. 12本", "answer": "A"}
{"question": "下列哪个是正确的乘法口诀？\nA. 三三得七\nB. 四四十六\nC. 五五二十五\nD. 六六三十六", "answer": "B"}
{"question": "如果一个数是另一个数的3倍，并且这个数是15，那么另一个数是多少？\nA. 5\nB. 10\nC. 15\nD. 45", "answer": "A"}
{"question": "下列哪个图形的周长最长？\nA. 正方形\nB. 长方形\nC. 圆形\nD. 三角形", "answer": "C"}
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

