Large Language Model Edge Benchmark Suite: Implementation on KubeEdge-lanvs


## dataset

### Prepare Data

The data of llm-edge-benchmark-suite example structure is:

```
.
├── test_data
│   └── data.jsonl
└── train_data
    └── data.jsonl
```

`train_data/data.jsonl` is empty, and the `test_data/data.jsonl` is as follows:

```
{"question": "Which of the following numbers is the smallest prime number?\nA. 0\nB. 1\nC. 2\nD. 4", "answer": "C"}
```
### prepare env

```shell
python setup.py install
```

### Run Ianvs



```shell
ianvs -f examples/llm-edge-benchmark-suite/single_task_bench/benchmarkingjob.yaml
```


```shell
ianvs -f examples/llm-edge-benchmark-suite/single_task_bench_with_compression/benchmarkingjob.yaml
```

