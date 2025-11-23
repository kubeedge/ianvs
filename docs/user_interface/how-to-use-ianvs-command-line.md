<<<<<<< HEAD
# How to use Ianvs command line

### List available commands

Command line: `ianvs -h`  
For example:

```shell
$ ianvs -h
usage: ianvs [-h] [-f [BENCHMARKING_CONFIG_FILE]] [-v]

AI Benchmarking Tool

optional arguments:
  -h, --help            show this help message and exit
  -f [BENCHMARKING_CONFIG_FILE], --benchmarking_config_file [BENCHMARKING_CONFIG_FILE]
                        run a benchmarking job, and the benchmarking config
                        file must be yaml/yml file.
  -v, --version         show program version info and exit.

```

### Show the version of ianvs

Command line: `ianvs -v`  
For example:

```shell
$ ianvs -v
0.1.0
```

### Run a benchmarking job

Command line: `ianvs -f [BENCHMARKING_CONFIG_FILE]`  
For example:

```yaml
ianvs -f examples/cloud-edge-collaborative-inference-for-llm/benchmarkingjob.yaml
```

The final output might look like:

```bash
+------+---------------+----------+------------+---------------------+------------+------------------------+---------------------+-------------------------+--------------------+------------------------+----------------+---------------------+----------------------------+-------------------+------------------+---------------------+-------------------------------------------------------------------------------------+
| rank |   algorithm   | Accuracy | Edge Ratio | Time to First Token | Throughput | Internal Token Latency | Cloud Prompt Tokens | Cloud Completion Tokens | Edge Prompt Tokens | Edge Completion Tokens |    paradigm    | hard_example_mining |      edgemodel-model       | edgemodel-backend | cloudmodel-model |         time        |                                         url                                         |
+------+---------------+----------+------------+---------------------+------------+------------------------+---------------------+-------------------------+--------------------+------------------------+----------------+---------------------+----------------------------+-------------------+------------------+---------------------+-------------------------------------------------------------------------------------+
|  1   | query-routing |  84.22   |   87.62    |        0.347        |   179.28   |         0.006          |       1560307       |          20339          |      10695142      |         30104          | jointinference |     OracleRouter    |  Qwen/Qwen2.5-7B-Instruct  |        vllm       |   gpt-4o-mini    | 2024-10-28 16:58:30 | ./workspace-mmlu/benchmarkingjob/query-routing/b8eb2606-950a-11ef-8cbc-c97e05df5d14 |
|  2   | query-routing |  82.75   |   77.55    |        0.316        |   216.72   |         0.005          |       2727792       |          18177          |      9470276       |         291364         | jointinference |     OracleRouter    |  Qwen/Qwen2.5-3B-Instruct  |        vllm       |   gpt-4o-mini    | 2024-10-28 16:58:19 | ./workspace-mmlu/benchmarkingjob/query-routing/b8eb2605-950a-11ef-8cbc-c97e05df5d14 |
|  3   | query-routing |  82.22   |   76.12    |        0.256        |   320.39   |         0.003          |       2978026       |          23254          |      9209538       |         29126          | jointinference |     OracleRouter    | Qwen/Qwen2.5-1.5B-Instruct |        vllm       |   gpt-4o-mini    | 2024-10-28 16:58:09 | ./workspace-mmlu/benchmarkingjob/query-routing/b8eb2604-950a-11ef-8cbc-c97e05df5d14 |
|  4   | query-routing |  75.99   |    0.0     |        0.691        |   698.83   |         0.001          |       11739216      |          79115          |         0          |           0            | jointinference |      CloudOnly      | Qwen/Qwen2.5-1.5B-Instruct |        vllm       |   gpt-4o-mini    | 2024-10-28 16:57:43 | ./workspace-mmlu/benchmarkingjob/query-routing/abe4062e-950a-11ef-8cbc-c97e05df5d14 |
|  5   | query-routing |  71.84   |   100.0    |        0.301        |   164.34   |         0.006          |          0          |            0            |      12335559      |         34817          | jointinference |       EdgeOnly      |  Qwen/Qwen2.5-7B-Instruct  |        vllm       |   gpt-4o-mini    | 2024-10-28 16:57:30 | ./workspace-mmlu/benchmarkingjob/query-routing/9b726328-950a-11ef-8cbc-c97e05df5d14 |
|  6   | query-routing |   60.3   |   100.0    |        0.206        |   176.71   |         0.006          |          0          |            0            |      12335559      |         397386         | jointinference |       EdgeOnly      |  Qwen/Qwen2.5-3B-Instruct  |        vllm       |   gpt-4o-mini    | 2024-10-28 16:57:23 | ./workspace-mmlu/benchmarkingjob/query-routing/9b726327-950a-11ef-8cbc-c97e05df5d14 |
|  7   | query-routing |  58.35   |   100.0    |        0.123        |   271.81   |         0.004          |          0          |            0            |      12335559      |         38982          | jointinference |       EdgeOnly      | Qwen/Qwen2.5-1.5B-Instruct |        vllm       |   gpt-4o-mini    | 2024-10-28 16:57:16 | ./workspace-mmlu/benchmarkingjob/query-routing/9b726326-950a-11ef-8cbc-c97e05df5d14 |
+------+---------------+----------+------------+---------------------+------------+------------------------+---------------------+-------------------------+--------------------+------------------------+----------------+---------------------+----------------------------+-------------------+------------------+---------------------+-------------------------------------------------------------------------------------+
```

Refer to [details of example].

[details of example]: ../guides/quick-start.md

=======
version https://git-lfs.github.com/spec/v1
oid sha256:350165b08e406924b824141d30da03c0df7197b3b8927226be15605a7d247003
size 5578
>>>>>>> 9676c3e (ya toh aar ya toh par)
