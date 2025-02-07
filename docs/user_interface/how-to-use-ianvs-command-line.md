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
ianvs -f examples/pcb-aoi/singletask_learning_bench/fault_detection/benchmarkingjob.yaml
```

The final output might look like:

```bash
+------+-------------------------+----------+--------------------+-----------+--------------------+-------------------------+---------------------+------------------------------------------------------------------------------------------+
| rank |        algorithm        | f1_score |      paradigm      | basemodel | basemodel-momentum | basemodel-learning_rate |         time        |                                           url                                            |
+------+-------------------------+----------+--------------------+-----------+--------------------+-------------------------+---------------------+------------------------------------------------------------------------------------------+
|  1   | fpn_singletask_learning |  0.8527  | singletasklearning |    FPN    |        0.5         |           0.1           | 2025-01-06 14:30:30 | ./workspace/benchmarkingjob/fpn_singletask_learning/3a76bc25-cc0b-11ef-9f00-65cc74a7c013 |
|  2   | fpn_singletask_learning |  0.844   | singletasklearning |    FPN    |        0.95        |           0.1           | 2025-01-06 14:25:18 | ./workspace/benchmarkingjob/fpn_singletask_learning/3a76bc24-cc0b-11ef-9f00-65cc74a7c013 |
+------+-------------------------+----------+--------------------+-----------+--------------------+-------------------------+---------------------+------------------------------------------------------------------------------------------+
```

Refer to [details of example].

[details of example]: ../guides/quick-start.md

