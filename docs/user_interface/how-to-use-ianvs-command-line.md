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
ianvs -f examples/pcb-aoi/singletask_learning_bench/benchmarkingjob.yaml
```

The final output might look like:

|rank  |algorithm                |f1_score  |paradigm            |basemodel  |learning_rate  |momentum  |time                     |url                                                                                                                             |
|:----:|:-----------------------:|:--------:|:------------------:|:---------:|:-------------:|:--------:|:------------------------|:-------------------------------------------------------------------------------------------------------------------------------|
|1     |fpn_singletask_learning  | 0.8396   |singletasklearning  | FPN       | 0.1           | 0.5      | 2022-07-07 20:33:53     |/ianvs/pcb-aoi/singletask_learning_bench/workspace/benchmarkingjob/fpn_singletask_learning/49eb5ffd-fdf0-11ec-8d5d-fa163eaa99d5 |
|2     |fpn_singletask_learning  | 0.8353   |singletasklearning  | FPN       | 0.1           | 0.95     | 2022-07-07 20:31:08     |/ianvs/pcb-aoi/singletask_learning_bench/workspace/benchmarkingjob/fpn_singletask_learning/49eb5ffc-fdf0-11ec-8d5d-fa163eaa99d5 |

Refer to [details of example].

[details of example]: ../guides/quick-start.md

