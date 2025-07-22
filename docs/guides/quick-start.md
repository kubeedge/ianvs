[Links of scenarios]: ../proposals/scenarios/industrial-defect-detection/pcb-aoi.md

[the PCB-AoI public dataset]: https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi
[Details of PCB-AoI dataset]: ../proposals/scenarios/industrial-defect-detection/pcb-aoi.md
[XFTP]: https://www.xshell.com/en/xftp/
[FPN-model]: https://kubeedge.obs.cn-north-1.myhuaweicloud.com:443/ianvs/pcb-aoi/model.zip
[How to test algorithms]: how-to-test-algorithms.md
[How to contribute algorithms]: how-to-contribute-algorithms.md
[How to contribute test environments]: how-to-contribute-test-environments.md

# Quick Start

Welcome to Ianvs! Ianvs aims to test the performance of distributed synergy AI solutions following recognized standards,
in order to facilitate more efficient and effective development. **This quick start guide helps you to implement benchmarks for LLM in cloud-edge collaborative inference scenario**. You can reduce manual procedures to just a few steps so that you can
build and start your distributed synergy AI solution development within minutes.

Before using Ianvs, you might want to have the device ready:

- One machine is all you need, i.e., a laptop or a virtual machine is sufficient and a cluster is not necessary
- 2 CPUs or more
- 1 GPU with at least 6GB of memory, depends on the tested model
- 4GB+ free memory, depends on algorithm and simulation setting
- 10GB+ free disk space (depends on your model size)
- Internet connection for GitHub, PyPI,  HuggingFace, etc
- Python 3.8+ environment

In this example, we are using the Linux platform with **Python 3.8**. If you are using Windows, most steps should still apply but a few commands and package requirements might be different.

## Methods for Benchmarking with Ianvs

- To quickly experience benchmarking with Ianvs, proceed with the **Docker-Based Setup**.
- For a detailed setup process refer to the **Detailed Setup Guide**.

### Docker based setup

The Docker-based setup assumes you have Docker installed on your system and are using an Ubuntu-based Linux distribution.

**Note**: 
- If you don't have Docker installed, follow the Docker Engine installation guide [here](https://docs.docker.com/engine/install/ubuntu/). 
- To enable Docker to download datasets from Kaggle within your docker container, you need to configure the Kaggle CLI authentication token. Please follow the [official Kaggle API documentation](https://www.kaggle.com/docs/api#:~:text=is%20%24PYTHON_HOME/Scripts.-,Authentication,-In%20order%20to) to download your `kaggle.json` token. Once downloaded, move the file to the `~/ianvs/examples/cloud-edge-collaborative-inference-for-llm/` directory after doing step 1(cloning the ianvs repo):

```bash
mv /path/to/kaggle.json ~/ianvs/examples/cloud-edge-collaborative-inference-for-llm/
```

1. Clone Ianvs Repo
```
git clone https://github.com/kubeedge/ianvs.git
cd ianvs
```

2. From the root directory of Ianvs, build the `cloud-edge-collaborative-inference-for-llm` Docker image:
```bash 
docker build -t ianvs-experiment-image ./examples/cloud-edge-collaborative-inference-for-llm/
```

3. Run the image in an interactive shell:
```bash 
docker run -it ianvs-experiment-image /bin/bash 
```

4. Activate the ianvs-experiment Conda environment:
```bash 
conda activate ianvs-experiment
```

5. Set the required environment variables for the API (use either OpenAI or GROQ credentials):
```bash 
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY=sk_xxxxxxxx
```
(Alternatively, for GROQ, use GROQ_BASE_URL and GROQ_API_KEY.)

6. Run the Ianvs benchmark:
```bash 
ianvs -f examples/cloud-edge-collaborative-inference-for-llm/benchmarkingjob.yaml
```

**Note**: To help you get results quickly, we have provided a workspace folder with cached results for `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`,`Qwen/Qwen2.5-7B-Instruct` and `gpt-4o-mini`.

### Detailed Setup Guide

#### Step 1. Ianvs Preparation

```bash

# Clone Ianvs Repo
git clone https://github.com/kubeedge/ianvs.git
cd ianvs

# Create a new conda environment with Python>=3.8 and rust(venv users can do it in their own way).
conda create -n ianvs-experiment python=3.8 rust -c conda-forge

# Activate our environment
conda activate ianvs-experiment

# Install Sedna
pip install examples/resources/third_party/sedna-0.6.0.1-py3-none-any.whl

# Install dependencies for Ianvs Core.
pip install -r requirements.txt

# Install dependencies for this example.
pip install -r examples/cloud-edge-collaborative-inference-for-llm/requirements.txt

# Install ianvs
python setup.py install
```

#### Step 2. Dataset and Model Preparation

##### Dataset Preparation

1. Download `mmlu-5-shot` in the root directory of ianvs from [Ianvs-MMLU-5-shot](https://www.kaggle.com/datasets/kubeedgeianvs/ianvs-mmlu-5shot), which is a transformed MMLU-5-shot dataset formatted to fit Ianvs's requirements.
**Note**: To enable Docker to download datasets from Kaggle within your docker container, you need to configure the Kaggle CLI authentication token. Please follow the [official Kaggle API documentation](https://www.kaggle.com/docs/api#:~:text=is%20%24PYTHON_HOME/Scripts.-,Authentication,-In%20order%20to) to download your `kaggle.json` token. 
```bash
kaggle datasets download -d kubeedgeianvs/ianvs-mmlu-5shot
unzip -o ianvs-mmlu-5shot.zip
rm -rf ianvs-mmlu-5shot.zip
```

2. Then, check the path of `train_data` and `test_data` in 
`examples/cloud-edge-collaborative-inference-for-llm/testenv/testenv.yaml`.

    - If you created the `dataset` folder inside `ianvs/` as mentioned earlier, then the relative path is correct and does not need to be modified.

    - If your `dataset` is created in a different location, please use an absolute path, and using `~` to represent the home directory is not supported.

##### Model Preparation

The models are configured in `examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/test_queryrouting.yaml`.

In the configuration file, there are two models available for configuration: `EdgeModel` and `CloudModel`.

###### EdgeModel 

The `EdgeModel` is the model that will be deployed on your local machine, supporting `huggingface` and `vllm` as serving backends.

###### CloudModel

The `CloudModel` represents the model on cloud, it will call LLM API via OpenAI API format. You need to set your OPENAI_BASE_URL and OPENAI_API_KEY in the environment variables yourself, for example.

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY=sk_xxxxxxxx
```
(Alternatively, for GROQ, use GROQ_BASE_URL and GROQ_API_KEY.)

#### Step 3. Run Ianvs

##### Provided Response Cache
The testing process may take much time, depending on the number of test cases and the inference speed of the model.

To enable you directly get the results, here we provide a workspace folder with cached results of `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`,`Qwen/Qwen2.5-7B-Instruct` and `gpt-4o-mini`.

You can download `workspace-mmlu` folder from [Ianvs-MMLU-5-shot](https://www.kaggle.com/datasets/kubeedgeianvs/ianvs-mmlu-5shot) and put it under your `ianvs` folder.

- Since we have already downloaded the `Ianvs-MMLU-5-shot` folder. There is no need to do this again. 

##### Run Joint Inference example

Run the following command:
```bash
ianvs -f examples/cloud-edge-collaborative-inference-for-llm/benchmarkingjob.yaml
```

After the process finished, you will see output like this:

```bash
[2025-04-12 09:20:14,523] edge_model.py(43) [INFO] - {'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'backend': 'vllm', 'temperature': 0, 'top_p': 0.8, 'max_tokens': 512, 'repetition_penalty': 1.05, 'tensor_parallel_size': 4, 'gpu_memory_utilization': 0.9, 'use_cache': True}
[2025-04-12 09:20:14,524] cloud_model.py(34) [INFO] - {'model': 'gpt-4o-mini', 'temperature': 0, 'top_p': 0.8, 'max_tokens': 512, 'repetition_penalty': 1.05, 'use_cache': True}
[2025-04-12 09:20:14,880] joint_inference.py(73) [INFO] - Loading dataset
[2025-04-12 09:20:15,943] hard_sample_mining.py(30) [INFO] - USING EdgeOnlyFilter
[2025-04-12 09:20:15,943] joint_inference.py(162) [INFO] - Inference Start
100%|██████████████████████████████████| 14042/14042 [00:03<00:00, 4418.66it/s, Edge=14042, Cloud=0]
[2025-04-12 09:20:19,122] joint_inference.py(186) [INFO] - Inference Finished
[2025-04-12 09:20:19,122] joint_inference.py(131) [INFO] - Release models
[2025-04-12 09:20:23,844] edge_model.py(43) [INFO] - {'model': 'Qwen/Qwen2.5-3B-Instruct', 'backend': 'vllm', 'temperature': 0, 'top_p': 0.8, 'max_tokens': 512, 'repetition_penalty': 1.05, 'tensor_parallel_size': 4, 'gpu_memory_utilization': 0.9, 'use_cache': True}
[2025-04-12 09:20:23,844] cloud_model.py(34) [INFO] - {'model': 'gpt-4o-mini', 'temperature': 0, 'top_p': 0.8, 'max_tokens': 512, 'repetition_penalty': 1.05, 'use_cache': True}
[2025-04-12 09:20:23,851] joint_inference.py(73) [INFO] - Loading dataset
[2025-04-12 09:20:24,845] hard_sample_mining.py(30) [INFO] - USING EdgeOnlyFilter
[2025-04-12 09:20:24,845] joint_inference.py(162) [INFO] - Inference Start
100%|██████████████████████████████████| 14042/14042 [00:03<00:00, 4413.68it/s, Edge=14042, Cloud=0]
[2025-04-12 09:20:28,027] joint_inference.py(186) [INFO] - Inference Finished
[2025-04-12 09:20:28,027] joint_inference.py(131) [INFO] - Release models
[2025-04-12 09:20:32,741] edge_model.py(43) [INFO] - {'model': 'Qwen/Qwen2.5-7B-Instruct', 'backend': 'vllm', 'temperature': 0, 'top_p': 0.8, 'max_tokens': 512, 'repetition_penalty': 1.05, 'tensor_parallel_size': 4, 'gpu_memory_utilization': 0.9, 'use_cache': True}
[2025-04-12 09:20:32,741] cloud_model.py(34) [INFO] - {'model': 'gpt-4o-mini', 'temperature': 0, 'top_p': 0.8, 'max_tokens': 512, 'repetition_penalty': 1.05, 'use_cache': True}
[2025-04-12 09:20:32,749] joint_inference.py(73) [INFO] - Loading dataset
[2025-04-12 09:20:33,738] hard_sample_mining.py(30) [INFO] - USING EdgeOnlyFilter
[2025-04-12 09:20:33,738] joint_inference.py(162) [INFO] - Inference Start
100%|██████████████████████████████████| 14042/14042 [00:03<00:00, 4456.34it/s, Edge=14042, Cloud=0]
[2025-04-12 09:20:36,890] joint_inference.py(186) [INFO] - Inference Finished
[2025-04-12 09:20:36,890] joint_inference.py(131) [INFO] - Release models
```

### Results

Change the Router type to `EdgeOnly`, `CloudOnly`, `OracleRouter` (or another router) will yield better results.

The recommend testing order is `EdgeOnly`, `CloudOnly`, `OracleRouter`, `BERTRouter`, `RandomRouter`.

By changing different models and Router parameters, you may see output like:

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

Ianvs will output a `rank.csv` and `selected_rank.csv` in `ianvs/workspace`, which will record the test results of each test.

You can modify the relevant model parameters in `examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/test_queryrouting.yaml`, conduct multiple tests, and compare the results of different configurations.

This ends the quick start experiment. For more details on **cloud-edge collaborative inference scenario** example, you can refer to [this](https://github.com/AryanNanda17/ianvs/tree/main/examples/cloud-edge-collaborative-inference-for-llm) folder on github.

# What is next

If the reader is ready to explore more on Ianvs, e.g., after the quick start, the following links might help:

[How to test algorithms]

[How to contribute algorithms]

[How to contribute test environments]

[Links of scenarios]

[Details of PCB-AoI dataset]

If any problems happen, the user can refer to [the issue page on Github](https://github.com/kubeedge/ianvs/issues) for help and are also welcome to raise any new issue.

Enjoy your journey on Ianvs!