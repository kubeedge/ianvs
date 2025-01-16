# Quick Start

## Introduction

This example aims to implement benchmarks for **LLM in cloud-edge collaborative inference scenario**. 

### Why LLM need cloud-edge collaborative inference?

Currently, such LLMs have billions or even trillions of parameters, requiring massive computing power for training and deployment. Therefore, they are often deployed in cloud computing centers and serving via APIs. However, such service paradigm faces many drawbacks.

- Time to First Token(TTFT) is quite long, due to transmission delays from the distance to the data center.
- Uploading user data to the cloud may lead to additional privacy risks and retraining risks.
- Calling APIs of the most advanced models (GPT-4o *et.al*) is often very expensive.
- Not all tasks require high-performance models to complete.

These issues can be addressed by introducing Edge Computing, which is an architecture featured by low-latency, privacy security, energy-efficient. 

By deploying small-scale LLMs on edge devices like mobile phones, PCs and communication base station, users will have low-latency and privacy-secure services. Empirically, models with fewer than 3B parameters are possible to be deployed on the aforementioned edge devices. However, due to Scaling Law, smaller models perform worse than larger models, so they can only maintain good performance on certain tasks. 

Thus, smaller models on edge should collaborate with larger models on cloud to achieve better performance on other tasks.

### Possible Collaborative Inference Strategy 

There are several cloud-edge collaborative inference strategy, one of which is Query Routing $^{[1, 2]}$, which routes query to smaller-scale model on edge or larger-scale model on cloud based on its difficulty.

Additionally, Speculative Decoding $^{[3]}$ is another promising strategy to further improve the performance of collaborative inference, where smaller-scale models predicting future multiple words quickly during decoding followed by parallel validation via larger-scale models; if validation fails then re-generation by larger-scale occurs.


### Details of Design

The overall design is shown in the figure below.

![image-20240926143857223](./assets/image-20250115535482354.png)

When Ianvs starts the benchmarking job, the Test Env Manager will first pass the data of the user-specified Dataset to the Test Case Controller for Joint Inference one by one.

Joint Inference supports multiple modes, including `mining-then-inference`, `inference-then-mining`, and `self-design`. Among them, `mining-then-inference` is suitable for LLM scenarios, `inference-then-mining` is suitable for CV scenarios, and `self-design` allows you to implement more complex collaborative inference strategies on your own.

In this example, we will rely on Ianvs' Joint Inference Paradigm using the `inference-then-mining` mode to implement a Query Routing strategy. First, we call your custom Hard Example Mining module to determine if it is a hard case. If it is, we call the inference interface of the Edge Model to complete the inference; if not, we call the inference interface of the Cloud Model to complete it.

To save API calls during multi-round testing, this example has designed a result caching mechanism in both EdgeModel and Cloud Model. For questions that have already been tested, cached results will be read and returned.

After all tests are completed, the Test Env Manager will calculate relevant metrics based on selected Metrics and hand over to Story Manager for printing test reports and generating Leader Board.

## Required Resources

Before using this example, you need to have the device ready:

One machine is all you need, i.e., a laptop or a virtual machine is sufficient and a cluster is not necessary

- 2 CPUs or more

- 1 GPU with at least 6GB of memory, depends on the tested model

- 4GB+ free memory, depends on algorithm and simulation setting

- 10GB+ free disk space (depends on your model size)

- Internet connection for GitHub, PyPI,  HuggingFace, etc

- Python 3.8+ environment

## Step 1. Ianvs Preparation

```bash
# Create a new conda environment with Python>=3.8 (venv users can do it in their own way).
conda create -n ianvs-experiment python=3.8

# Activate our environment
conda activate ianvs-experiment

# Clone Ianvs Repo
git clone https://github.com/kubeedge/ianvs.git
cd ianvs

# Install Sedna
pip install examples/resources/third_party/sedna-0.6.0.1-py3-none-any.whl

# Install dependencies for this example.
pip install -r examples/cloud-edge-collaborative-inference-for-llm/requirements.txt

# Install dependencies for Ianvs Core.
pip install -r requirements.txt

# Install ianvs
python setup.py install
```

## Step 2. Dataset and Model Preparation

### Dataset Configuration

Here, we provide `MMLU-5-shot` dataset and `GPQA-diamond` dataset for testing. The following is the instruction for dataset preparation for `MMLU-5-shot`, `GPQA-diamond` follows the same progress.

1. Download `mmlu-5-shot` from [Ianvs-MMLU-5-shot](https://huggingface.co/datasets/FuryMartin/Ianvs-MMLU-5-shot), (or [Ianvs-GPQA-diamond](https://huggingface.co/datasets/FuryMartin/Ianvs-GPQA-diamond)) which is a transformed MMLU-5-shot dataset formatted to fit Ianvs's requirements.

2. Create a `dataset` folder in the root directory of Ianvs and move `mmlu-5-shot` into the `dataset` folder.

3. Then, check the path of `train_data` and `test_dat` in 
`examples/cloud-edge-collaborative-inference-for-llm/testenv/testenv.yaml`.

    - If you created the `dataset` folder inside `ianvs/` as mentioned earlier, then the relative path is correct and does not need to be modified.

    - If your `dataset` is created in a different location, please use an absolute path, and using `~` to represent the home directory is not supported.

#### Dataset Details

If you want to construct your own dataset, please see the details below and follow the instruction.

```
.
├── dataset
│   └── mmlu-5-shot
│       ├── test_data
│       │   ├── data.jsonl
│       │   └── metadata.json
│       └── train_data
│           └── data.json
```

Leave `train_data/data.jsonl` as empty.

The file `data.jsonl` stores the main content of the dataset. Each line contains must contain keys `query`, `response`, `explanation`,`level_1_dim`, `level_2_dim`, `level_3_dim`, `level_4_dim`

Here is an example:

```json
{"query": "Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6", "response": "B", "explanation": "", "level_1_dim": "single-modal", "level_2_dim": "text", "level_3_dim": "knowledge Q&A", "level_4_dim": "abstract_algebra"}
{"query": "Question: Let p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5.\nA. 8\nB. 2\nC. 24\nD. 120", "response": "C", "explanation": "", "level_1_dim": "single-modal", "level_2_dim": "text", "level_3_dim": "knowledge Q&A", "level_4_dim": "abstract_algebra"}
``` 

The `metadata.jsonl` stores information about the data, including `dataset`, `description`, `level_1_dim`, `level_2_dim`, `level_3_dim`, `level_4_dim`. 

Here is an example:

```json
{
    "dataset": "MMLU",
    "description": "Measuring Massive Multitask Language Understanding by Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt (ICLR 2021).",
    "level_1_dim": "single-modal",
    "level_2_dim": "text", 
    "level_3_dim": "Q&A",
    "level_4_dim": "general"
}
```



### Metric Configuration

*Note: If you just want to run this example quickly, you can skip this step.*

We have designed multiple metrics for edge-cloud collaborative inference, including:

| Metric                  | Description                                             | Unit    |
| :---------------------- | :------------------------------------------------------ | ------- |
| Accuracy                | Accuracy on the test Dataset                            | -       |
| Edge Ratio            | proportion of queries router to edge                    | -       |
| Time to First Token     | Time taken to generate the first token                  | s       |
| Internal Token Latency  | Time taken to generate each token                       | s       |
| Throughput              | Token generation speed                                  | token/s |
| Cloud Prompt Tokens     | Number of prompt tokens consumed by Cloud Model         | -       |
| Cloud Completion Tokens | Number of completion tokens generated by Cloud Model    | -       |
| Edge Prompt Tokens      | Number of prompt tokens consumed by the Edge Model      | -       |
| Edge Completion Tokens  | Number of completion tokens generated by the Edge Model | -       |

Each metric is calculated by a module in `examples/cloud-edge-collaborative-inference-for-llm/testenv`. For more details, please check the folder.

You can select multiple metrics in `examples/cloud-edge-collaborative-inference-for-llm/testenv/testenv.yaml`.

### Model Configuration

*Note: If you just want to run this example quickly, you can skip this step.*

The models are configured in `examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/test_queryrouting.yaml`.

In the configuration file, there are two models available for configuration: `EdgeModel` and `CloudModel`.

#### EdgeModel Configuration

The `EdgeModel` is designed to be deployed on your local machine, offering support for multiple serving backends including `huggingface`, `vllm`, `EAGLE`, and `LADE`. Additionally, it provides the flexibility to integrate with API-based model services.

The `CloudModel` represents the model on cloud. For extensibility, it supports both API-based models (which call LLM API via OpenAI API format) and local inference using backends like `huggingface`, `vllm`, `EAGLE`, and `LADE`. For API-based models, you need to set your `OPENAI_BASE_URL` and `OPENAI_API_KEY` in the environment variables yourself, for example:

For both `EdgeModel` and `CloudModel`, the open parameters are:

| Parameter Name         | Type  | Description                                                  | Defalut                  |
| ---------------------- | ----- | ------------------------------------------------------------ | ------------------------ |
| model                  | str   | model name                                                   | Qwen/Qwen2-1.5B-Instruct |
| backend                | str   | model serving framework                                      | huggingface              |
| temperature            | float | What sampling temperature to use, between 0 and 2            | 0.8                      |
| top_p                  | float | nucleus sampling parameter                                   | 0.8                      |
| max_tokens             | int   | The maximum number of tokens that can be generated in the chat completion | 512                      |
| repetition_penalty     | float | The parameter for repetition penalty                         | 1.05                     |
| tensor_parallel_size   | int   | The size of tensor parallelism (Used for vLLM)               | 1                        |
| gpu_memory_utilization | float | The percentage of GPU memory utilization (Used for vLLM)     | 0.9                      |

If you want to call API-based models, you need to set your `OPENAI_BASE_URL` and `OPENAI_API_KEY` in the environment variables yourself, for example:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY=sk_xxxxxxxx
```

#### Router Configuration

Router is a component that routes the query to the edge or cloud model. The router is configured by `hard_example_mining` in `examples/cloud-edge-collaborative-inference-for-llm/testrouters/query-routing/test_queryrouting.yaml`.

Currently, supported routers include:

| Router Type  | Description                                                  | Parameters       |
| ------------ | ------------------------------------------------------------ | ---------------- |
| EdgeOnly     | Route all queries to the edge model.                         | -                |
| CloudOnly    | Route all queries to the cloud model.                        | -                |
| OracleRouter | Optimal Router         |         |
| BERTRouter   | Use a BERT classifier to route the query to the edge or cloud model. | model, threshold |
| RandomRouter | Route the query to the edge or cloud model randomly.         | threshold        |

You can modify the `router` parameter in `test_queryrouting.yaml` to select the router you want to use.

For BERT router, you can use [routellm/bert](https://huggingface.co/routellm/bert) or [routellm/bert_mmlu_augmented](https://huggingface.co/routellm/bert_mmlu_augmented) or your own BERT model/

#### Data Processor Configuration
The Data Processor allows you to custom your own data format after the dataset loaded.

Currently, supported routers include:

| Data Processor  | Description                                                  | Parameters       |
| ------------ | ------------------------------------------------------------ | ---------------- |
| OracleRouterDatasetProcessor     |  Expose `gold` label to OracleRouter                      |   -         |

## Step 3. Run Ianvs

### Provided Response Cache
The testing process may take much time, depending on the number of test cases and the inference speed of the model.

To enable you directly get the results, here we provide a workspace folder with cached results of `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`,`Qwen/Qwen2.5-7B-Instruct` and `gpt-4o-mini`.

You can download `workspace-mmlu` folder from [Ianvs-MMLU-5-shot](https://huggingface.co/datasets/FuryMartin/Ianvs-MMLU-5-shot) and put it under your `ianvs` folder.

### Run Joint Inference example

Run the following command:

`ianvs -f examples/cloud-edge-collaborative-inference-for-llm/benchmarkingjob.yaml`

After the process finished, you will see output like this:

```bash
[2024-10-28 18:03:37,314] edge_model.py(43) [INFO] - {'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'backend': 'vllm', 'temperature': 0, 'top_p': 0.8, 'max_tokens': 512, 'repetition_penalty': 1.05, 'tensor_parallel_size': 4, 'gpu_memory_utilization': 0.9, 'use_cache': True}
[2024-10-28 18:03:37,314] cloud_model.py(34) [INFO] - {'model': 'gpt-4o-mini', 'temperature': 0, 'top_p': 0.8, 'max_tokens': 512, 'repetition_penalty': 1.05, 'use_cache': True}
[2024-10-28 18:03:37,850] joint_inference.py(73) [INFO] - Loading dataset
[2024-10-28 18:03:38,703] hard_sample_mining.py(30) [INFO] - USING EdgeOnlyFilter
[2024-10-28 18:03:38,704] joint_inference.py(162) [INFO] - Inference Start
100%|██████████████████████████████████| 14042/14042 [00:02<00:00, 6182.92it/s, Edge=14042, Cloud=0]
[2024-10-28 18:03:40,975] joint_inference.py(186) [INFO] - Inference Finished
[2024-10-28 18:03:40,976] joint_inference.py(131) [INFO] - Release models
```

### Results

Change the Router type to `EdgeOnly`, `CloudOnly`, `OracleRouter` (or another router) will yield better results.

The recommend testing order is `EdgeOnly`, `CloudOnly`, `OracleRouter`, `BERTRouter`, `RandomRouter`.

By changing different models and Router parameters, you may see output like the following table tested on `MMLU-5-shot` dataset:

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


Since MMLU-5-shot has a large amount of data, we recommend using the GPQA dataset to test the latency and throughput performance under different inference frameworks and Oracle Router. Below are the test results for two inference frameworks `vllm` and `EAGLE` under Oracle Router:

```bash
+------+---------------+----------+------------+---------------------+------------+------------------------+---------------------+-------------------------+--------------------+------------------------+----------------+---------------------+------------------------+-------------------+------------------+---------------------+-------------------------------------------------------------------------------------+
| rank |   algorithm   | Accuracy | Edge Ratio | Time to First Token | Throughput | Internal Token Latency | Cloud Prompt Tokens | Cloud Completion Tokens | Edge Prompt Tokens | Edge Completion Tokens |    paradigm    | hard_example_mining |    edgemodel-model     | edgemodel-backend | cloudmodel-model |         time        |                                         url                                         |
+------+---------------+----------+------------+---------------------+------------+------------------------+---------------------+-------------------------+--------------------+------------------------+----------------+---------------------+------------------------+-------------------+------------------+---------------------+-------------------------------------------------------------------------------------+
|  1   | query-routing |  54.04   |   78.79    |        0.278        |    47.1    |         0.021          |        12081        |          20383          |       43636        |         64042          | jointinference |     OracleRouter    | Qwen/Qwen2-7B-Instruct |        vllm       |   gpt-4o-mini    | 2025-01-16 16:27:00 | ./workspace-gpqa/benchmarkingjob/query-routing/a5477f86-d3e3-11ef-aa28-0242ac110008 |
|  2   | query-routing |  39.39   |    0.0     |        1.388        |   57.48    |         0.017          |        52553        |          100395         |         0          |           0            | jointinference |      CloudOnly      | Qwen/Qwen2-7B-Instruct |        vllm       |   gpt-4o-mini    | 2025-01-16 16:13:12 | ./workspace-gpqa/benchmarkingjob/query-routing/e204bac6-d3dc-11ef-8dfe-0242ac110008 |
|  3   | query-routing |  32.83   |   100.0    |        0.059        |   44.95    |         0.022          |          0          |            0            |       56550        |         80731          | jointinference |       EdgeOnly      | Qwen/Qwen2-7B-Instruct |        vllm       |   gpt-4o-mini    | 2025-01-16 13:12:20 | ./workspace-gpqa/benchmarkingjob/query-routing/fdda7ce2-d3c1-11ef-8ea0-0242ac110008 |
|  4   | query-routing |  28.28   |   100.0    |        0.137        |   66.12    |         0.015          |          0          |            0            |       56550        |         67426          | jointinference |       EdgeOnly      | Qwen/Qwen2-7B-Instruct |    EagleSpecDec   |   gpt-4o-mini    | 2025-01-16 12:43:05 | ./workspace-gpqa/benchmarkingjob/query-routing/fdda7aa8-d3c1-11ef-8ea0-0242ac110008 |
+------+---------------+----------+------------+---------------------+------------+------------------------+---------------------+-------------------------+--------------------+------------------------+----------------+---------------------+------------------------+-------------------+------------------+---------------------+-------------------------------------------------------------------------------------+
```


## Discussion

### Query Routing's Application Scenario

Query Routing is a very useful cloud-edge collaboration strategy based on two facts:

- Calling top-tier large language models is expensive: For GPT-4o, the pricing is \$5.00 / 1M input tokens and \$15.00 / 1M output tokens.

-  Not all tasks require calling top-tier models: For tasks like translation, organization, summarization, data formatting,and casual conversation, small models with 3B parameters or less can achieve satisfactory results.

These two facts suggest that if we can call different models based on the difficulty of the task, it will help save unnecessary API calls and thus reduce costs. Additionally, if edge device prformance is sufficient, locally deployed small models can also demonstrate excellent latency and throughput metrics, further enhancing user experience.

Our Oracle Router is the ideal router that can route problems where the actual performance of edge small models outperforms that of cloud large models to the edge. Experiments have shown that when Qwen2.5-7B-Instruct collaborates with gpt-4o-mini, the accuracy on the MMLU (5-shot) dataset is +12.38% compared to pure edge and +8.23% absolute accuracy compared to pure cloud, with 87.62% of queries routed to edge.

![](./assets/Oracle%20Router%20Demo.png)

You can modify and run `performance-cost-plot.py` to get your Performance-Cost figure.

Some related research $^{[1]}$ has trained pratical routers that can save up to 40% of GPT-4 API calls while maintaining essentially unchanged accuracy on the test set.


## Future

This example builds an architecture for testing query routing strategies, but the provided dataset has some drawbacks such as being one-sided and singular, making it difficult to reflect effects in real-world scenarios. 

Besides, Speculative Decoding is another promising cloud-edge collaborative inference strategy, we should also implement it.

Thus, the future tasks of this example include:

- Build a more comprehensive dataset for better router evaluation
- Try to consider a native Speculative Decoding in cloud-edge collaborative inference scenario.



**Reference**

[1] Ding, Dujian, et al. "Hybrid LLM: Cost-efficient and quality-aware query routing." *arXiv preprint arXiv:2404.14618* (2024).

[2] Ong, Isaac, et al. "Routellm: Learning to route llms with preference data." *arXiv preprint arXiv:2406.18665* (2024).

[3] Xia, Heming, et al. "Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding." *arXiv preprint arXiv:2401.07851* (2024).