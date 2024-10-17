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

There are several cloud-edge collaborative inference strategy, including:

- Query Routing $^{[1, 2]}$ : route query to smaller-scale model on edge or larger-scale model on cloud based on its difficulty.
- Speculative Decoding $^{[3]}$ : smaller-scale models predicting future multiple words quickly during decoding followed by parallel validation via larger-scale models; if validation fails then re-generation by larger-scale occurs.
- Other possible ways. 

This example currently supports convenient benchmark testing for Query-Routing strategy. 

### Details of Design

The overall design is shown in the figure below.

![image-20240926143857223](./assets/image-20240926143857223.png)

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

# Install a modified sedna wheel (a small bug and dependencies was fixed)
wget https://github.com/FuryMartin/sedna/releases/download/v0.4.1.1/sedna-0.4.1.1-py3-none-any.whl
pip install sedna-0.4.1.1-py3-none-any.whl

# Install dependencies for this example.
pip install examples/cloud-edge-collaborative-inference-for-llm/requirements.txt

# Install dependencies for Ianvs Core.
pip install requirements.txt

# Install ianvs
python setup.py install
```

## Step 2. Dataset and Model Preparation

### Dataset Configuration

Note: The currently supported dataset includes MMLU, and you can also construct the dataset you need for testing according to the format requirements of the dataset.

You need to create a dataset folder in`ianvs/` in the following structure.

```
.
├── dataset
│   └── mmlu
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

Then, check the path of `train_data` and `test_dat` in 
`examples/cloud-edge-collaborative-inference-for-llm/testenv/testenv.yaml`.

- If you created the `dataset` folder inside `ianvs/` as mentioned earlier, then the relative path is correct and does not need to be modified.

- If your `dataset` is created in a different location, please use an absolute path, and using `~` to represent the home directory is not supported.

### Metric Configuration

*Note: If you just want to run this example quickly, you can skip this step.*

We have designed multiple metrics for edge-cloud collaborative inference, including:

| Metric                  | Description                                             | Unit    |
| :---------------------- | :------------------------------------------------------ | ------- |
| Accuracy                | Accuracy on the test Dataset                            | -       |
| Rate to Edge            | proportion of queries router to edge                    | -       |
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

The `EdgeModel` is the model that will be deployed on your local machine, supporting `huggingface` and `vllm` as serving backends.

For `EdgeModel`, the open parameters are:

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

#### CloudModel Configuration

The `CloudModel` represents the model on cloud, it will call LLM API via OpenAI API format. You need to set your OPENAI_BASE_URL and OPENAI_API_KEY in the environment variables yourself, for example.

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY=sk_xxxxxxxx
```

For `CloudModel`, the open parameters are:

| Parameter Name     | Type | Description                                                  | Defalut     |
| ------------------ | ---- | ------------------------------------------------------------ | ----------- |
| model              | str  | model name                                                   | gpt-4o-mini |
| temperature        | float  | What sampling temperature to use, between 0 and 2            | 0.8         |
| top_p              | float  | nucleus sampling parameter                                   | 0.8         |
| max_tokens         | int  | The maximum number of tokens that can be generated in the chat completion | 512         |
| repetition_penalty | float  | The parameter for repetition penalty                         | 1.05        |

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
Data Processor 允许你在 `ianvs` 读取数据集后，自行实现需要的数据构造形式，如 few-shot、CoT 等复杂的 prompts 等。

Currently, supported routers include:

| Data Processor  | Description                                                  | Parameters       |
| ------------ | ------------------------------------------------------------ | ---------------- |
| MultiShotGenertor     | Few-shot query generator                      |   shot-nums         |

## Step 3. Run Ianvs

Run the following command:

`ianvs -f examples/llm/singletask_learning_bench/simple_qa/benchmarkingjob.yaml`

After the process finished, you will see output.

By changing different models and Router parameters, you may see output like:

```bash
+------+---------------+----------+--------------+---------------------+------------+------------------------+---------------------+-------------------------+--------------------+------------------------+----------------+---------------------+----------------------------+-------------------+------------------+---------------------+-------------------------------------------------------------------------------------+
| rank |   algorithm   | Accuracy | Rate to Edge | Time to First Token | Throughput | Internal Token Latency | Cloud Prompt Tokens | Cloud Completion Tokens | Edge Prompt Tokens | Edge Completion Tokens |    paradigm    | hard_example_mining |      edgemodel-model       | edgemodel-backend | cloudmodel-model |         time        |                                         url                                         |
+------+---------------+----------+--------------+---------------------+------------+------------------------+---------------------+-------------------------+--------------------+------------------------+----------------+---------------------+----------------------------+-------------------+------------------+---------------------+-------------------------------------------------------------------------------------+
|  1   | query-routing |  83.48   |    88.32     |        0.362        |   139.53   |         0.007          |       1416860       |          11836          |      10987945      |         48533          | jointinference |     OracleRouter    |  Qwen/Qwen2.5-7B-Instruct  |        vllm       |   gpt-4o-mini    | 2024-10-17 15:52:21 | ./workspace-mmlu/benchmarkingjob/query-routing/9f85b598-8c5c-11ef-ad26-51366965e425 |
|  2   | query-routing |  82.64   |    76.89     |        0.277        |   338.51   |         0.003          |       2804317       |          15707          |      9547941       |         24060          | jointinference |     OracleRouter    | Qwen/Qwen2.5-1.5B-Instruct |        vllm       |   gpt-4o-mini    | 2024-10-17 15:51:51 | ./workspace-mmlu/benchmarkingjob/query-routing/9f85b596-8c5c-11ef-ad26-51366965e425 |
|  3   | query-routing |   82.1   |    81.78     |        0.313        |   248.38   |         0.004          |       2214701       |          11887          |      10161486      |         81147          | jointinference |     OracleRouter    |  Qwen/Qwen2.5-3B-Instruct  |        vllm       |   gpt-4o-mini    | 2024-10-17 15:52:06 | ./workspace-mmlu/benchmarkingjob/query-routing/9f85b597-8c5c-11ef-ad26-51366965e425 |
|  4   | query-routing |  76.43   |     0.0      |        0.782        |  1194.58   |         0.001          |       12017546      |          47583          |         0          |           0            | jointinference |      CloudOnly      | Qwen/Qwen2.5-1.5B-Instruct |        vllm       |   gpt-4o-mini    | 2024-10-17 15:50:39 | ./workspace-mmlu/benchmarkingjob/query-routing/747c4176-8c5c-11ef-ad26-51366965e425 |
|  5   | query-routing |   71.8   |    100.0     |        0.306        |   125.22   |         0.008          |          0          |            0            |      12456589      |         55634          | jointinference |       EdgeOnly      |  Qwen/Qwen2.5-7B-Instruct  |        vllm       |   gpt-4o-mini    | 2024-10-17 13:46:20 | ./workspace-mmlu/benchmarkingjob/query-routing/0ca33c20-8c4b-11ef-ad26-51366965e425 |
|  6   | query-routing |  63.89   |    100.0     |        0.209        |   210.62   |         0.005          |          0          |            0            |      12456589      |         103378         | jointinference |       EdgeOnly      |  Qwen/Qwen2.5-3B-Instruct  |        vllm       |   gpt-4o-mini    | 2024-10-17 13:46:09 | ./workspace-mmlu/benchmarkingjob/query-routing/0ca33c1f-8c4b-11ef-ad26-51366965e425 |
|  7   | query-routing |  59.53   |    100.0     |        0.124        |   278.34   |         0.004          |          0          |            0            |      12454484      |         31193          | jointinference |       EdgeOnly      | Qwen/Qwen2.5-1.5B-Instruct |        vllm       |   gpt-4o-mini    | 2024-10-17 13:45:58 | ./workspace-mmlu/benchmarkingjob/query-routing/0ca33c1e-8c4b-11ef-ad26-51366965e425 |
+------+---------------+----------+--------------+---------------------+------------+------------------------+---------------------+-------------------------+--------------------+------------------------+----------------+---------------------+----------------------------+-------------------+------------------+---------------------+-------------------------------------------------------------------------------------+
```

Ianvs will output a `rank.csv` and `selected_rank.csv` in `ianvs/workspace`, which will record the test results of each test.

You can modify the relevant model parameters in `examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/test_queryrouting.yaml`, conduct multiple tests, and compare the results of different configurations.

## Discussion

### Query Routing's Application Scenario

Query Routing is a very useful cloud-edge collaboration strategy based on two facts:

- Calling top-tier large language models is expensive: For GPT-4o, the pricing is $5.00 / 1M input tokens and \$15.00 / 1M output tokens.

-  Not all tasks require calling top-tier models: For tasks like translation, organization, summarization, data formatting,and casual conversation, small models with 3B parameters or less can achieve satisfactory results.

These two facts suggest that if we can call different models based on the difficulty of the task, it will help save unnecessary API calls and thus reduce costs. Additionally, if edge device prformance is sufficient, locally deployed small models can also demonstrate excellent latency and throughput metrics, further enhancing user experience.

Our Oracle Router is the ideal router that can route problems where the actual performance of edge small models outperforms that of cloud large models to the edge. Experiments have shown that when Qwen2.5-7B-Instruct collaborates with gpt-4o-mini, the accuracy on the MMLU (5-shot) dataset is +11.68% compared to pure edge and +8.85% compared to pure cloud, with 88.32% of queries routed to edge.

![](./assets/Oracle%20Router%20Demo.png)

Some related research $^{[1]}$ has trained pratical routers that can save up to 40% of GPT-4 API calls while maintaining essentially unchanged accuracy on the test set.


## Future

This example builds an architecture for testing query routing strategies, but the provided dataset has some drawbacks such as being one-sided and singular, making it difficult to reflect effects in real-world scenarios. 

Besides, Speculative Decoding is another promising cloud-edge collaborative inference strategy, we should also implement it.

Thus, the future tasks of this example include:

- Build a more comprehensive dataset for better router evaluation
- Implement a Speculative Decoding example



**Reference**

[1] Ding, Dujian, et al. "Hybrid LLM: Cost-efficient and quality-aware query routing." *arXiv preprint arXiv:2404.14618* (2024).

[2] Ong, Isaac, et al. "Routellm: Learning to route llms with preference data." *arXiv preprint arXiv:2406.18665* (2024).

[3] Xia, Heming, et al. "Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding." *arXiv preprint arXiv:2401.07851* (2024).