# Cloud Edge Collaborative inference for LLM: the Ianvs-MMLU-5-shot dataset

Download link: [Kaggle](https://www.kaggle.com/datasets/kubeedgeianvs/ianvs-mmlu-5shot)

## Authors

- KubeEdge/Ianvs: Yu Fan

## Background

### Why LLM need cloud-edge collaborative inference?

Currently, LLMs have billions or even trillions of parameters, requiring massive computing power for training and deployment. Therefore, they are often deployed in cloud computing centers and serving via APIs. However, such service paradigm faces many drawbacks.

- Time to First Token(TTFT) is quite long, due to transmission delays from the distance to the data center.
- Uploading user data to the cloud may lead to additional privacy risks and retraining risks.
- Calling APIs of the most advanced models (GPT-4o *et.al*) is often very expensive.
- Not all tasks require high-performance models to complete.

These issues can be addressed by introducing Edge Computing, which is an architecture featured by low-latency, privacy security, energy-efficient. 

By deploying small-scale LLMs on edge devices like mobile phones, PCs and communication base station, users will have low-latency and privacy-secure services. Empirically, models with fewer than 3B parameters are possible to be deployed on the aforementioned edge devices. However, due to Scaling Law, smaller models perform worse than larger models, so they can only maintain good performance on certain tasks. 

Thus, smaller models on edge should collaborate with larger models on cloud to achieve better performance on other tasks.

### Possible Collaborative Inference Strategy 

There are several cloud-edge collaborative inference strategy, including:

- **Query Routing**: route query to smaller-scale model on edge or larger-scale model on cloud based on its difficulty.
- **Speculative Decoding**: smaller-scale models predicting future multiple words quickly during decoding followed by parallel validation via larger-scale models; if validation fails then re-generation by larger-scale occurs.

## Data Explorer

`Ianvs-MMLU-5-shot`, is a transformed MMLU-5-shot dataset formatted to fit Ianvs's requirements.

The MMLU (Massive Multitask Language Understanding) 5-shot dataset is a benchmark designed to evaluate the multitask and reasoning capabilities of language models. In the 5-shot setting, each test question is preceded by 5 example Q&A pairs from the same subject to test few-shot generalization. The dataset includes multiple-choice questions, making it suitable for assessing models' broad knowledge and reasoning across many domains.


The Structure of this dataset is as follows:
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