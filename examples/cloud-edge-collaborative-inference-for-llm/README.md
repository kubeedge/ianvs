# Quick Start

Before using this example, you need to have the device ready:

One machine is all you need, i.e., a laptop or a virtual machine is sufficient and a cluster is not necessary

- 2 CPUs or more

- 1 GPU with at least 4GB of memory, depends on the tested model

- 4GB+ free memory, depends on algorithm and simulation setting

- 10GB+ free disk space

- Internet connection for GitHub and pip, etc

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

You need to create a dataset folder in`ianvs/` in the following structure.

```
.
├── dataset
    ├── test_data
    │   └── data.jsonl
    └── train_data
        └── data.jsonl
```

Leave `train_data/data.jsonl` as empty.

Fill the `test_data/data.jsonl` as follows:

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

Then, check the path of `train_data` and `test_dat` in 
`examples/cloud-edge-collaborative-inference-for-llm/testenv/testenv.yaml`.

- If you created the `dataset` folder inside `ianvs/` as mentioned earlier, then the relative path is correct and does not need to be modified.

- If your `dataset` is created in a different location, please use an absolute path, and using `~` to represent the home directory is not supported.

### Model Configuration

The models are configured in `examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/test_queryrouting.yaml`.

If you just want to run this example quickly, you can skip the configuration.

The default model is `Qwen/Qwen2-1.5B-Instruct`, which is configured by `model_name`. We use transformers to load the model, which is completely consistent with most LLMs you can see. If you don't have this model, HuggingFace will download it automatically.

This example support two LLM serving framework: HuggingFace and vllm. You can choose one and change the value of `backend`.

Besides, this example support different quatization methods. You can pass "full", "4-bit", "8-bit" to the value of `quantization`


## Step 3. Run Ianvs

Run the following command:

`ianvs -f examples/llm/singletask_learning_bench/simple_qa/benchmarkingjob.yaml`

After several seconds, you will see the following output:

```bash
+------+---------------+-----+-----------+----------------+-----------+------------+---------------------+--------------------------+-------------------+------------------------+---------------------+---------------------+--------------------------------------------------------------------------------+
| rank |   algorithm   | acc | edge-rate |    paradigm    | basemodel |  apimodel  | hard_example_mining |   basemodel-model_name   | basemodel-backend | basemodel-quantization | apimodel-model_name |         time        |                                      url                                       |
+------+---------------+-----+-----------+----------------+-----------+------------+---------------------+--------------------------+-------------------+------------------------+---------------------+---------------------+--------------------------------------------------------------------------------+
|  1   | query-routing | 1.0 |    0.4    | jointinference | EdgeModel | CloudModel |         BERT        | Qwen/Qwen2-1.5B-Instruct |    huggingface    |          full          |     gpt-4o-mini     | 2024-08-17 01:02:45 | ./workspace/benchmarkingjob/query-routing/493d14ea-5bf1-11ef-bf9b-755996a48c84 |
+------+---------------+-----+-----------+----------------+-----------+------------+---------------------+--------------------------+-------------------+------------------------+---------------------+---------------------+--------------------------------------------------------------------------------+
```

Ianvs will output a `rank.csv` and `selected_rank.csv` in `ianvs/workspace`, which will record the test results of each test.

You can modify the relevant model parameters in `examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/test_queryrouting.yaml`, conduct multiple tests, and compare the results of different configurations.

