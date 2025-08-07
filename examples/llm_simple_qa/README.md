# README

## Simple QA

### Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Detailed Setup Guide](#detailed-setup-guide)  
  - [Prerequisites](#prerequisites)  
  - [Step 1: Clone the Repository](#step-1-clone-the-repository)  
  - [Step 2: Install Dependencies](#step-2-install-dependencies)  
  - [Step 3: Prepare the Dataset](#step-3-prepare-the-dataset)  
  - [Step 4: Configure the Benchmark](#step-4-configure-the-benchmark)  
  - [Step 5: Run the Benchmark](#step-5-run-the-benchmark)
- [OpenCompass Evaluation](#opencompass-evaluation)
- [Results and Discussion](#results-and-discussion)
- [Contributing](#contributing)
- [License](#license)

---

### Introduction

This example demonstrates how to use the ianvs benchmarking tool to evaluate a basic question-answering model using a small custom dataset. It covers setup, benchmarking, and result interpretation for a simple LLM use case.

---

### Quick Start

1. **Clone the `ianvs` repository:**

    ```bash
    git clone https://github.com/kubeedge/ianvs.git
    cd ianvs/examples/llm_simple_qa
    ```

2. **Prepare the Dataset:**

    Create the `dataset` folder and the required JSONL file using the provided data.

3. **Install Dependencies:** 
 
    Install all necessary Python packages and configure the `PYTHONPATH`.

4. **Run the Benchmark:** 

    Execute the `ianvs` command from the example directory.

---

### Detailed Setup Guide

#### Prerequisites

- Python 3.8+ environment (e.g., using `conda` or `venv`).
  [Create virtualenv](#create-virtualenv)
---

#### Step 1: Clone the Repository

Clone the `ianvs` repository to your local machine and navigate to this example's directory.

```bash
git clone https://github.com/kubeedge/ianvs.git
cd ianvs
```

---

#### Step 2: Install Dependencies

This example requires a few dependencies. We will also set up the `PYTHONPATH` to ensure `ianvs` can find the custom metrics and other modules.

```bash
# Install this example's dependencies
pip install -r examples/llm_simple_qa/requirements.txt

# Install the OpenCompass dependency
pip install examples/resources/opencompass-0.2.5-py3-none-any.whl

# Set the PYTHONPATH to include the project's root for metric discovery
export PYTHONPATH="/mnt/c/Users/ronak/OneDrive/Desktop/ianvs:$PYTHONPATH"
```

---

#### Step 3: Prepare the Dataset

Ensure a reproducible and clean setup by following these steps:

**Data Structure**

The benchmark expects the following structure:

```
.
├── examples
│   └── llm_simple_qa
│       └── dataset
│           └── llm_simple_qa
│               ├── test_data
│               │   └── data.jsonl
│               └── train_data
│                   └── data.jsonl
```

1. **Create the dataset directories:**

```bash
mkdir -p examples/llm_simple_qa/dataset/llm_simple_qa/test_data
mkdir -p examples/llm_simple_qa/dataset/llm_simple_qa/train_data
```

2. **Create an empty training data file:**

```bash
touch examples/llm_simple_qa/dataset/llm_simple_qa/train_data/data.jsonl
```

3. **Create the test data file:**

Manually create and open a file at `dataset/llm_simple_qa/test_data/data.jsonl` and paste the following:

```json
{"question": "If Xiao Ming has 5 apples, and he gives 3 to Xiao Hua, how many apples does Xiao Ming have left?\nA. 2\nB. 3\nC. 4\nD. 5", "answer": "A"}
{"question": "Which of the following numbers is the smallest prime number?\nA. 0\nB. 1\nC. 2\nD. 4", "answer": "C"}
{"question": "A rectangle has a length of 10 centimeters and a width of 5 centimeters, what is its perimeter in centimeters?\nA. 20 centimeters\nB. 30 centimeters\nC. 40 centimeters\nD. 50 centimeters", "answer": "B"}
{"question": "Which of the following fractions is closest to 1?\nA. 1/2\nB. 3/4\nC. 4/5\nD. 5/6", "answer": "D"}
{"question": "If a number plus 10 equals 30, what is the number?\nA. 20\nB. 21\nC. 22\nD. 23", "answer": "A"}
{"question": "Which of the following expressions has the largest result?\nA. 3 + 4\nB. 5 - 2\nC. 6 * 2\nD. 7 ÷ 2", "answer": "C"}
{"question": "A class has 24 students, and if each student brings 2 books, how many books are there in total?\nA. 48\nB. 36\nC. 24\nD. 12", "answer": "A"}
{"question": "Which of the following is the correct multiplication rhyme?\nA. Three threes are seven\nB. Four fours are sixteen\nC. Five fives are twenty-five\nD. Six sixes are thirty-six", "answer": "B"}
{"question": "If one number is three times another number, and this number is 15, what is the other number?\nA. 5\nB. 10\nC. 15\nD. 45", "answer": "A"}
{"question": "Which of the following shapes has the longest perimeter?", "answer": "C"}
```

---

#### Step 4: Configure the Benchmark

Ensure that `benchmarkingjob.yaml` and `testenv.yaml` are using **relative paths** for dataset, metrics, and workspace. No additional edits are needed.

---

#### Step 5: Run the Benchmark

From the `examples/llm_simple_qa` directory:

```bash
ianvs -f examples/llm_simple_qa/benchmarkingjob.yaml
```

---

### OpenCompass Evaluation

This example can also be run using the **OpenCompass** evaluation framework.

#### Prepare Environment

```bash
pip install examples/resources/opencompass-0.2.5-py3-none-any.whl
```

#### Run Evaluation

```bash
python examples/llm_simple_qa/run_op.py examples/llm/singletask_learning_bench/simple_qa/testalgorithms/gen/op_eval.py
```

---

### Results and Discussion

A successful run will produce an output similar to the following, showing a leaderboard with calculated accuracy:

```bash
BaseModel predict
['A' 'C' 'B' 'D' 'A' 'C' 'A' 'B' 'A' 'C']
['A', 'C', 'B', 'B', 'A', 'D', 'A', 'D', 'B', 'A']
+------+-------------------------------+-----+--------------------+-----------+---------------------+------------------------------------------------------------------------------------------------+
| rank |        algorithm              | acc |      paradigm      | basemodel |         time        |                                            url                                                 |
+------+-------------------------------+-----+--------------------+-----------+---------------------+------------------------------------------------------------------------------------------------+
|  1   | simple_qa_singletask_learning | 0.5 | singletasklearning |    gen    | 2025-08-06 11:50:43 | ./workspace/benchmarkingjob/simple_qa_singletask_learning/c1165a5e-72ba-11f0-80d8-00155dc14dbd |
+------+-------------------------------+-----+--------------------+-----------+------------------------------------------------------------------------------------------------+
```

The accuracy score of `0.5` confirms that the framework works correctly. Results are saved in the `workspace` directory.

---

### Contributing

-[Contributing](./CONTRIBUTING.md)
-[Contribution to Documentation](./how-to-cotribute-documentation.md)

### License

-[License](./LICENSE)