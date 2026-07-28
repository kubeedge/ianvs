# Table of Contents 

- [Enhance Dependency Management and Documentation for Ianvs](#enhance-dependency-management-and-documentation-for-ianvs)
  - [Motivation](#motivation)
    - [Background](#background)
    - [Goals](#goals)
      - [Basic Goals](#basic-goals)
      - [Advance Goals](#advance-goals)
  - [Proposal](#proposal)
  - [Details](#details)
    - [1. Updating the cloud-edge-collaborative-inference-for-llm Example](#1-updating-the-cloud-edge-collaborative-inference-for-llm-example)
    - [2. Updating the Documentation for Old Examples](#2-updating-the-documentation-for-old-examples)
      - [Completed Updates](#completed-updates)
      - [Next Steps](#next-steps)
    - [3. Update Ianvs Quick Start Guide on the web doc](#3-update-ianvs-quick-start-guide-on-the-web-doc)
    - [4. Developing CI-CD pipelines](#4-developing-ci-cd-pipelines)
  - [Roadmap](#roadmap)
    - [March](#march)
    - [April](#april)
    - [May](#may)
    
# Enhance Dependency Management and Documentation for Ianvs

## Motivation

### Background

Ianvs is currently grappling with significant dependency management challenges. It lacks a robust system to handle updates and ensure backward compatibility. As Python versions, dependency libraries, and Ianvs features continuously evolve, many existing examples fail to run, resulting in a surge of inquiries in the Issues section like [#170](https://github.com/kubeedge/ianvs/issues/170), [#152](https://github.com/kubeedge/ianvs/issues/152), [#132](https://github.com/kubeedge/ianvs/issues/132), [#106](https://github.com/kubeedge/ianvs/issues/106) . Moreover, new PRs are often merged without being tested against historical examples, making it difficult to guarantee the functionality of past features through manual Code Review alone. There is an urgent need for a more comprehensive CI testing framework to maintain the usability of Ianvs features as the project progresses. Additionally, the online documentation is outdated, which can be quite confusing for new users.

### Goals 

#### Basic Goals 

1. Develop a New Quick Start Example with Comprehensive Documentation: 
- Refine the existing example [examples/cloud-edge-collaborative-inference-for-llm](https://github.com/kubeedge/ianvs/tree/82606430abd8828749d7d7725d7f38e92d0c2310/examples/cloud-edge-collaborative-inference-for-llm) to fully demonstrate the usage of ianvs. This example should illustrate the setup of each module, highlight potential pitfalls, and serve as a blueprint for others to follow.
- The example must be accompanied by comprehensive documentation.

2. Update Documentation for Other Paradigm Usage:
- Paradigms: SingleTaskLearning, LifeLongLearning
- Examples: [examples/llm_simple_qa](https://github.com/kubeedge/ianvs/tree/82606430abd8828749d7d7725d7f38e92d0c2310/examples/llm_simple_qa), [examples/robot/lifelong_learning_bench](https://github.com/kubeedge/ianvs/tree/82606430abd8828749d7d7725d7f38e92d0c2310/examples/robot/lifelong_learning_bench)
- Ensure that the examples for these paradigms are runnable and are supported by detailed documentation.

3. Update the Contributing Guide
- Clearly document the dependency relationship between ianvs and sedna in the Contributing Guide. Explain how to resolve this dependency, and provide the rationale behind it.

#### Advance Goals

1. Ensure other old examples are runnable
2. An implementation of CI/CD that automatically verifies whether existing examples are runnable.

## Proposal

We propose a comprehensive enhancement of Ianvs' dependency management and documentation to address current challenges and improve user experience. This initiative includes updating the existing example of cloud-edge-collaborative-inference-for-llm to serve as a detailed blueprint for users, illustrating module setup and potential pitfalls. We will also verify and correct potential dependency conflicts and issues in the documentation of Ianvs' current examples to ensure they are accurate and up-to-date. Additionally, we will implement a robust CI/CD framework to automatically verify the functionality of existing examples, ensuring they remain runnable and relevant. By refining documentation and clarifying the dependency relationship between Ianvs and Sedna, we aim to provide clear guidance for users and contributors while maintaining backward compatibility as the project evolves.

## Details 

### 1. Updating the cloud-edge-collaborative-inference-for-llm Example

The example "Cloud-Edge Collaborative Inference for LLM" is well-structured. However,
there are a few areas which we can improve to make it a fully functional quick-start guide
with minimal errors.
1. Ease of Setting up the environment:-

- Here, first of all I suggest correcting the missing dependencies errors which are present in this example and are highlighted in my [pre-test report](https://drive.google.com/file/d/1c69pk4u9_kcKkz3GheIizq0tI1BBJ0db/view) for this project. We can include the onnx dependency in [requirements.txt](https://github.com/kubeedge/ianvs/blob/main/examples/cloud-edge-collaborative-inference-for-llm/requirements.txt) file of “Cloud-Edge Collaborative Inference for LLM" example and we can give one additional command `conda install -c conda-forge rust` to run before [downloading the sedna wheel](https://github.com/kubeedge/ianvs/tree/main/examples/cloud-edge-collaborative-inference-for-llm).
- I suggest adding detailed commands for downloading the dataset and workspace folder with cached results and placing it in the correct directory. Currently, we ask
users to [download the dataset](https://github.com/kubeedge/ianvs/tree/main/examples/cloud-edge-collaborative-inference-for-llm#cloudmodel-configuration) and [workspace folder with cached results](https://github.com/kubeedge/ianvs/tree/main/examples/cloud-edge-collaborative-inference-for-llm#cloudmodel-configuration) from
Hugging Face manually and set it up themselves. Instead, we should provide commands to clone the repository, pull large files using git-lfs and then place them at the correct location. This will reduce manual effort for users.

- I also suggest adding a Docker-based setup for this example. We can create a **Dockerfile** that automates the entire setup process for the "Cloud-Edge Collaborative
Inference for LLM" example. This would ensure a seamless and consistent environment across different operating systems, including Windows, macOS, and Linux. By providing a Docker-based setup, users can run the example on their system without worrying about manual dependencies or configuration issues.

2. Backward Compatibility

- To ensure backward compatibility, a GitHub Actions workflow should be added to automatically set up and test the example. This workflow will be triggered on every push and pull request to the repository. By doing so, it will verify that new changes do not introduce dependency conflicts and that the example continues to function correctly after each update.

3. Add `api_provider`, `api_base_url` and `api_key_env` Parameters to cloudmodel in [test_queryrouting.yaml](https://github.com/kubeedge/ianvs/blob/main/examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/test_queryrouting.yaml) file.


- I propose adding two new hyperparameters, `api_provider` and `api_base_url`, under the cloudmodel section of the `test_queryrouting.yaml configuration` file. The updated section would look like this:

```yml
- type: "cloudmodel"
  name: "CloudModel"
  url: "./examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/cloud_model.py"
  hyperparameters:
    - api_provider:  # New parameter: Specifies the API provider
        values:
          - "openai"  # Default value; can be changed to "groq" or others
    - api_base_url:  # New parameter: Base URL for the provider's API
        values:
          - "https://api.openai.com/v1"  # Default; can be changed to "https://api.groq.com/openai/v1"
    - api_key_env:   # Optional addition: Clarifies the API key source
        values:
          - "OPENAI_API_KEY"  # Default; can be changed to "GROQ_API_KEY"
    - model:
        values:
          - "gpt-4o-mini"  # Default; can be changed to provider-specific models
    - temperature:
        values:
          - 0
    - top_p:
        values:
          - 0.8
    - max_tokens:
        values:
          - 512
    - repetition_penalty:
        values:
          - 1.05
    - use_cache:
        values:
          - true
```

These new parameters are needed because currently in [api_llm.py](https://github.com/kubeedge/ianvs/blob/d209bdeca69078073a2481306918bf3f14669f48/examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/models/api_llm.py#L27), the `api_key` and `base_url` are hardcoded to `OPENAI_API_KEY` and `OPENAI_BASE_URL`. 

With the new parameters we can update the `__init__` method in `api_llm.py` to 
```python
def __init__(self, **kwargs) -> None:
        """Initialize the APIBasedLLM class with flexible provider details."""
        BaseLLM.__init__(self, **kwargs)

        # Get API details from kwargs (from YAML) or environment variables
        api_key_env = kwargs.get("api_key_env", "OPENAI_API_KEY")  # Default to OPENAI_API_KEY
        api_key = os.environ.get(api_key_env)  # Fetch key from specified env var
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {api_key_env}")

        base_url = kwargs.get("api_base_url", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
```

The current `cloudmodel` configuration assumes the use of OpenAI’s API (e.g., `gpt-4o-mini`) by relying on environment variables like `OPENAI_API_KEY` and `OPENAI_BASE_URL` in `api_llm.py`. While the system is already compatible with any OpenAI API-compliant model, explicitly adding `api_provider` and `api_base_url` as configurable parameters in the `test_queryrouting.yaml` file would significantly enhance its flexibility and usability. Here’s why:

- **Broader Model Support**: By parameterizing the API provider and its base URL, users can seamlessly switch between different OpenAI API-compatible providers—such as Groq, Together AI, or others—without modifying the underlying Python code. For example, integrating Groq’s free API (e.g., `https://api.groq.com/openai/v1` with models like `llama3-8b-8192`) becomes a simple config change.
- **Cost-Effectiveness**: OpenAI models like `gpt-4o-mini` are paid, whereas alternatives like Groq offer free tiers with high-performance models (e.g., Llama 3.1, Mixtral). Adding these parameters enables users to leverage cost-effective or free options without altering the system’s architecture, aligning with the goal of expanding cloud model support economically.
- **User Convenience**: Currently, switching providers requires setting environment variables or hardcoding changes in `api_llm.py`. With `api_provider` and `api_base_url` in the `test_queryrouting.yaml`, users can define everything in one place—provider, endpoint, and model—making the system more intuitive and reducing setup friction. For instance, a user could specify:

4. Updating the Threshold for Random Routing

- Currently, in [hard_sample_mining.py](https://github.com/kubeedge/ianvs/blob/1dc6706f0f208d08b749ef2cb6e81233cf18bb5c/examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/hard_sample_mining.py#L185), the threshold for random routing is set to 0, causing all queries to be routed to the cloud. To better observe the behavior of random routing, this threshold should be updated to 0.5, ensuring an equal probability of directing queries to either the cloud or the edge.

5. Error handling in this example could be improved by:-

- Explicitly checking prerequisites
- Providing clear, meaningful error messages instead of generic or overly verbose ones
- Offering graceful fallbacks when possible
- Failing fast when critical components are missing.

6. Correcting device = “cuda” assumption

- At many places, the default device is cuda. But `device="cuda"` assumes CUDA is available. If CUDA is not present, the initialization will fail. A better approach would be to use cpu if cuda is not present and print the necessary logs.

7. Including a Resource-Sensitive Router

- This router addresses real-world edge computing constraints by dynamically adapting to device conditions. When an edge device is running hot, low on battery, or
resource-constrained, offloading to cloud preserves device functionality and extends battery life. Conversely, when resources are abundant, processing locally reduces latency and cloud dependencies. This practical approach ensures reliable operation across diverse edge environments and varying workloads.

### 2. Updating the Documentation for Old Examples 

There are multiple dependency conflicts and path errors in different examples that need to be verified to determine whether they are functional.

#### Completed Updates
- So far, I have updated the documentation and verified the installation process for [PCB-AOI Single-Task Learning example](https://github.com/kubeedge/ianvs/tree/main/examples/pcb-aoi). The PRs [#174](https://github.com/kubeedge/ianvs/pull/174), [#182](https://github.com/kubeedge/ianvs/pull/182), [#171](https://github.com/kubeedge/ianvs/pull/171) resolve the path errors and dependency conflicts for this example. 

#### Next Steps 
I will be verifying the setup and updating the documentation for the following examples next:
- [examples/llm_simple_qa](https://github.com/kubeedge/ianvs/tree/82606430abd8828749d7d7725d7f38e92d0c2310/examples/llm_simple_qa).
- [examples/robot/lifelong_learning_bench](https://github.com/kubeedge/ianvs/tree/82606430abd8828749d7d7725d7f38e92d0c2310/examples/robot/lifelong_learning_bench). 

### 3. Update Ianvs Quick Start Guide on the [web doc](https://ianvs.readthedocs.io/en/latest/)

- Currently, the ianvs [quick start guide](https://ianvs.readthedocs.io/en/latest/guides/quick-start.html) uses the example of `PCB-AOI` as a quick start example. The PCB-AOI is no longer suitable as a quick start example. We need to replace the `PCB-AOI` related content with `cloud-edge-collaborative-inference-for-LLM`.
- In order to ensure consistency, we also need to replace the [how-to-test-algorithms](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation), [how-to-config-algorithm](https://ianvs.readthedocs.io/en/latest/user_interface/how-to-config-algorithm.html), [how-to-config-testenv](https://ianvs.readthedocs.io/en/latest/user_interface/how-to-config-testenv.html), [how-to-config-benchmarkingjob](https://ianvs.readthedocs.io/en/latest/user_interface/how-to-config-benchmarkingjob.html) and [how-to-use-ianvs-command-line](https://ianvs.readthedocs.io/en/latest/user_interface/how-to-use-ianvs-command-line.html) pages. Currently, all these pages use the setup of `pcb-aoi` as an example for explanation, so we will replace the explanation with "cloud-edge-collaborative-inference-for-llm" as well.
- We will include a leaderboard for the "cloud-edge-collaborative-inference-for-LLM" example, similar to [this](https://ianvs.readthedocs.io/en/latest/leaderboards/leaderboard-in-semantic-segmentation-of-Cloud-Robotics/leaderboard-of-SAM-based-Edge-Cloud-Collaboration.html), along with a test report like [this](https://ianvs.readthedocs.io/en/latest/proposals/test-reports/testing-single-task-learning-in-industrial-defect-detection-with-pcb-aoi.html).
- We will also add the query routing algorithms we are using in `cloud-edge-collaborative-inference-for-LLM` under the Algorithms section, similar to [this](https://ianvs.readthedocs.io/en/latest/proposals/algorithms/single-task-learning/fpn.html).

### 4. Developing CI-CD pipelines

In order to ensure backward compatibility I will work on developing GitHub Actions workflows to automatically set up and test the examples. This workflow will be triggered on every push and pull request to the repository. By doing so, it will verify that new changes do not introduce dependency conflicts and that the example continues to function correctly after each update.

- Each workflow automates the testing of a specific Ianvs example (e.g., PCB-AoI defect detection, LLM cloud-edge inference) by:

1. Setting up the environment (Python, dependencies, system packages).
2. Preparing required datasets and models.
3. Running the benchmarking job.
4. Validating outputs (e.g., rank.csv or similar metrics).
5. Uploading artifacts for debugging.

- Naming Convention

-> Pattern: test-example-name.yml

-> `<example-name>` is a short, unique identifier for the example (e.g., pcb-aoi, llm-cloud-edge).

-> Prefix `test`- indicates it’s a testing workflow, consistent across examples.

-> Location: All files go in`~/ianvs/.github/workflows/`.

- Examples:
  - test-pcb-aoi.yml (for the PCB-AoI defect detection example).
  - test-llm-cloud-edge.yml (for the LLM cloud-edge collaborative inference example).
  - test-other-example.yml (for other examples).

- Examples:

GitHub Actions workflow for the `pcb-aoi` example. The following file, named `test-pcb-aoi.yml`, will be placed in the `.github/workflows/` folder.
```yml
name: CI-CD Pipeline for Ianvs PCB-AoI Example
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:
jobs:
  test-pcb-aoi:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      - name: Set up Python 3.6
        uses: actions/setup-python@v4
        with:
          python-version: "3.6"
      - name: Install system dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y libgl1-mesa-glx unzip wget
      - name: Upgrade pip
        run: python -m pip install --upgrade pip
      - name: Install Ianvs dependencies
        run: |
          python -m pip install ./examples/resources/third_party/*
          python -m pip install -r requirements.txt
          python -m pip install -r ./examples/pcb-aoi/requirements.txt
      - name: Install Ianvs
        run: python setup.py install
      - name: Prepare dataset
        run: |
          mkdir -p dataset
          wget -O dataset/dataset.zip "https://kubeedge.obs.cn-north-1.myhuaweicloud.com:443/ianvs/pcb-aoi/dataset.zip"
          unzip dataset/dataset.zip -d dataset/
          ls -lh dataset/
      - name: Prepare initial model
        run: |
          mkdir -p initial_model
          wget -O initial_model/model.zip "https://kubeedge.obs.cn-north-1.myhuaweicloud.com:443/ianvs/pcb-aoi/model.zip"
          unzip initial_model/model.zip -d initial_model/
          ls -lh initial_model/
      - name: Install FPN algorithm wheel
        run: python -m pip install examples/resources/algorithms/FPN_TensorFlow-0.1-py3-none-any.whl
      - name: Run Ianvs benchmarking
        run: |
          ianvs -f ./examples/pcb-aoi/singletask_learning_bench/fault_detection/benchmarkingjob.yaml
      - name: Validate results
        run: |
          if [ ! -f "workspace/singletask_learning_bench/benchmarkingjob/fpn_singletask_learning/rank.csv" ]; then
            echo "Error: rank.csv not generated!"
            exit 1
          fi
          grep "f1_score" workspace/singletask_learning_bench/benchmarkingjob/fpn_singletask_learning/rank.csv || echo "Warning: No f1_score metric found"
      - name: Upload artifacts
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: |
            workspace/singletask_learning_bench/benchmarkingjob/
            *.log
```
The above file: setup the environment, prepare dataset and models, runs the benchmarking job, validates the output and then uploads the artifacts for debugging. 

Similarily, workflows for other old examples will be added based on their running instructions. 

Hence, in summary, the goal of developing CI-CD pipelines will be to ensure that:

- For any new submission (pull requests or merges), all historical (existing) examples remain executable.
- For newly submitted examples, there will be a template that contributors adding new examples can refer to it and create the CI-CD workflow for their specific example themselves. This new template "How to Add CI-CD" will placed in [CONTRIBUTION section](https://github.com/kubeedge/ianvs/tree/main/docs/guides) of [ianvs web doc site](https://ianvs.readthedocs.io/en/latest/guides/how-to-contribute-test-environments.html).  

## Roadmap

### March 
- Submit a proposal explaining what new changes will be introduced in detail.
- Start working on refining the [cloud-edge-collaborative-inference-for-llm](https://github.com/kubeedge/ianvs/tree/82606430abd8828749d7d7725d7f38e92d0c2310/examples/cloud-edge-collaborative-inference-for-llm) example to fully demonstrate the usage of ianvs.

### April 
- Completing refinement of [examples/cloud-edge-collaborative-inference-for-llm](https://github.com/kubeedge/ianvs/tree/82606430abd8828749d7d7725d7f38e92d0c2310/examples/cloud-edge-collaborative-inference-for-llm) by mid term evaluation.
- Update Ianvs Quick Start Guide on the [web doc](https://ianvs.readthedocs.io/en/latest/).
- Start working on testing the other examples and finding errors in it. 

### May
- Developing CI pipelines for the examples. 
- Correcting all the errors present in old examples.