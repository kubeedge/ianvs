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
    - [3. Developing CI-CD pipelines](#3-developing-ci-cd-pipelines)
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

3. Expanding Cloud Model Support with Free API Integration

- Currently, the cloud model is limited to GPT-4o-mini, which is a paid option. To expand support, we can integrate additional models using free APIs, such as the [Groq API](https://console.groq.com/docs/models). A new parameter can be added to [test_queryrouting.yaml](https://github.com/kubeedge/ianvs/blob/main/examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/test_queryrouting.yaml) to specify whether to use GPT-4o-mini or alternative models via Groq, providing greater flexibility and cost-effective options.
- To benchmark different models on my custom dataset, I used the Groq API. The updated versions of api_llm.py and cloud_mode.py can be found [here](https://gist.github.com/AryanNanda17/0df5e3d6c6705f4a53b40b2c02c662ff).

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
- [examples/llm_simple_qa](https://github.com/kubeedge/ianvs/tree/82606430abd8828749d7d7725d7f38e92d0c2310/examples/llm_simple_qa)
- [examples/robot/lifelong_learning_bench](https://github.com/kubeedge/ianvs/tree/82606430abd8828749d7d7725d7f38e92d0c2310/examples/robot/lifelong_learning_bench). 

### 3. Developing CI-CD pipelines

In order to ensure backward compatibility I will work on developing GitHub Actions workflows to automatically set up and test the examples. This workflow will be triggered on every push and pull request to the repository. By doing so, it will verify that new changes do not introduce dependency conflicts and that the example continues to function correctly after each update.

## Roadmap

### March 
- Submit a proposal explaining what new changes will be introduced in detail.
- Start working on refining the [cloud-edge-collaborative-inference-for-llm](https://github.com/kubeedge/ianvs/tree/82606430abd8828749d7d7725d7f38e92d0c2310/examples/cloud-edge-collaborative-inference-for-llm) example to fully demonstrate the usage of ianvs

### April 
- Completing refinement of [examples/cloud-edge-collaborative-inference-for-llm](https://github.com/kubeedge/ianvs/tree/82606430abd8828749d7d7725d7f38e92d0c2310/examples/cloud-edge-collaborative-inference-for-llm) by mid term evaluation 
- Start working on testing the other examples and finding errors in it. 
- Developing CI pipelines for few main examples. 

### May
- Correcting all the errors present in old examples.
- Developing CI pipelines for the remaining examples. 