<<<<<<< HEAD
# Ianvs v0.1.0 release

## 1. Release the Ianvs distributed synergy AI benchmarking framework.
   a) Release test environment management and configuration.
   b) Release test case management and configuration.
   c) Release test story management and configuration.
   d) Release the open-source test case generation tool: Use hyperparameter enumeration to fill in one configuration file to generate multiple test cases.

## 2. Release the PCB-AoI public dataset.
Release the PCB-AoI public dataset, its corresponding preprocessing, and baseline algorithm projects. 
Ianvs is the first open-source site for that dataset.

## 3. Support two new paradigms in test environments and test cases. 
   a) Test environments and test cases that support the single-task learning paradigm.
   b) Test environments and test cases that support the incremental learning paradigm.

## 4. Release PCB-AoI benchmark cases based on the two new paradigms.
   a) Release PCB-AoI benchmark cases based on single-task learning, including leaderboards and test reports.
   b) Release PCB-AoI benchmark cases based on incremental learning, including leaderboards and test reports.

# Ianvs v0.2.0 release

This version of Ianvs supports the following functions of unstructured lifelong learning:

## 1. Support lifelong learning throughout the entire lifecycle, including task definition, task assignment, unknown task recognition, and unknown task handling, among other modules, with each module being decoupled.
   - Support unknown task recognition and provide corresponding usage examples based on semantic segmentation tasks in [this example](https://github.com/kubeedge/ianvs/tree/main/examples/robot-cityscapes-synthia/lifelong_learning_bench/semantic-segmentation).
   - Support multi-task joint inference and provide corresponding usage examples based on object detection tasks in [this example](https://github.com/kubeedge/ianvs/tree/main/examples/MOT17/multiedge_inference_bench/pedestrian_tracking).

## 2. Provide classic lifelong learning testing metrics, and support for visualizing test results.
   - Support lifelong learning system metrics such as BWT and FWT.
   - Support visualization of lifelong learning results.
   
## 3. Provide real-world datasets and rich examples for lifelong learning testing, to better evaluate the effectiveness of lifelong learning algorithms in real environments.
   - Provide cloud-robotics datasets in [this website](https://kubeedge-ianvs.github.io/).
   - Provide cloud-robotics semantic segmentation examples in [this example](https://github.com/kubeedge/ianvs/tree/main/examples/robot/lifelong_learning_bench).

# Ianvs v0.3.0 release

## What's New in v0.3.0

Ianvs v0.3.0 brings powerful new LLM-related features, including comprehensive (1) LLM testing and benchmarking tools, (2) advanced cloud-edge collaborative inference paradigms, and (3) innovative algorithms tailored for large model optimization.

### 1. Support for LLM Testing and Benchmarks
Ianvs now supports robust testing for both locally deployed LLMs and public LLM APIs (e.g., OpenAI). This release introduces three specialized benchmarks for evaluating LLM capabilities in diverse scenarios:

- Government-Specific Large Model Benchmark: Designed to assess LLM accuracy and reasoning in [government-specific scenarios](https://github.com/kubeedge/ianvs/blob/main/docs/proposals/scenarios/llm-benchmarks/llm-benchmarks.md). using objective (multiple-choice) and subjective (Q&A) tests. [Explore the benchmark dataset](https://www.kaggle.com/datasets/kubeedgeianvs/the-government-affairs-dataset-govaff/data?select=government_benchmark), [try the example](https://github.com/kubeedge/ianvs/tree/main/examples/government/singletask_learning_bench).

- Smart Coding Benchmark: This benchmark evaluates the debugging capabilities of LLMs using real-world coding issues from GitHub repositories. [Learn more through the example](https://github.com/kubeedge/ianvs/tree/main/examples/smart_coding/smart_coding_learning_bench) and [read the background documentation](https://github.com/kubeedge/ianvs/blob/main/docs/proposals/scenarios/Smart_Coding/Smart%20Coding%20benchmark%20suite%20Proposal.md).

- Large Language Model Edge Benchmark: Focused on testing LLM performance in edge environments, this benchmark evaluates resource efficiency and deployment performance. [Access datasets and examples here](https://github.com/kubeedge/ianvs/tree/main/examples/llm_simple_qa) and check out the [detailed documentation](https://github.com/kubeedge/ianvs/blob/main/docs/proposals/scenarios/llm-benchmark-suite/llm-edge-benchmark-suite.md).

### 2. Enhanced Cloud-Edge Collaborative Inference
This release introduces new paradigms and algorithms for collaborative inference to optimize cloud-edge cooperation and improve performance:

- Cloud-Edge Collaborative Inference Paradigm: A new architecture enables [efficient cloud-edge collaboration for LLM inference](https://github.com/kubeedge/ianvs/blob/main/docs/proposals/algorithms/joint-inference/cloud-edge-collaboration-inference-for-llm.md), featuring a baseline algorithm that delivers up to 50% token cost savings without compromising accuracy. [Try the example](https://github.com/kubeedge/ianvs/tree/main/examples/cloud-edge-collaborative-inference-for-llm).

- Speculative Decoding Algorithm (EAGLE, ICML'24): Integrated within the collaborative inference framework, this algorithm accelerates inference speeds by 20% or more. [Try the example](https://github.com/kubeedge/ianvs/tree/main/examples/cloud-edge-collaborative-inference-for-llm) and explore [detailed documentation](https://github.com/kubeedge/ianvs/blob/main/docs/proposals/algorithms/joint-inference/cloud-edge-speculative-decoding-for-llm.md).

- Joint Inference Paradigm for Pedestrian Tracking: A multi-edge inference paradigm for pedestrian tracking utilizing the pretrained ByteTrack model (ECCV'22). [See the pedestrian tracking example](https://github.com/kubeedge/ianvs/tree/main/examples/MOT17/multiedge_inference_bench/pedestrian_tracking) or refer to the [background documentation](https://github.com/kubeedge/ianvs/blob/main/docs/proposals/algorithms/multi-edge-inference/Heterogeneous%20Multi-Edge%20Collaborative%20Neural%20Network%20Inference%20for%20High%20Mobility%20Scenarios.md).

### 3. Support for New Large Model Algorithms
Ianvs includes new algorithms to improve LLM performance and usability in various scenarios:

- Personalized LLM Agent Algorithm: This algorithm supports single-task learning using the pretrained Bloom model, enabling personalized LLM operations. [Explore the example](https://github.com/kubeedge/ianvs/tree/main/examples/llm-agent/singletask_learning_bench) and review the [documentation](https://github.com/Frank-lilinjie/ianvs/blob/main/docs/proposals/algorithms/single-task-learning/Personalized%20LLM%20Agent%20based%20on%20KubeEdge-Ianvs%20Cloud-Edge%20Collaboration.md).

- Multimodal Large Model Joint Learning Algorithm: A joint learning algorithm for multimodal understanding with the pretrained RFNet model. [Try the example here](https://github.com/aryan0931/ianvs/tree/main/examples/Cloud_Robotics/singletask_learning_bench/Semantic_Segmentation) and learn more in the [documentation](https://github.com/kubeedge/ianvs/blob/main/docs/proposals/scenarios/Cloud_robotics/single_task_learning.md).

- Unseen Task Processing Algorithm: Supports lifelong learning with pretrained models to handle unseen tasks effectively. [Access the example](https://github.com/kubeedge/ianvs/tree/main/examples/cityscapes/lifelong_learning_bench/unseen_task_processing-GANwithSelfTaughtLearning) and gain insights from the [background documentation](https://github.com/kubeedge/ianvs/blob/main/docs/proposals/algorithms/lifelong-learning/Unknown_Task_Processing_Algorithm_based_on_Lifelong_Learning_of_Ianvs.md).
=======
version https://git-lfs.github.com/spec/v1
oid sha256:c034b13314ff7c0d7fb181d671ee87a391bf83b8924b08904104c716830ebb2c
size 7866
>>>>>>> 9676c3e (ya toh aar ya toh par)
