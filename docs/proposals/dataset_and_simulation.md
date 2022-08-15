# Benchmarks for Edge-cloud Collaborative Lifelong Learning

Artificial intelligence technology has served us in all aspects of life, especially in image, video, voice, recommendation system, etc. It has brought breakthrough results. AI Benchmark is designed to measure the performance and efficacy of AI models. There are already many authoritative AI Benchmark systems in the traditional AI field. As an emerging direction of edge AI, the corresponding Benchmark development is not perfect, and there are generally problems such as insufficient paradigm coverage, inactive communities, no edge/cloud side involved, and few test samples.
The specific research results of AI Benchmark for cloud-side-device collaboration can be found in this [document(in chinese)](https://github.com/iszhyang/AI-Benchmark-for-Cloud-Edge-Device).

As a very promising community in edge AI Benchmarking, ianvs still has the following problems to be solved

- The automatic dataset download and management features are missing, and users still need tedious operations to use the dataset.
- Repeated deployment of the edge-cloud collaboration environment is too heavy. The distributed collaborative system simulation feature of ianvs needs to be developed urgently.

This proposal provides features in datasets, edge-cloud collaborative AI system simulation, etc. for the current version of Ianvs

## Goals

The project is committed to building an edge-cloud collaborative AI Bechmark for edge-cloud collaborative lifelong detection in the Kubeedge open source community.

- Provide downloadable open source datasets and data processing scripts for the difficulty in obtaining datasets and supporting algorithms
- Provides industrial-grade distributed collaborative system simulation for the problem that the repeated deployment of edge-cloud collaboration is too heavy
- In view of the high cost of solution selection and the obscure problem of value, it supports the calculation and ranking of edge-cloud collaborative AI algorithm indicators

## Proposal

### Edge-Cloud Collaborative AI Dataset and Corresponding Data Processing Script

It will provide an open-source edge-cloud collaborative AI dataset management module that is convenient for algorithm developers to use, which can be used as a plug-in for Sedna. The following functions will be supported:

- Dataset management of edge-cloud collaborative AI
- Quick download and use of datasets

### Industrial Distributed Collaborative System Simulation

It will provide Ianvs with docker in docker system emulation features.

## Design Details

### Dataset automatic download function

The current `testenv.yaml` file is as follows

```yaml
testenv:
  # dataset configuration
  dataset:
    # the url address of train dataset index; string type;
    train_url: "/ianvs/dataset/train_data/index.txt"
    # the url address of test dataset index; string type;
    test_url: "/ianvs/dataset/test_data/index.txt"

  # metrics configuration for test case's evaluation; list type;
  metrics:
      # metric name; string type;
    - name: "f1_score"
      # the url address of python file
      url: "./examples/pcb-aoi/singletask_learning_bench/testenv/f1_score.py"
```

For the convenience of developers, the expected testenv.yaml file is as follows:

- The user fills in the `dataset_url`, here is the HUAWEI CLOUD OBS link.
- The user fills in `dataset_path`, where the dataset wants to be saved.

```yaml
testenv:
  # dataset configuration
  dataset:
    # the url address of dataset with zip file; string type;
    dataset_url: "https://kubeedge.obs.cn-north-1.myhuaweicloud.com:443/ianvs/pcb-aoi/dataset.zip"
    # the url address of saving dataset
    dataset_path: "./dataset/"

  # metrics configuration for test case's evaluation; list type;
  metrics:
      # metric name; string type;
    - name: "f1_score"
      # the url address of python file
      url: "./examples/pcb-aoi/singletask_learning_bench/testenv/f1_score.py"
```

The expected implementation method is to modify `dataset.py` in the `testenvmanager` module as follows:

1. Add `self.dataset_url`, `self.dataset_path`
2. Add `download_save()` function

![](https://github.com/iszhyang/images/blob/main/ianvs/dataset-download.png?raw=true)

### Semantic segmentation dataset and corresponding algorithm

dataset: [cityscapes](https://www.cityscapes-dataset.com)
Baseline algorithm: [RFNet](https://github.com/AHupuJR/RFNet)

We will provide the cityscapes dataset and corresponding methods for ianvs. The specific operation is as follows:

- dataset processing
  - generate the index file for train and test
- dataset download link
  - provide the download link of cityscapes dataset in HUAWEI Cloud OBS.
- Baseline algorithm
  ![](https://github.com/iszhyang/images/blob/main/ianvs/rfnet.png?raw=true)

### Industrial Distributed Collaborative System Simulation

Provide ianvs with the feature of industrial distributed system simulation.
![img](https://github.com/iszhyang/images/blob/main/ianvs/simulation.jpg?raw=true)

- The `System Config` denotes the system config of the current test case, such as the `number of edge nodes`.
- The `Simulation Controller` is the core module of system simulation, including `Paradigm Management` and `Enviroment Management`.
  - `Paradigm Management` is used to manage the paradigm in simulation, including the following
    - `Incremental Learning Management`
    - `Lifelong Learning Management`
    - etc.
  - `Enviroment Management` is used to build and manage the simulation enviroments.
    - `Enviroment Builder Controller` is used to parse the system config(simulation), build the simulation enviroment, and finally close the enviroment completely.
    - `Container Management` manage the containers used to build the simulation enviroment.
    - `Simulation Job Controller`is responsible for building and managing the job required in the simulation environment.

![img](https://github.com/iszhyang/images/blob/main/ianvs/simulation-dataflow.jpg?raw=true)
In the data flow diagram above, the expected flow is as follows:
1. user start the benchmarkjob
2. check whether to start emulation based on `system config`
3. parse simulation config and build the simulation enviroment
4. (3&&4) start the container mgmt
5. build the simulation jobs
6. (5&&6) build paradigm mgmt
7. run the simulation job
8. get the repoter of current test case

## Roadmap

The roadmap would be as follows

### July

1. Community proposal.
2. Dataset Download and Use (OBS).
3. Research on AI Benchmark for cloud-edge-device collaboration.

### August

1. (0801-0814) Attempt to design the `simulation controller`.
2. (0815-0821) Adjust and modify the architecture diagram according to the opinions of the community.
3. (0822-0829) Clarify the flow chart according to the architecture diagram.
4. (0830-0831) Start implementing the preliminary framework through code.

### September

1. (0901-0904) Start implementing the preliminary framework through code.
2. (0905-0911) Improve the code and verify the reliability.
3. (0912-0918) Prepare the final PR.
2. (0919-0930) Linkage integration with other projects
