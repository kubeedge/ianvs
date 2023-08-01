# Implementation of a Class Incremental Learning Algorithm Evaluation System based on Ianvs

## 1 Background

**Semantic Segmentation** is an image segmentation technique used to assign each pixel in an image to a specific semantic category, which can be widely applied in various fields, such as scene understanding, obstacle detection, and so on. The key challenge of this task is to consider both pixel-level details and global semantic information to accurately segment the image.

In recent years, developers often reuse training data multiple times to improve the model's performance. In practical applications, previous training data is often no longer available due to privacy concerns or other reasons. In such cases, if a machine learning model continues to learn new tasks, it often leads to **catastrophic forgetting**. Therefore, we hope the model has the characteristic of **lifelong learning**, where it can learn new knowledge while retaining old knowledge. The same is true of semantic segmentation tasks. The reason why semantic segmentation requires lifelong learning is that during the lifetime of a semantic segmentation system, it needs to process images of different kinds, different scenes, and different datasets, and there may be significant differences between them. Therefore, if the system is unable to engage in lifelong learning, there will be the problem of catastrophic forgetting and poor performance.

In automatic driving scenarios, images may come from different domains, such as different cities, different weather, etc. Therefore, we hope that the model applied to semantic segmentation has the adaptive ability to different data domains, and **domain incremental learning** (a subset of lifelong learning) is a method to achieve this adaptive ability. In domain incremental learning, the model needs to learn how to process different domain data from the new domain, and combine the knowledge of both old and new domains to improve the performance of the model.

The focus of this project is the scenario for domain incremental semantic segmention (Multi-Domain Incremental Learning for Semantic Segmentation, **MDIL-SS**).

However, at present, MDIL-SS algorithms lacks a unified testing environment. In some cases, new algorithms are only test on certain datasets, which is not rigorous. It is in this context that we need to develop a system to perform standardized tests on MDIL-SS algorithms which is increasingly widely used in the industry, and evaluate the effectiveness of these algorithms.

[KubeEdge-Ianvs](https://github.com/kubeedge/ianvs) is a distributed collaborative AI benchmarking project which can perform benchmarks with respect to several types of paradigms (e.g. single-task learning, incremental learning, etc.). This project aims to take advantage of the benchmarking capabilities of ianvs to develop the test system for MDIL-SS algorithms to meet benchmarking requirements for this type of algorithm.

## 2 Goal

As mentioned in `Motivation` section, the goal of this project is to develop a benchmarking evaluation system for MDIL-SS algorithms based on ianvs. Specifically, this project will reproduce the algorithm proposed in the [WACV2022 paper](https://github.com/prachigarg23/MDIL-SS) (an MDIL-SS algorithm) on ianvs, and use three specified datasets (including Cityscapes, SYNTHIA, and the Cloud-Robotic dataset provided by KubeEdge SIG AI) to conduct baseline tests. In addition, a comprehensive test report (including rankings, time, algorithm name, dataset, dataset distribution type, and test metrics, among other details) will be generated.

It should be noted that testing will be executed in the scenario of domain incremental learning, that is, a base model learns and tests on different data domains successively. And class increment will also emerge in the process, so the evaluation system developed in this project can also deal with the class-incremental problem in the domain-incremental scenario.

## 3 Proposal

`Implementation of a Class Incremental Learning Algorithm Evaluation System based on Ianvs` aims to test the performance of MDIL-SS models following recognized standards, to make the development more efficient and productive.

The scope of the system includes

- A test case for lifelong learning semantic segmentation algorithms, in which a test report can be successfully generated following instructions.
- Easy to expand, allowing users to seamlessly integrate existing algorithms into the system for testing.

Targeting users include

- Developers: Quickly test the performance of lifelong learning semantic segmentation algorithms for further optimization.
- Beginners: Familiarize with distributed synergy AI and lifelong learning, among other concepts.

## 4 Design Details

### 4.1 Datasets

This project will use three datasets, namely **Cityscapes**, **SYNTHIA**, and KubeEdge SIG AI's **Cloud-Robotics** dataset (**CS**, **SYN**, **CR**).

Ianvs has already provides [Cityscapes and SYNTHIA datasets](https://github.com/kubeedge/ianvs/blob/main/docs/proposals/algorithms/lifelong-learning/Additional-documentation/curb_detetion_datasets.md). The following two images are examples from them respectively.

|                          CS Example                          |                         SYN Example                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![MDIL-SS](images/OSPP_MDIL-SS_1.png)  |![MDIL-SS](images/OSPP_MDIL-SS_2.png) |

In addition, this project utilizes the CR dataset from KubeEdge.

|                          CR Example                          |
| :----------------------------------------------------------: |
| ![MDIL-SS](images/OSPP_MDIL-SS_3.png) |

The following code is an excerpt from the `train-index-mix.txt` file. The first column represents the path to the original image, and the second column represents the corresponding label image path. This training set consists of 2363 image pairs used for training.

```txt
rgb/train/20220420_garden/00480.png gtFine/train/20220420_garden/00480_TrainIds.png
rgb/train/20220420_garden/00481.png gtFine/train/20220420_garden/00481_TrainIds.png
rgb/train/20220420_garden/00483.png gtFine/train/20220420_garden/00483_TrainIds.png
```

The following code snippet is an excerpt from the `test-index.txt` file, which follows a similar format to the training set. It contains 257 image pairs used for testing.

```txt
rgb/test/20220420_garden/01357.png gtFine/test/20220420_garden/01357_TrainIds.png
rgb/test/20220420_garden/01362.png gtFine/test/20220420_garden/01362_TrainIds.png
rgb/test/20220420_garden/01386.png gtFine/test/20220420_garden/01386_TrainIds.png
rgb/test/20220420_garden/01387.png gtFine/test/20220420_garden/01387_TrainIds.png
```

As shown in the table below, this dataset contains 7 groups and 30 classes.

|    Group     |                           Classes                            |
| :----------: | :----------------------------------------------------------: |
|     flat     |               road · sidewalk · ramp · runway                |
|    human     |                        person · rider                        |
|   vehicle    |       car · truck · bus · train · motorcycle · bicycle       |
| construction |  building · wall · fence · stair · curb · flowerbed · door   |
|    object    | pole · traffic sign · traffic light · CCTV camera · Manhole · hydrant · belt · dustbin |
|    nature    |                     vegetation · terrain                     |
|     sky      |                             sky                              |

### 4.2 Overall Design

The development consists of two main parts, which are **test environment (test env)** and **test algorithms**.

Test environment can be understood as an exam paper, which specifies the dataset, evaluation metrics, and the number of increments used for testing. It is used to evaluate the performance of the "students". And test algorithms can be seen as the students who will take the exam.

![MDIL-SS](images/OSPP_MDIL-SS_4.png)

In addition, `benchmarkingjob.yaml` is used for integrating the configuration of test env and test algorithms, and is a necessary Ianvs configuration file.

For test env, the development work mainly focuses on the implementation of `mIoU.py`. And for test algorithms, development is concentrated on `basemodel.py`, as shown in the picture below.

![MDIL-SS](images/OSPP_MDIL-SS_5.png)

### 4.3 Test Environment

The following code is the `testenv.yaml` file designed for this project. 

As a configuration file for test env, it contains the 3 aspects, which are the dataset and the number of increments, model validation logic, and model evaluation metrics.

```yaml
# testenv.yaml

testenv:

  # 1
  dataset:
    train_url: "/home/QXY/ianvs/dataset/mdil-ss-dataset/train_data/index.txt"
    test_url: "/home/QXY/ianvs/dataset/mdil-ss-dataset/test_data/index.txt"
    using: "CS SYN CR"
  incremental_rounds: 3
  
  # 2
  model_eval:
    model_metric:
      name: "mIoU"
      url: "/home/QXY/ianvs/examples/mdil-ss/testenv/mIoU.py"
    threshold: 0
    operator: ">="

  # 3
  metrics:
    - name: "mIoU"
      url: "/home/QXY/ianvs/examples/mdil-ss/testenv/mIoU.py"
    - name: "BWT"
    - name: "FWT"
```

After each round of lifelong learning, the model will be evaluated on the validation set. In this project, **mIoU** (mean Intersection over Union) is used as the evaluation metric. If the model achieves an mIoU greater than the specified threshold on the validation set, the model will be updated. 

**BWT** (Backward Transfer) and **FWT** (Forward Transfer) are two important concepts in the field of lifelong learning. BWT refers to the impact of previously learned knowledge on the learning of the current task, while FWT refers to the impact of the current task on the learning of future tasks. Along with mIoU, they serve as testing metrics to assess the lifelong learning capability of the model in semantic segmentation. Functions related to BWT and FWT have already been implemented in [ianvs repository](https://github.com/kubeedge/ianvs/blob/main/core/testcasecontroller/metrics/metrics.py).

### 4.4 Test Algorithm

The following code is the `mdil-ss_algorithm.yaml` file designed for this project. 

```yaml
# mdil-ss_algorithm.yaml

algorithm:
  paradigm_type: "incrementallearning"
  
  incremental_learning_data_setting:
    train_ratio: 0.8
    splitting_method: "default"
  
  modules:
    - type: "basemodel"

      # 1
      name: "ERFNet"
      url: "/home/QXY/ianvs/examples/mdil-ss/testalgorithms/mdil-ss/basemodel.py"
      
      # 2
      hyperparameters:
        - learning_rate:
            values:
              - 0.01
              - 0.0001
        - epochs:
            values:
              - 5
              - 10
        - batch_size:
            values:
              - 10
              - 20
```

First, `basemodel.py`, which involves encapsulating various functional components of the model, including its architecture, layers, and operations, which is the focus of development.

Second, **hyperparameters** setting for the model is also defined in this yaml file. In addition, the evaluation system can perform tests with multiple combinations of hyperparameters at once by configuring multiple hyperparameters in `mdil-ss_algorithm.yaml`.

### 4.5 Test Report

The test report is designed as follows, which contains the ranking, algorithm name, three metrics, dataset name, base model, three hyperparameters, and time.

| Rank | Algorithm | mIoU   | BWT   | FWT   | Paradigm         | Round | Dataset   | Basemodel | Learning_rate | Epoch | Batch_size | Time                |
| ---- | :-------: | ------ | ----- | ----- | ---------------- | ----- | --------- | --------- | ------------- | ----- | ---------- | ------------------- |
| 1    |  MDIL-SS  | 0.6521 | 0.075 | 0.021 | Lifelonglearning | 3     | CS SYN CR | ERFNet    | 0.0001        | 1     | 10         | 2023-05-28 17:05:15 |

## 6 Roadmap

### 6.1 Phase 1 (July 1st - August 15th)

1. Engage in discussions with the project mentor and the community to finalize the development details.

2. Further refine the workflow of the MDIL-SS testing task, including the relationships between different components and modules.

3. Develop the test environment, including datasets and model metrics.

4. Begin the development of the base model encapsulation for the test algorithms.

### 6.2 Phase 2 (August 16th - September 30th)

1. Summarize the progress of Phase 1 and generate relevant documentation.
2. Complete the remaining development tasks, including models, test reports, etc. 
3. Generate initial algorithm evaluation reports.
4. Engage in discussions with the project mentor and the community to further supplement and improve the project.
5. Organize the project code and related documentation, and merge them into the Ianvs repository.
6. Upon merging into the repository, explore new research areas and produce additional outcomes based on this project (e.g., research papers).