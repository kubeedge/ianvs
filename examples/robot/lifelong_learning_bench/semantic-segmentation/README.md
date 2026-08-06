# README

## Simple QA

### Table of Contents

   -[Introduction](#introduction)  
   -[Prerequisites](#prerequisites)  
   -[IAVNS Setup and Installation](#iavns-setup-and-installation)  
   -[Lifelong Learning Benchmark-Simple](#lifelong-learning-benchmark-simple)  
            -[Step 1 - Setup Dataset](#step-1-Setup-dataset)  
            -[Step 2 - Prepare Config](#step-2-Prepare-config)  
            -[Step 3 - Run Benchmarking](#step-3-Benchmarking)  
            -[Step 4 - Output](#step-4-Output)  
   -[Large Vision Model Preparation (Optional)](#large-vision-model-preparation-optional)  
            -[Step 1 - Prepare Directory](#step-1-Prepare-directory)  
            -[Step 2 - SAM Model Installation](#step-2-Sam-model-installation)  
            -[Step 3 - MMCV and MMDetection](#step-3-mmcv-and-mmdetection)  
            -[Step 4 - Download Cache](#step-4-Download-cache)  
            -[Step 5 - Pretrained RFNet Model](#step-5-Pretrained-rfnet-model)  
            -[Step 6 - Run Benchmarking](#step-6-Benchmarking)  
            -[Step 7 - Output](#step-7-Output)  
- [Troubleshooting](#troubleshooting)  
- [License](#license) 

---

### Introduction

Welcome to Ianvs! Ianvs aims to test the performance of distributed synergy AI solutions following recognized standards,
in order to facilitate more efficient and effective development. Quick start helps you to test your algorithm on Ianvs
with a simple example of semantic segmentation based on lifelong learning. You can reduce manual procedures to just a few steps so that you can build and start your distributed synergy AI solution development within minutes.

---

### Prerequisites

Before using Ianvs, you might want to have the device ready:

- One machine is all you need, i.e., a laptop or a virtual machine is sufficient and a cluster is not necessary
- 2 CPUs or more
- 4GB+ free memory, depends on algorithm and simulation setting
- 10GB+ free disk space
- Internet connection for GitHub and pip, etc
- Python 3.6+ installed 

In this example, we are using the Linux platform with **Python 3.9**. If you are using Windows, most steps should still apply but a few like commands and package requirements might be different.

---

### IAVNS Setup and Installation

First, we download the code of Ianvs. Assuming that we are using `/ianvs` as workspace, Ianvs can be cloned with `Git`
as: 
 
```bash
git clone https://github.com/kubeedge/ianvs.git
cd ianvs
``` 

Create a virtualenv

```shell
sudo apt-get install -y virtualenv
mkdir ~/venv 
virtualenv -p python3 ~/venv/ianvs
source ~/venv/ianvs/bin/activate
```

Then, we install third-party dependencies for ianvs.

```shell
sudo apt-get update
sudo apt-get install libgl1-mesa-glx -y
python -m pip install --upgrade pip

python -m pip install ./examples/resources/third_party/*
python -m pip install -r requirements.txt
pip install -r examples/robot/lifelong_learning_bench/semantic-segmentation/requirements.txt
```

We are now ready to install Ianvs.

```shell
python setup.py install  
ianvs -v
```
If the version information is printed, Ianvs is installed successfully. 

---

### Lifelong Learning Benchmark-Simple

To run a simple example of semantic segmentation based on lifelong learning , we will follow the following steps:


#### Step 1 - Setup Dataset

Datasets and models can be large. To avoid over-size projects in the Github repository of Ianvs, the Ianvs code base does
not include origin datasets. Then developers do not need to download non-necessary datasets for a quick start. 
To download the datasets for this example from kaggle, developers can go in:

```shell
download datasets in https://www.kaggle.com/datasets/kubeedgeianvs/cloud-robotics
```
The download may take time , as datasets can be large.


```shell
mkdir datasets
cd datasets
```
**Create a folder named datasets and move the downloaded `.zip` file into the datasets folder you just created. One can transfer the dataset(zip file) to the path, e.g., on a remote Linux system using [XFTP].Once the file is in the datasets directory, run the unzip command to extract its contents**

```shell
unzip [name_of_the_zip_file].zip #usually "unzip archive.zip"
```

Wait for the content to unzip and your datasets directory will turn into this structure:

```
   ├── 1280x760
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   └── val
│   ├── rgb
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── viz
│       ├── test
│       ├── train
│       └── val
├── 2048x1024
│   ├── gtFine
│   │   ├── test
│   │   ├── train
│   │   └── val
│   ├── rgb
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── viz
│       ├── test
│       ├── train
│       └── val
└── 640x480
    ├── gtFine
    │   ├── test
    │   ├── train
    │   └── val
    ├── json
    │   ├── test
    │   ├── train
    │   └── val
    ├── rgb
    │   ├── test
    │   ├── train
    │   └── val
    └── viz
        ├── test
        ├── train
        └── val     
        
```

**The benchmark requires the train_index.txt and test_index.txt files, which are not included in the original dataset. These are automatically generated during the initial run to split the data and ensure reproducibility**


#### Step 2 - Prepare Config

The URL address of this dataset then should be filled in the configuration file ``testenv.yaml``. In this quick start,
we have done that for you and the interested readers can refer to [testenv.yaml](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for more details.

Related algorithm is also ready in this quick start.

```shell
export PYTHONPATH=$PYTHONPATH:/ianvs/project/ianvs/examples/robot/lifelong_learning_bench/semantic-segmentation/testalgorithms/rfnet/RFNet
```

The URL address of this algorithm then should be filled in the configuration file ``algorithm.yaml``. In this quick
start, we have done that for you and the interested readers can refer to [algorithm.yaml](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for more details.

#### Step 3 - Run Benchmarking

We are now ready to run the ianvs for benchmarking.
To run the basic lifelong learning process:

```shell
cd /ianvs
ianvs -f examples/robot/lifelong_learning_bench/semantic-segmentation/benchmarkingjob-simple.yaml
```

#### Step 4 - Output

Finally, the user can check the result of benchmarking on the console and also in the output path(
e.g. `/ianvs/lifelong_learning_bench/workspace`) defined in the benchmarking config file (
e.g. `benchmarkingjob.yaml`). In this quick start, we have done all configurations for you and the interested readers
can refer to [benchmarkingJob.yaml](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for more details.

The final output might look like this:
```
| rank |        algorithm        |      accuracy      |         BWT         |         FWT         |     paradigm     | basemodel |    task_definition    |    task_allocation    | basemodel-learning_rate | basemodel-epochs | task_definition-origins | task_allocation-origins |        time        |                                                               url                                                               |
| :--: | :---------------------: | :----------------: | :-----------------: | :-----------------: | :--------------: | :-------: | :--------------------: | :--------------------: | :---------------------: | :--------------: | :---------------------: | :---------------------: | :-----------------: | :-----------------------------------------------------------------------------------------------------------------------------: |
|  1  | rfnet_lifelong_learning | 0.2970033189775575 | 0.04239649121511442 | 0.02299711942108413 | lifelonglearning | BaseModel | TaskDefinitionByOrigin | TaskAllocationByOrigin |         0.0001         |        1        |   ['front', 'garden']   |   ['front', 'garden']   | 2023-05-24 15:07:57 | /ianvs/lifelong_learning_bench/robot-workspace-bwt/benchmarkingjob/rfnet_lifelong_learning/efdc47a2-f9fb-11ed-8f8b-0242ac110007 | 
```
---

### Large Vision Model Preparation (Optional)

If you want to run the large vision model based cloud-edge collaboration process, then you need to follow the steps below to install the large vision model additionally. If you only want to run the basic lifelong learning process, you can ignore the steps below.

#### Step 1 - Prepare Directory

Use this directory pattern to avoid confusion:

```bash
mkdir /ianvs
cd /ianvs #One might use another path preferred
mkdir project
cd project
```

#### Step 2 - SAM Model Installation

In this example, we use [SAM model](https://segment-anything.com/) as the cloud large vision model. So, we need to install SAM by the following instructions:

```bash
cd /ianvs/project
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything 
python -m pip install -e .
```

Then, we need to download the pretrained SAM model:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

#### Step 3 - MMCV and MMDetection

In order to save the inference result, we need to install mmcv and mmdetection by the following instructions:

```bash
python -m pip install https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/mmcv-2.0.0-cp39-cp39-manylinux1_x86_64.whl
cd /ianvs/project
git clone https://github.com/hsj576/mmdetection.git
cd mmdetection
python -m pip install -v -e .
```

P.S. The mmcv is heavily relying on the versions of the PyTorch and Cuda installed. The installation of mmcv should ref to [this link](https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html#install-with-pip)


#### Step 4 - Download Cache

In case that your computer couldn't run SAM model, we prepare a cache for all the SAM inference results in Cloud-Robotics dataset. You could download the cache from [this link](https://pan.baidu.com/s/1oGGBa8TjZn0ccbznQsl48g?pwd=wpp1) and put the cache file in "/ianvs/project/":

```bash
cp cache.pickle /ianvs/project
```

 By using the cache, you could simulate the edge-cloud joint inference without installing SAM model.

#### Step 5 - Pretrained RFNet Model

Besides that, we also provided you a pretrained RFNet model in [this link](https://pan.baidu.com/s/1h8JnUgr1hfx5QnaFLLkMAg?pwd=jts4), you could use it if you don't want to train the RFNet model from zero. This instruction is optional:

```bash
cd /ianvs/project
mkdir pretrain
cp pretrain_model.pth /ianvs/project/pretrain
in /ianvs/project/ianvs/examples/robot/lifelong_learning_bench/semantic-segmentation/testalgorithms/rfnet/RFNet/utils/args.py set self.resume = '/ianvs/project/pretrain/pretrain_model.pth'
```

#### Step 6 - Run Benchmarking

To run the large vision model based cloud-edge collaboration process:

```shell
cd /ianvs/project/ianvs
ianvs -f examples/robot/lifelong_learning_bench/semantic-segmentation/benchmarkingjob-sam.yaml
```

#### Step 7 - Output

Finally, the user can check the result of benchmarking on the console and also in the output path(
e.g. `/ianvs/lifelong_learning_bench/workspace`) defined in the benchmarking config file (
e.g. `benchmarkingjob.yaml`). In this quick start, we have done all configurations for you and the interested readers
can refer to [benchmarkingJob.yaml](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for more details.

The final output might look like this:
```
| rank |          algorithm          |      accuracy      |    Task_Avg_Acc    |     paradigm     | basemodel |    task_definition    |    task_allocation    | unseen_sample_recognition | basemodel-learning_rate | basemodel-epochs | task_definition-origins | task_allocation-origins | unseen_sample_recognition-threhold | time                |                                                        url                                                        |
| :--: | :-------------------------: | :----------------: | :----------------: | :--------------: | :-------: | :--------------------: | :--------------------: | :-----------------------: | :---------------------: | :--------------: | :---------------------: | :---------------------: | :--------------------------------: | ------------------- | :---------------------------------------------------------------------------------------------------------------: |
|  1  | sam_rfnet_lifelong_learning | 0.7052917006987501 | 0.6258875117354328 | lifelonglearning | BaseModel | TaskDefinitionByOrigin | TaskAllocationByOrigin |     HardSampleMining     |         0.0001         |        1        |   ['front', 'garden']   |   ['front', 'garden']   |                0.95                | 2023-08-24 12:43:19 | /ianvs/sam_bench/robot-workspace/benchmarkingjob/sam_rfnet_lifelong_learning/9465c47a-4235-11ee-8519-ec2a724ccd3e |
```

This ends the quick start experiment.

---

### Troubleshooting

If any problems happen, the user can refer to [the issue page on Github](https://github.com/kubeedge/ianvs/issues) for help and are also welcome to raise any new issue.

Enjoy your journey on Ianvs!

---

### License

- [License](../../LICENSE)

---