# Quick Start

Welcome to Ianvs! Ianvs aims to test the performance of distributed synergy AI solutions following recognized standards, 
in order to facilitate more efficient and effective development. Quick start helps you to test your algorithm on Ianvs 
with a simple example of industrial defect detection. You can reduce manual procedures to just a few steps so that you can 
build and start your distributed synergy AI solution development within minutes. 

Before using Ianvs, you might want to have the device ready: 
- One machine is all you need, i.e., a laptop or a virtual machine is sufficient and a cluster is not necessary
- 2 CPUs or more
- 4GB+ free memory, depends on algorithm and simulation setting
- 10GB+ free disk space
- Internet connection for GitHub and pip, etc
- Python 3.6+ installed
  

In this example, we are using the Linux platform with Python 3.9. If you are using Windows, most steps should still apply but a few like commands and package requirements might be different. 

## Step 1. Ianvs Preparation

First, we download the code of Ianvs. Assuming that we are using `/ianvs` as workspace, Ianvs can be cloned with `Git`
as:

``` shell
mkdir /ianvs
cd /ianvs #One might use another path preferred

mkdir project
cd project
git clone https://github.com/kubeedge/ianvs.git   
```


Then, we install third-party dependencies for ianvs. 
``` shell
sudo apt-get update
sudo apt-get install libgl1-mesa-glx -y
python -m pip install --upgrade pip

cd ianvs 
python -m pip install ./examples/resources/third_party/*
python -m pip install -r requirements.txt
```

We are now ready to install Ianvs. 
``` shell
python setup.py install  
```

## Step 2. Dataset Preparation

Datasets and models can be large. To avoid over-size projects in the Github repository of Ianvs, the Ianvs code base does
not include origin datasets. Then developers do not need to download non-necessary datasets for a quick start.

``` shell
mkdir /data
cd /data
mkdir datasets
cd datasets
python -m pip install kaggle
kaggle datasets download -d hsj576/cloud-robotics
unzip cloud-robotics.zip
```

The URL address of this dataset then should be filled in the configuration file ``testenv.yaml``. In this quick start,
we have done that for you and the interested readers can refer to [testenv.yaml](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for more details.

<!-- Please put the downloaded dataset on the above dataset path, e.g., `/ianvs/dataset`. One can transfer the dataset to the path, e.g., on a remote Linux system using [XFTP].  -->


Related algorithm is also ready in this quick start. 
``` shell
export PYTHONPATH=$PYTHONPATH:/ianvs/project/ianvs/examples/robot-sam/lifelong_learning_bench/testalgorithms/rfnet/RFNet
```

The URL address of this algorithm then should be filled in the configuration file ``algorithm.yaml``. In this quick
start, we have done that for you and the interested readers can refer to [algorithm.yaml](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for more details.

In this example, we use [SAM model](https://segment-anything.com/) as the cloud large model. So, we need to install SAM by the following instructions:

~~~bash
cd /ianvs/project
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything 
python -m pip install -e .
~~~

Then, we need to download the pretrained SAM model:

~~~bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
~~~

In order to save the inference result, we need to install mmcv and mmdetection by the following instructions:

~~~bash
python -m pip install https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/mmcv-2.0.0-cp39-cp39-manylinux1_x86_64.whl
cd /ianvs/project
git clone https://github.com/hsj576/mmdetection.git
cd mmdetection
python -m pip install -v -e .
~~~

In case that your computer couldn't run SAM model, we prepare a cache for all the SAM inference results in Cloud-Robotics dataset. You could download the cache from [this link](https://pan.baidu.com/s/1oGGBa8TjZn0ccbznQsl48g?pwd=wpp1) and put the cache file in "/ianvs/project/":

~~~bash
cp cache.pickle /ianvs/project
~~~

 By using the cache, you could simulate the edge-cloud joint inference without installing SAM model.

Besides that, we also provided you a pretrained RFNet model in [this link](https://pan.baidu.com/s/1h8JnUgr1hfx5QnaFLLkMAg?pwd=jts4), you could use it if you don't want to train the RFNet model from zero. This instruction is optional:

~~~bash
cd /ianvs/project
mkdir pretrain
cp pretrain_model.pth /ianvs/project/pretrain
~~~

## Step 3. Ianvs Execution and Presentation

We are now ready to run the ianvs for benchmarking. 

``` shell
cd /ianvs/project/ianvs
ianvs -f examples/robot-sam/lifelong_learning_bench/benchmarkingjob.yaml
```

Finally, the user can check the result of benchmarking on the console and also in the output path(
e.g. `/ianvs/lifelong_learning_bench/workspace`) defined in the benchmarking config file (
e.g. `benchmarkingjob.yaml`). In this quick start, we have done all configurations for you and the interested readers
can refer to [benchmarkingJob.yaml](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for more details.

The final output might look like this:   


| rank |        algorithm        |      accuracy      |         Task_Avg_Acc         |     paradigm     | basemodel |    task_definition     |    task_allocation     | unseen_sample_recognition | basemodel-learning_rate | basemodel-epochs | task_definition-origins | task_allocation-origins | unseen_sample_recognition-threhold |         time        |                                                               url                                                               |
|:------:|:-------------------------:|:--------------------:|:---------------------:|:------------------:|:-----------:|:------------------------:|:------------------------:|:-------------------------:|:------------------:|:-------------------------:|:-------------------------:|:---------------------:|:---------------------------------------------------------------------------------------------------------------------------------:|-----------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------:|
|  1   | sam_rfnet_lifelong_learning | 0.7052917006987501 | 0.6258875117354328 | lifelonglearning | BaseModel | TaskDefinitionByOrigin | TaskAllocationByOrigin | HardSampleMining |          0.0001         |        1         |   ['front', 'garden']   |   ['front', 'garden']   |   0.95   | 2023-08-24 12:43:19 | /ianvs/sam_bench/robot-workspace/benchmarkingjob/sam_rfnet_lifelong_learning/9465c47a-4235-11ee-8519-ec2a724ccd3e |



This ends the quick start experiment.

# What is next

If any problems happen, the user can refer to [the issue page on Github](https://github.com/kubeedge/ianvs/issues) for help and are also welcome to raise any new issue. 

Enjoy your journey on Ianvs!