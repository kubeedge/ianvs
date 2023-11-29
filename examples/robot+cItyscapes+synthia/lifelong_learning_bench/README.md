# Quick Start about Class Incremental Semantic Segmentation

Welcome to Ianvs! Ianvs aims to test the performance of distributed synergy AI solutions following recognized standards, 
in order to facilitate more efficient and effective development. This  semantic segmentation scenario quick start guides you how to test your class incremental algorithm on Ianvs. You can reduce manual procedures to just a few steps so that you can 
build and start your distributed synergy AI solution development within minutes. 

Before using Ianvs, you might want to have the device ready: 
- One machine is all you need, i.e., a laptop or a virtual machine is sufficient and a cluster is not necessary
- 2 CPUs or more
- 4GB+ free memory, depends on algorithm and simulation setting
- 10GB+ free disk space
- Internet connection for GitHub and pip, etc
- Python 3.6+ installed


In this example, we are using the Linux platform with Python 3.8. If you are using Windows, most steps should still apply but a few like commands and package requirements might be different. 

## Step 1. Ianvs Preparation

First, we download the code of Ianvs. Assuming that we are using `/ianvs` as workspace, Ianvs can be cloned with `Git`
as:

``` shell
mkdir /ianvs
cd /ianvs # One might use another path preferred

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
mkdir dataset
cd dataset
unzip mdil-ss.zip
```

The URL address of this dataset then should be filled in the configuration file ``testenv.yaml``. In this quick start,
we have done that for you and the interested readers can refer to [testenv.yaml](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for more details.


Related algorithm is also ready in this quick start. 

``` shell
export PYTHONPATH=$PYTHONPATH:/ianvs/project/examples/class_increment_semantic_segmentation/lifelong_learning_bench/testalgorithms/erfnet/ERFNet
```

The URL address of this algorithm then should be filled in the configuration file ``algorithm.yaml``. In this quick
start, we have done that for you and the interested readers can refer to [algorithm.yaml](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for more details.


## Step 3. Ianvs Execution and Presentation

We are now ready to run the ianvs for benchmarking. 

``` shell
cd /ianvs/project
ianvs -f examples/class_increment_semantic_segmentation/lifelong_learning_bench/benchmarkingjob.yaml
```

Finally, the user can check the result of benchmarking on the console and also in the output path(
e.g. `/ianvs/project/ianvs-workspace/mdil-ss/lifelong_learning_bench`) defined in the benchmarking config file (
e.g. `benchmarkingjob.yaml`). In this quick start, we have done all configurations for you and the interested readers
can refer to [benchmarkingJob.yaml](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for more details.

The final output might look like this:   

| rank |        algorithm         |          Task_Avg_Acc     |         BWT          |         FWT          |     paradigm     | basemodel |    task_definition     |    task_allocation     | basemodel-learning_rate | basemodel-epochs |           task_definition-origins           |           task_allocation-origins           |         time        |                                                                url                                                               |
|:----:|:------------------------:|:--------------------:|:--------------------:|:--------------------:|:----------------:|:---------:|:----------------------:|:----------------------:|:-----------------------:|:----------------:|:-----------------------------------------:|:-----------------------------------------:|:-------------------:|:-------------------------------------------------------------------------------------------------------------------------------:|
|  1   | erfnet_lifelong_learning |  0.027414088670437726 | 0.010395591126145793 | 0.002835451693721201 | lifelonglearning | BaseModel | TaskDefinitionByDomain | TaskAllocationByDomain |         0.0001          |        1         | ['Cityscapes', 'Synthia', 'Cloud-Robotics'] | ['Cityscapes', 'Synthia', 'Cloud-Robotics'] | 2023-09-26 20:13:21 | ./ianvs-workspace/mdil-ss/lifelong_learning_bench/benchmarkingjob/erfnet_lifelong_learning/3a8c73ba-5c64-11ee-8ebd-b07b25dd6922 |


In addition, in the log displayed at the end of the test, you can see the accuracy of known and unknown tasks in each round, as shown in the table below (in the testing phase of round 3, all classes are seen).


| Round | Seen Class Accuracy | Unseen Class Accuracy |
|:-----:|:---------------------:|:-------------------:|
|   1   |        0.176         |       0.0293        |
|   2   |        0.203         |       0.0265        |
|   3   |        0.311         |       0.0000        |



This ends the quick start experiment.

# What is next

If any problems happen, the user can refer to [the issue page on Github](https://github.com/kubeedge/ianvs/issues) for help and are also welcome to raise any new issue. 

Enjoy your journey on Ianvs!