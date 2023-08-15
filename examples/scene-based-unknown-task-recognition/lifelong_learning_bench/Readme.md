# Quick Start about Unknown task recognition

Welcome to Ianvs! Ianvs aims to test the performance of distributed synergy AI solutions following recognized standards, in order to facilitate more efficient and effective development. Quick start helps you to test your algorithm on Ianvs with a simple example of industrial defect detection. You can reduce manual procedures to just a few steps so that you can build and start your distributed synergy AI solution development within minutes.

Before using Ianvs, you might want to have the device ready:

- One machine is all you need, i.e., a laptop or a virtual machine is sufficient and a cluster is not necessary
- 2 CPUs or more
- 4GB+ free memory depends on the algorithm and simulation setting
- 10GB+ free disk space
- Internet connection for GitHub and pip, etc
- Python 3.6+ installed

In this example, we are using the Linux platform with Python 3.7.1. If you are using Windows, most steps should still apply but a few like commands and package requirements might be different.

## Step 1. Ianvs Preparation

First, we download the code of Ianvs. Assuming that we are using `/ianvs` as workspace, Ianvs can be cloned with `Git` as:

```shell
mkdir /ianvs
cd /ianvs #One might use another path preferred

mkdir project
cd project
git clone https://github.com/kubeedge/ianvs.git
```

Then, we install third-party dependencies for ianvs.

```shell
sudo apt-get update
sudo apt-get install libgl1-mesa-glx -y
python -m pip install --upgrade pip

cd ianvs
python -m pip install ./examples/resources/third_party/*
python -m pip install -r requirements.txt
```

We are now ready to install Ianvs.

```shell
python setup.py install
```

## Step 2. Dataset and Model Preparation

Datasets and models can be large. To avoid over-size projects in the GitHub repository of Ianvs, the Ianvs code base does not include origin datasets. Then developers do not need to download non-necessary datasets for a quick start.

```shell
cd /ianvs #One might use another path preferred
mkdir dataset
cd dataset
wget https://kubeedge.obs.cn-north-1.myhuaweicloud.com/ianvs/curb-detection/curb-detection.zip
unzip dataset.zip
```

The URL address of this dataset then should be filled in the configuration file `testenv.yaml`. In this quick start, we have done that for you and interested readers can refer to [testenv.yaml](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for more details.

```shell
cd /ianvs/examples/scene-based-unknown-task-recognition/lifelong_learning_bench/testalgorithms/rfnet
mkdir results
```

Put the model to results.Download [model](https://pan.baidu.com/s/18MA8Gaw7ptpipfLD6Hz6SA) *access code*: 37ff.

The related algorithm is also ready for this quick start.

```shell
export PYTHONPATH=$PYTHONPATH:/ianvs/project/examples/scene-based-unknown-task-recognition/lifelong_learning_bench/testalgorithms/rfnet/RFNet
export PYTHONPATH=$PYTHONPATH:/ianvs/project/examples/scene-based-unknown-task-recognition/lifelong_learning_bench/testalgorithms/rfnet/
```

The URL address of this algorithm then should be filled in the configuration file `algorithm.yaml`. In this quick start, we have done that for you and interested readers can refer to [algorithm.yaml](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for more details.

## Step 3. Ianvs Execution and Presentation

We are now ready to run the ianvs for benchmarking.

```shell
cd /ianvs/project
ianvs -f examples/scene-based-unknown-task-recognition/lifelong_learning_bench/benchmarkingjob.yaml
```

Finally, the user can check the result of benchmarking on the console and also in the output path( e.g. `/ianvs/lifelong_learning_bench/workspace`) defined in the benchmarking config file ( e.g. `benchmarkingjob.yaml`). In this quick start, we have done all configurations for you and the interested readers can refer to [benchmarkingJob.yaml](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for more details.

The final output might look like this:

| rank | algorithm               | accuracy            | samples_transfer_ratio | paradigm         | basemodel | task_definition        | task_allocation        | unseen_sample_recognition      | basemodel-learning_rate | task_definition-origins | task_allocation-origins | unseen_sample_recognition-model_path                         | time                | url                                                          |
| ---- | ----------------------- | ------------------- | ---------------------- | ---------------- | --------- | ---------------------- | ---------------------- | ------------------------------ | ----------------------- | ----------------------- | ----------------------- | :----------------------------------------------------------- | ------------------- | ------------------------------------------------------------ |
| 1    | rfnet_lifelong_learning | 0.30090234155994056 | 0.4535                 | lifelonglearning | BaseModel | TaskDefinitionByOrigin | TaskAllocationByOrigin | UnseenSampleRecognitionByScene | 0.0001                  | ['real', 'sim']         | ['real', 'sim']         | /examples/scene-based-unknown-task-recognition/lifelong_learning_bench/testalgorithms/rfnet/results/Epochofprose17.pth | 2022-10-25 14:50:01 | /ianvs/lifelong_learning_bench/workspace/benchmarkingjob/rfnet_lifelong_learning/1dfff552-542f-11ed-b875-b07b25dd6922 |

This ends the quick start experiment.

# What is next

If any problems happen, the user can refer to [the issue page on Github](https://github.com/kubeedge/ianvs/issues) for help and are also welcome to raise any new issue.

Enjoy your journey on Ianvs!
