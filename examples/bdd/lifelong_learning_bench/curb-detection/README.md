# Quick Start for multi-task inference

Welcome to Ianvs! Ianvs aims to test the performance of distributed synergy AI solutions following recognized standards,  in order to facilitate more efficient and effective development. Quick start helps you to test your algorithm on Ianvs with a simple example of industrial defect detection. You can reduce manual procedures to just a few steps so that you can build and start your distributed synergy AI solution development within minutes. 

Before using Ianvs, you might want to have the device ready: 
- One machine is all you need, i.e., a laptop or a virtual machine is sufficient and a cluster is not necessary
- 2 CPUs or more
- 4GB+ free memory, depends on algorithm and simulation setting
- 10GB+ free disk space
- Internet connection for GitHub and pip, etc
- Python 3.6+ installed
  

In this example, we are using the Linux platform with Python 3.6.9. If you are using Windows, most steps should still apply but a few like commands and package requirements might be different. 

## Step 1. Ianvs Preparation

First, we download the code of Ianvs. Assuming that we are using `/ianvs` as workspace, Ianvs can be cloned with `Git` as:

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

Datasets and models can be large. To avoid over-size projects in the Github repository of Ianvs, the Ianvs code base does not include origin datasets. Then developers do not need to download non-necessary datasets for a quick start.

``` shell
cd /ianvs #One might use another path preferred
mkdir dataset   
cd dataset
mkdir bdd
```

The url of bdd is BDD: https://bdd-data.berkeley.edu/, we download the dataset and put it in bdd

The URL address of this dataset then should be filled in the configuration file ``testenv.yaml``. In this quick start,
we have done that for you and the interested readers can refer to [testenv.yaml](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for more details.

<!-- Please put the downloaded dataset on the above dataset path, e.g., `/ianvs/dataset`. One can transfer the dataset to the path, e.g., on a remote Linux system using [XFTP].  -->


Related algorithm is also ready in this quick start. 
``` shell
export PYTHONPATH=$PYTHONPATH:/ianvs/project/examples/bdd/lifelong_learning_bench/curb-detection/testalgorithms/rfnet/RFNet
```

The URL address of this algorithm then should be filled in the configuration file ``algorithm.yaml``. In this quick
start, we have done that for you and the interested readers can refer to [algorithm.yaml](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for more details.

## Step 3. Model Preparation

### Yolov5 model Preparation

1. We  install mmcls: https://mmclassification.readthedocs.io/zh_CN/latest/install.html
2. We download the code of yolov5:

```
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

3. We modify the yolo_hub_path in `basemodel.py` 

```
yolo_hub_path= '/home/yourname/.cache/torch/hub/ultralytics_yolov5_master'
```

4. We replace the `model/commom.py ` under yolo_hub_path with `examples/resources/algorithms/common.py`

### Selector Model Preparation

First, we config the model to select which models will be chose

- We download the checkpoint_file from [百度网盘](https://pan.baidu.com/s/1hCbjrSW7A0J8tgfc-s5R1g), 提取码:d1b5,  which is the weight file for model selector
- The checkpoint_file need to be put under `bdd/lifelong_learning_bench/testalgorithms/yolo/model_selector/`

Then, we config the models which could be selected to realize multi-joint inference 

- We download the models from  [百度网盘](https://pan.baidu.com/s/1HE10JVbQgnam264f4m57Nw), 提取码：p21x
- The file called `yolo_model` need to be put under `examples/resources/`

### Sedna preparation

We replace the file in `yourpath/anaconda3/envs/ianvs/lib/python3.7/site-packages/sedna` with `examples/resources/sedna.zip`  



## Step 4. Ianvs **Execution and Presentation**

We are now ready to run the ianvs for benchmarking. 

``` shell
cd /ianvs/project
ianvs -f examples/bdd/lifelong_learning_bench/curb-detection/benchmarkingjob.yaml
```

Finally, the user can check the result of benchmarking on the console and also in the output path(
e.g. `/ianvs/lifelong_learning_bench/workspace`) defined in the benchmarking config file (
e.g. `benchmarkingjob.yaml`). In this quick start, we have done all configurations for you and the interested readers
can refer to [benchmarkingJob.yaml](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for more details.

The final output might look like this:   

| rank |                 algorithm                 | map  |     paradigm     | basemodel |    task_allocation     | task_remodeling | inference_integrate |
| :--: | :---------------------------------------: | :--: | :--------------: | :-------: | :--------------------: | --------------- | :-----------------: |
|  1   | yolo_lifelong_learning_five_model_750data | 0.5  | lifelonglearning | BaseModel | TaskAllocationByOrigin | TaskRemodeling  | InferenceIntegrate  |

This ends the quick start experiment.

# What is next

If any problems happen, the user can refer to [the issue page on Github](https://github.com/kubeedge/ianvs/issues) for help and are also welcome to raise any new issue. 

Enjoy your journey on Ianvs!