 

---

# Quick Start about Single Task Learning Semantic Segmentation

Welcome to Ianvs! Ianvs is a benchmarking platform designed to evaluate the performance of distributed synergy AI solutions in accordance with recognized standards. This quick start guide will help you test your **Single Task Learning (STL)** algorithm on Ianvs. By following these streamlined steps, you can efficiently develop and benchmark your solution within minutes.

### **Prerequisites**  
Before using Ianvs, ensure that your system meets the following requirements:  
- A single machine (a laptop or a virtual machine is sufficient; no cluster is needed)  
- At least 2 CPUs  
- 4GB+ of free memory (depending on the algorithm and simulation settings)  
- 10GB+ of free disk space  
- An internet connection for accessing GitHub, pip, etc.  
- Python 3.6+ installed  

This guide assumes you are using **Linux** with Python 3.8. If you’re on Windows, most steps will apply, but some commands and package requirements may differ.  

---

## Step 1. Ianvs Installation  

### Clone Ianvs  
First, set up a workspace and clone Ianvs:  
```shell
mkdir /ianvs
cd /ianvs

mkdir project
cd project
git clone https://github.com/kubeedge/ianvs.git
```  

### Install Dependencies  
Next, install the required third-party dependencies:  
```shell
sudo apt-get update
sudo apt-get install libgl1-mesa-glx -y
python -m pip install --upgrade pip

cd ianvs
python -m pip install ./examples/resources/third_party/*
python -m pip install -r requirements.txt
```  

### Install Ianvs  
Finally, install Ianvs:  
```shell
python setup.py install
```  

---

## Step 2. Dataset Preparation  

### Cloud-Robotics Dataset Summary

The **Cloud-Robotics Dataset** features **annotated real-world images** with **dense semantic and instance segmentation** across **30 classes** in 7 groups (e.g., vehicles, humans, nature, objects). It includes polygonal annotations, diverse daytime scenes, dynamic objects, and varying layouts. Data is provided in a JSON format, making it ideal for pixel-level semantic labeling and benchmarking vision models for robotics.

Organize the dataset for STL as shown below:  

```plaintext
Dataset/
├── 1280x760
│   ├── gtFine
│   │   ├── train
│   │   ├── test
│   │   └── val
│   ├── rgb
│   │   ├── train
│   │   ├── test
│   │   └── val
│   └── viz
│       ├── train
│       ├── test
│       └── val
├── 2048x1024
│   ├── gtFine
│   │   ├── train
│   │   ├── test
│   │   └── val
│   ├── rgb
│   │   ├── train
│   │   ├── test
│   │   └── val
│   └── viz
│       ├── train
│       ├── test
│       └── val
├── 640x480
    ├── gtFine
    │   ├── train
    │   ├── test
    │   └── val
    ├── json
    │   ├── train
    │   ├── test
    │   └── val
    ├── rgb
    │   ├── train
    │   ├── test
    │   └── val
    └── viz
        ├── train
        ├── test
        └── val
```  

### Dataset Preparation Command  
```shell
mkdir dataset
cd dataset
unzip dataset.zip
```  

Update the dataset's **URL address** in the `testenv.yaml` configuration file. More details can be found in the [testenv.yaml guide](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation).  

---

## Step 3. Configure Algorithm  
 

Update the algorithm's **URL address** in the `algorithm.yaml` file. Refer to the [algorithm.yaml guide](https://ianvs.readthedocs.io/en/latest/guides/how-to-test-algorithms.html#step-1-test-environment-preparation) for detailed instructions.  

---

## Step 4. Ianvs Execution and Results  

Run Ianvs for benchmarking:  
```shell
cd /ianvs/project
ianvs -f examples/cloud-robotics/single_task_learning/semantic-segmentation/benchmarkingjob.yaml
```   

---
