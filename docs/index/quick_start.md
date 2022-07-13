### 0.Check the Environment
For ianvs installation, it requires you:
* 1 VM(one machine is OK)
* 2 CPUs or more 
* 4GB+ free memory
* 10GB+ free disk space
* Internet connection(github, pip etc.)
* Linux platform, such as ubuntu
* Python 3.6+
you can check the python version by the following command:
```
python -V
```
after doing that, the output will be like this, that means your version fits the bill.
```
Python 3.6.9
```

### 1. Install ianvs
**step1: download ianvs project**
```
mkdir -p /ianvs/project/
cd /ianvs/project/
git clone https://github.com/JimmyYang20/ianvs.git    
```

**step2: install third-party dependencies**
```
apt update
apt install libgl1-mesa-glx -y
python -m pip install --upgrade pip
cd ianvs
python -m pip install third_party/*
python -m pip install -r requirements.txt
```

**step3: install ianvs**
```
python setup.py install  
```

### 2.show case: pcb-aoi

**step1: prepare dataset and initial model**

```
mkdir -p /ianvs/dataset/   
cd /ianvs/dataset/  
wget https://kubeedge.obs.cn-north-1.myhuaweicloud.com:443/ianvs/pcb-aoi/dataset.zip
unzip dataset.zip

mkdir -p /ianvs/initial_model/   
cd /ianvs/initial_model/
wget https://kubeedge.obs.cn-north-1.myhuaweicloud.com:443/ianvs/pcb-aoi/model.zip
```

**step2: install fpn algorithm packages**
```
cd /ianvs/project/ianvs/
python -m pip install examples/resources/algorithms/FPN_TensorFlow-0.1-py3-none-any.whl
```

**step3: run benchmarking of pcb-aoi**

```
cd /ianvs/project/ianvs/
ianvs -f examples/pcb-aoi/singletask_learning_bench/benchmarkingjob.yaml
```

**step4: check result of benchmarking**  

you can check the result of benchmarking in the output path(e.g. `/ianvs/singletask_learning_bench/workspace`) defined in the
benchmarking config file(e.g. `benchmarkingjob.yaml`).

### API
todo

### Contributing
todo

### Community
todo