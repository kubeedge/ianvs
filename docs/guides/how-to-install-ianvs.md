# How to install Ianvs

It is recommended to use Ianvs on Linux machine. But for quick algorith development, windows is also planed to support, to reduce the configuration cost of development environment.  

This guide covers how to install Ianvs on a Linux environment.

## Prerequisites
- one machine is all you need, i.e., a laptop or a virtual machine is sufficient and cluster is not necessary
- 2 CPUs or more
- 4GB+ free memory, depends on algorithm and simulation setting
- 10GB+ free disk space
- internet connection for github and pip, etc
- Python 3.6+ installed

you can check the python version by the following command:
```
python -V
```
after doing that, the output will be like this, that means your version fits the bill.
```
Python 3.6.9
```

## Install ianvs on Linux


### Create virtualenv
```shell
sudo apt install -y virtualenv
mkdir ~/venv 
virtualenv -p python3 ~/venv/ianvs
source ~/venv/ianvs/bin/activate
```

> If you prefer conda, you can create a python environment by referring to the [creating steps](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) provided by conda. 

### Download ianvs project
```
cd ~
git clone https://github.com/JimmyYang20/ianvs.git    
```

### Install third-party dependencies
```
sudo apt update
sudo apt install libgl1-mesa-glx -y
cd ~/ianvs
python -m pip install third_party/*
python -m pip install -r requirements.txt
```

### Install ianvs 
```
python setup.py install  
```

### Check the installation
```shell
ianvs -v
```
If the version information is printed, Ianvs is installed successful. 




## About Windows

If you are used to develop on Windows, it is still suggested that you run ianvs on Linux to reduce dependency installation time, and get better running efficiency. 