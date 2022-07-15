# How to install Ianvs

It is recommended to use Ianvs on a Linux machine. But for quick algorithm development, the Windows platform is also planned to support, to reduce the configuration cost of the development environment.  

This guide covers how to install Ianvs on a Linux environment.

## Prerequisites
- One machine is all you need, i.e., a laptop or a virtual machine is sufficient and a cluster is not necessary
- 2 CPUs or more
- 4GB+ free memory, depends on algorithm and simulation setting
- 10GB+ free disk space
- Internet connection for GitHub and pip, etc
- Python 3.6+ installed

you can check the python version by the following command:
```
python -V
```
after doing that, the output will be like this, which means your version fits the bill.
```
Python 3.6.9
```

## Install ianvs on Linux


### Create virtualenv
```shell
sudo apt-get install -y virtualenv
mkdir ~/venv 
virtualenv -p python3 ~/venv/ianvs
source ~/venv/ianvs/bin/activate
```

> If you prefer conda, you can create a python environment by referring to the [creating steps](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) provided by conda. 

### Download ianvs project
```
cd ~
git clone https://github.com/kubeedge/ianvs.git 
```

### Install third-party dependencies
```
sudo apt-get update
sudo apt-get install libgl1-mesa-glx -y
python -m pip install --upgrade pip

cd ~/ianvs 
python -m pip install ./examples/resources/third_party/*
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
If the version information is printed, Ianvs is installed successfully. 




## About Windows

At the time being, the package requirements of Ianvs are only applicable for Linux, to ensure comprehensive support from the Linux ecosystem and to ease the burden of manual installation for users in Windows.

If you are more used to developing on Windows, you can still do so with remote connections like SSH from Windows connecting to a Linux machine with ianvs installed. Such remote connection is already supported in common Python coding tools like VScode, Pycharm, etc. By doing so, it helps to provide efficient installation and robust functionality of Ianvs.