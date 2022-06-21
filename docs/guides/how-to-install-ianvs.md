# How to install Ianvs

It is recommended to use Ianvs on Linux machine. But for quick algorith development, windows is also planed to support, to reduce the configuration cost of development environment.  

This guide covers how to install Ianvs on a Linux environment.

## Prerequisites
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

## Install ianvs on Linux


### Create virtualenv
```shell
sudo apt install -y virtualenv
mkdir ~/venv 
virtualenv -p python3 ~/venv/ianvs
source ~/venv/ianvs/bin/activate
```

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




## Install ianvs on Windows

TODO