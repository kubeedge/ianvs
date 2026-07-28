# Use Miniconda base image
FROM continuumio/miniconda3:latest

ENV CONDA_ENV=ianvs-experiment \
    PYTHON_VERSION=3.8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    git \
    unzip 

# Copy kaggle.json (Make sure this file is in the same directory as your Dockerfile)
COPY kaggle.json /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json

# Clone Ianvs repo
RUN git clone https://github.com/kubeedge/ianvs.git
WORKDIR /ianvs

# Create conda environment with Python and Rust
RUN conda create -y -n $CONDA_ENV python=$PYTHON_VERSION rust -c conda-forge

# Install dependencies inside the conda environment and Ianvs
RUN /bin/bash -c "source activate $CONDA_ENV && \
    pip install examples/resources/third_party/sedna-0.6.0.1-py3-none-any.whl && \
    pip install -r requirements.txt && \
    pip install -r examples/cloud-edge-collaborative-inference-for-llm/requirements.txt && \
    python setup.py install"

# Download Kaggle CLI
RUN pip install kaggle

# Download dataset
RUN cd /ianvs && \
    kaggle datasets download -d kubeedgeianvs/ianvs-mmlu-5shot && \
    kaggle datasets download -d kubeedgeianvs/ianvs-gpqa-diamond && \
    unzip -o ianvs-mmlu-5shot.zip && \
    unzip -o ianvs-gpqa-diamond.zip && \
    rm -rf ianvs-mmlu-5shot.zip && \
    rm -rf ianvs-gpqa-diamond.zip

# Set final working directory
WORKDIR /ianvs
