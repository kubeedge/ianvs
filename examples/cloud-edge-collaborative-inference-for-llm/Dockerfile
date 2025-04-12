# Use Miniconda base image
FROM continuumio/miniconda3:latest

ENV CONDA_ENV=ianvs-experiment \
    PYTHON_VERSION=3.8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    git \
    git-lfs && \
    git lfs install

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

# Download and move dataset (still run inside /ianvs)
RUN cd /ianvs && \
    git clone https://huggingface.co/datasets/FuryMartin/Ianvs-MMLU-5-shot && \
    git lfs install && \
    cd Ianvs-MMLU-5-shot && \
    git lfs pull && \
    mkdir -p /ianvs/dataset && \
    mv mmlu-5-shot /ianvs/dataset/ && \
    mv workspace-mmlu /ianvs/ && \
    rm -rf Ianvs-MMLU-5-shot  # Optional cleanup

# Set final working directory
WORKDIR /ianvs
