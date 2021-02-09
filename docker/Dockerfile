ARG PYTORCH="1.7.1"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

LABEL maintainer='zhaoshu@iie.ac.cn'

RUN apt-get update \
    && apt-get install -y tmux git vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/

ADD nvidia_dali_cuda110-0.30.0-1983575-py3-none-manylinux2014_x86_64.whl /tmp/
RUN pip install loguru pytorch-metric-learning /tmp/nvidia_dali_cuda110-0.30.0-1983575-py3-none-manylinux2014_x86_64.whl && rm /tmp/nvidia_dali_cuda110-0.30.0-1983575-py3-none-manylinux2014_x86_64.whl

WORKDIR /workspace
RUN git clone https://github.com/TreezzZ/Hashing-Pretrain.git
WORKDIR Hashing-Pretrain