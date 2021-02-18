ARG PYTORCH="1.5"
ARG CUDA="10.1"
ARG CUDNN="7"
ARG DALI="nvidia_dali_cuda100-0.30.0-1983576-py3-none-manylinux2014_x86_64.whl"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

LABEL maintainer='zhaoshu@iie.ac.cn'

RUN apt-get update \
    && apt-get install -y tmux git vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/

ADD ${DALI} /tmp/
RUN pip install loguru pytorch-metric-learning onnx /tmp/${DALI} && rm /tmp/${DALI}

WORKDIR /workspace
RUN git clone https://github.com/TreezzZ/Hashing-Pretrain.git
WORKDIR Hashing-Pretrain