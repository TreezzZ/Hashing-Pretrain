ARG PYTORCH="1.7.1"
ARG CUDA="11.0"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

LABEL maintainer='zhaoshu@iie.ac.cn'

RUN apt-get update \
    && apt-get install -y tmux git vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/

ADD nvidia_dali_cuda110-0.30.0-1983575-py3-none-manylinux2014_x86_64.whl /tmp/
ADD TensorRT-7.2.2.3.Ubuntu-18.04.x86_64-gnu.cuda-11.0.cudnn8.0.tar.gz /workspace/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/TensorRT-7.2.2.3/lib
RUN pip install loguru pytorch-metric-learning onnx 'pycuda>=2019.1.1' /workspace/TensorRT-7.2.2.3/python/tensorrt-7.2.2.3-cp38-none-linux_x86_64.whl /workspace/TensorRT-7.2.2.3/graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl /workspace/TensorRT-7.2.2.3/onnx_graphsurgeon/onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl /tmp/nvidia_dali_cuda110-0.30.0-1983575-py3-none-manylinux2014_x86_64.whl && rm /tmp/nvidia_dali_cuda110-0.30.0-1983575-py3-none-manylinux2014_x86_64.whl

WORKDIR /workspace/TensorRT-7.2.2.3/samples/trtexec
RUN make && ln -s /workspace/TensorRT-7.2.2.3/bin/trtexec /usr/bin

WORKDIR /workspace
RUN git clone https://github.com/TreezzZ/Hashing-Pretrain.git
WORKDIR Hashing-Pretrain