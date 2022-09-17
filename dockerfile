FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
RUN mkdir /workspace/OrdinalCLIP
COPY . /workspace/OrdinalCLIP/

# uncomment this if you have trouble downloading packages
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 

RUN pip install -r /workspace/OrdinalCLIP/requirements.txt && \
    pip install -e /workspace/OrdinalCLIP && \
    pip install -e /workspace/OrdinalCLIP/CLIP

WORKDIR /workspace/OrdinalCLIP