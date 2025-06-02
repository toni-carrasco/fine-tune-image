FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        build-essential \
        git \
        libgoogle-perftools-dev && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      torch==2.7.0+cu126 \
      torchvision==0.22.0+cu126 \
      torchaudio==2.7.0+cu126 \
      --index-url https://download.pytorch.org/whl/cu126

RUN pip install --no-cache-dir \
      transformers \
      datasets \
      accelerate \
      sentencepiece \
      protobuf \
      bitsandbytes \
      psutil \
      pynvml

RUN pip install --no-cache-dir git+https://github.com/huggingface/peft.git

COPY configs/* .
COPY src/* .

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["python", "train.py"]
