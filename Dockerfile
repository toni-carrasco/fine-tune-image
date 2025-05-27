FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      torch==2.1.0+cu118 \
      torchvision==0.16.0+cu118 \
      torchaudio==2.1.0+cu118 \
      --extra-index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir \
      transformers \
      datasets \
      accelerate \
      peft

COPY lora.py .

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["python", "lora.py"]
