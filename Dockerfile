FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Definir directorio de trabajo
WORKDIR /app

# Copiar archivos de requerimientos (si existen)
# COPY requirements.txt ./

# Actualizar pip e instalar dependencias necesarias
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       transformers>=4.30.0 \
       datasets \
       accelerate \
       peft \
       bitsandbytes \
       huggingface_hub

# Copiar el código de entrenamiento al contenedor
COPY train.py /app/train.py

# Definir el comando por defecto para lanzar el script de entrenamiento
# Asumimos que tienes un train.py que implementa el fine-tuning con LoRA
CMD ["python", "train.py"]
