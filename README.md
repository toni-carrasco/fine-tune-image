# fine-tune-image

1. **Construir la imagen Docker**
En el directorio que contiene tu `Dockerfile` ejecuta:

`docker build -t gpt2-lora-finetune .`

2. **Ejecutar el contenedor con GPU**:
```
docker run --gpus all --rm gpt2-lora-gpu
```
