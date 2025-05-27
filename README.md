# **Proyecto Fine-Tuning Image**

Este repositorio contiene un script para entrenar y ejecutar adaptadores LoRA sobre modelos **GPT-2** o **LLaMA-7B** dentro de un contenedor Docker.

## **Prerrequisitos**

* Docker instalado y configurado para usar GPUs (NVIDIA Docker).

* (Opcional) Token de Hugging Face para acceder a repositorios con acceso restringido:

 `export HUGGINGFACE_TOKEN=<tu_token_HF>`

* Make (opcional, pero recomendado para simplificar comandos).

## **Estructura**

* **Dockerfile**: Define la imagen base con dependencias (transformers, peft, etc.).

* **lora.py**: Script principal que:

  1. Carga el modelo (`gpt-2` o `llama-7b`).

  2. Configura y entrena adaptadores LoRA.

  3. Guarda y recarga adaptadores para generación.

* **Makefile**: Simplifica la construcción de la imagen y la ejecución del contenedor.

---

## **Uso con Make**

### **1\. Construir la imagen**

make build

Esto ejecuta:

docker build \-t finetune-image .

y construye la imagen **finetune-image**.

### **2\. Ejecutar el contenedor**

Para entrenar/ejecutar dentro del contenedor, debes proporcionar dos variables:

* `PEFT`: tecnica peft a aplicar sobre el modelo que se quiere hacer fine-tunning.

* `MODEL`: modelo a usar (`gpt-2` o `llama-7b`).

`make run PEFT=lora MODEL=gpt-2`


