# **Proyecto Fine-Tuning Image**

Este repositorio contiene un script para entrenar y ejecutar adaptadores LoRA sobre modelos **GPT-2** o **LLaMA-7B** dentro de un contenedor Docker.

## **Prerrequisitos**

* Docker instalado y configurado para usar GPUs (NVIDIA Docker).
    * [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    * [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)


* Token de Hugging Face para acceder a repositorios con acceso restringido:

```
export HUGGINGFACE_TOKEN=<tu_token_HF>
```

## **Estructura**

* **Dockerfile**: Define la imagen base con dependencias (transformers, peft, etc.).

* **lora.py**: Aplica LoRA .... (TODO)

* **Makefile**: Simplifica la construcción de la imagen y la ejecución del contenedor.

---

## **Uso con Make**

### **1\. Construir la imagen**

```
make build`
```

### **2\. Ejecutar el contenedor**

Para entrenar/ejecutar dentro del contenedor, debes proporcionar dos variables:

* `PEFT`: tecnica peft a aplicar (`lora`,`qlora`, ...)

* `MODEL`: modelo a usar (`gpt-2` o `llama-7b`).

`make run PEFT=lora MODEL=gpt-2`

