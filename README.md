# **Proyecto Fine-Tuning Image**

Este repositorio proporciona una **imagen Docker** preconfigurada con todas las dependencias necesarias para aplicar técnicas de **PEFT** (Parameter-Efficient Fine-Tuning) sobre modelos de lenguaje de última generación. El objetivo principal es facilitar un entorno reproducible y listo para usar, donde puedas experimentar y desplegar entrenamientos eficientes sin preocuparte por la instalación manual de librerías.

### Técnicas PEFT soportadas

- **LoRA** (Low-Rank Adaptation)  
- **QLoRA** (Quantized LoRA)  
- **BitFit**  
- **Prefix Tuning v2**  
- **IA³** (Injected Attention Adapter)

### Modelos compatibles

- **GPT-2**  
- **LLaMA 7B**

### Características principales

1. **Entorno aislado**  
   La imagen Docker incluye Python 3.x y todas las librerías (Transformers, Accelerate, BitsAndBytes, PEFT, etc.) necesarias para correr tus scripts de fine-tuning.

2. **Optimización de recursos**  
   Configuraciones por defecto orientadas a entrenamiento en GPU, con soporte para cuantización y bajo uso de memoria.

3. **Reproducibilidad**  
   Versión fija de cada dependencia para garantizar que tus experimentos sean consistentes y fácilmente replicables.

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
make build
```

### **2\. Ejecutar el contenedor**

Para entrenar/ejecutar dentro del contenedor, debes proporcionar dos variables:

* `PEFT`: tecnica peft a aplicar (`lora`,`qlora`, ...)

* `MODEL`: modelo a usar (`gpt-2` o `llama-7b`).

```
make run PEFT=lora MODEL=gpt-2
```

