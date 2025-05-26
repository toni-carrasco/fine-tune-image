# fine-tune-image

Ejemplo completo de cómo lanzar el fine-tuning sobre GPT-2 con LoRA usando “wikitext” dataset de Hugging Face:

**Construir la imagen Docker**
En el directorio que contiene tu `Dockerfile` y `train.py` ejecuta:

`docker build -t gpt2-lora-finetune .`

1. **Ejecutar el contenedor con GPU**:
```
docker run --gpus all \
	-v $(pwd)/output:/app/lora-gpt2-output \
	gpt2-lora-finetune \
	--model_name_or_path gpt2 \
	--dataset_name wikitext \
	--dataset_config_name wikitext-2-raw-v1 \
	--output_dir /app/lora-gpt2-output \
	--per_device_train_batch_size 4 \
	--num_train_epochs 3 \
	--learning_rate 2e-4
```

2.  Explicación de los flags más importantes:

   * `--dataset_name wikitext` y `--dataset_config_name wikitext-2-raw-v1` le indican a Transformers qué subset usar.

   * `-v $(pwd)/output:/app/lora-gpt2-output` monta tu carpeta local `./output` dentro del contenedor para persistir los adaptadores LoRA.

   * Ajusta `per_device_train_batch_size`, `num_train_epochs` y `learning_rate` según tu GPU y tamaño de dataset.

3. **Ver los resultados**  
    Una vez termine, encontrarás en `./output`:

   * Los checkpoints periódicos (según `save_steps`)

   * El directorio final con los adaptadores LoRA (`pytorch_model.bin`, `adapter_config.json`, etc.)

