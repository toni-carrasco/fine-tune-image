.PHONY: help build train shell check-dir check-vars

# Default target: show help
help:
	@echo ""
	@echo "Usage:"
	@echo "  make <target> [VARIABLE=value]"
	@echo ""
	@echo "Targets:"
	@echo "  build         - Construye la imagen Docker 'finetune-image'."
	@echo "  train         - Ejecuta el entrenamiento dentro del contenedor (requiere PEFT y MODEL)."
	@echo "                  Ejemplo: make train PEFT=lora MODEL=gpt-2"
	@echo "  infer         - Ejecuta inferencia dentro del contenedor (requiere PEFT y MODEL)."
	@echo "                  Ejemplo: make infer PEFT=lora MODEL=gpt-2"
	@echo "  shell         - Inicia una shell interactiva en el contenedor."
	@echo "  help          - Muestra esta ayuda."
	@echo ""

# Build Docker image
build:
	@echo "Building Docker image finetune-image..."
	docker build -t finetune-image .

# Ensure output directory exists and is writable
check-dir:
	@echo "Checking if output directory exists and is writable..."
	@if [ ! -d "$$HOME/fine-tune-outputs" ]; then \
		echo "Creating output directory at $$HOME/fine-tune-outputs"; \
		mkdir -p $$HOME/fine-tune-outputs; \
	fi
	@chmod u+w $$HOME/fine-tune-outputs

check-vars:
	@# Verifica PEFT
	@if [ -z "$(PEFT)" ]; then \
		echo "Error: PEFT no está definido. Uso: make run PEFT=<peft> MODEL=<model>"; \
		exit 1; \
	fi
	@# Verifica MODEL
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL no está definido. Uso: make run PEFT=<peft> MODEL=<model>"; \
		exit 1; \
	fi

# Run the fine-tuning script with volume mount
train: check-vars check-dir
	@echo "Running container finetune-image to train with PEFT=$(PEFT) and MODEL=$(MODEL) on all GPUs..."
	docker run --rm --gpus all \
		-e HUGGINGFACE_TOKEN \
		-e DATASET_SAMPLE_SIZE \
		-v $$HOME/fine-tune-outputs:/app/outputs \
		-v ./training_configuration.json:/app/training_configuration.json \
		finetune-image python train.py --model $(MODEL) --peft $(PEFT)

# Run the fine-tuning script with volume mount
infer: check-vars check-dir
	@echo "Running container finetune-image to infer with PEFT=$(PEFT) and MODEL=$(MODEL) on all GPUs..."
	docker run -it --rm --gpus all \
		-e HUGGINGFACE_TOKEN \
		-v $$HOME/fine-tune-outputs:/app/outputs \
		finetune-image python infer.py --model $(MODEL) --peft $(PEFT)

# Start a bash shell inside the container for debugging or exploration
shell: check-dir
	@echo "Starting interactive shell in finetune-image..."
	docker run -it --rm --gpus all \
		-v $$HOME/finetune-outputs:/app/outputs \
		--entrypoint /bin/bash \
		finetune-image
