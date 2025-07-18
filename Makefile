.PHONY: help build train shell check-dir check-vars infer define-image-name

# Default target: show help
help:
	@echo ""
	@echo "Usage:"
	@echo "  make <target> [VARIABLE=value]"
	@echo ""
	@echo "Targets:"
	@echo "  build         - Construye la imagen Docker 'finetune-image-{branch}'."
	@echo "  train         - Ejecuta el entrenamiento dentro del contenedor (requiere PEFT y MODEL)."
	@echo "                  Ejemplo: make train PEFT=lora MODEL=gpt-2"
	@echo "  infer         - Ejecuta inferencia dentro del contenedor (requiere PEFT y MODEL)."
	@echo "                  Ejemplo: make infer PEFT=lora MODEL=gpt-2"
	@echo "  shell         - Inicia una shell interactiva en el contenedor."
	@echo "  help          - Muestra esta ayuda."
	@echo ""

# Etapa para definir el nombre de la imagen Docker según la rama actual
define-image-name:
	@echo "Detectando rama git actual y generando nombre de imagen..."
	@GIT_BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	IMAGE_NAME=finetune-image-$${GIT_BRANCH}; \
	echo $${IMAGE_NAME} > .image_name; \
	echo "Nombre de imagen Docker: $${IMAGE_NAME}"

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
		echo "Error: PEFT no está definido. Uso: make train PEFT=<peft> MODEL=<model>"; \
		exit 1; \
	fi
	@# Verifica MODEL
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL no está definido. Uso: make train PEFT=<peft> MODEL=<model>"; \
		exit 1; \
	fi

# Build Docker image
build: define-image-name
	@IMAGE_NAME=$$(cat .image_name); \
	echo "Building Docker image $$IMAGE_NAME..."; \
	sudo docker build -t $$IMAGE_NAME .

# Train
train: define-image-name check-vars check-dir
	@IMAGE_NAME=$$(cat .image_name); \
	echo "Running container $$IMAGE_NAME to train with PEFT=$(PEFT) and MODEL=$(MODEL)..."; \
	sudo -E docker run --rm --gpus all \
		-e HUGGINGFACE_TOKEN \
		-e DATASET_SIZE_RATIO \
		-v $$HOME/fine-tune-outputs:/app/outputs \
		-v ./configs:/app/configs \
		$$IMAGE_NAME python train.py --model $(MODEL) --peft $(PEFT)

# Infer
infer: define-image-name check-vars check-dir
	@IMAGE_NAME=$$(cat .image_name); \
	echo "Running container $$IMAGE_NAME to infer with PEFT=$(PEFT) and MODEL=$(MODEL)..."; \
	sudo -E docker run -it --rm --gpus all \
		-e HUGGINGFACE_TOKEN \
		-v $$HOME/fine-tune-outputs:/app/outputs \
		$$IMAGE_NAME python infer.py --model $(MODEL) --peft $(PEFT)

# Shell
shell: define-image-name check-dir
	@IMAGE_NAME=$$(cat .image_name); \
	echo "Starting interactive shell in $$IMAGE_NAME..."; \
	sudo -E docker run -it --rm --gpus all \
		-v $$HOME/fine-tune-outputs:/app/outputs \
		-v ./configs:/app/configs \
		--entrypoint /bin/bash \
		$$IMAGE_NAME

