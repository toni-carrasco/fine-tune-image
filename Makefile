.PHONY: build run

# Build Docker image
build:
	@echo "Building Docker image finetune-image..."
	docker build -t finetune-image .

# Run the LoRA script in container with GPU support
# Exige pasar variables PEFT y MODEL en tiempo de ejecución
run:
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
	@echo "Running container finetune-image with PEFT=$(PEFT) and MODEL=$(MODEL) on all GPUs..."
	docker run -e HUGGINGFACE_TOKEN --gpus all --rm finetune-image --model $(MODEL) --peft $(PEFT)

# Start a bash shell inside the container for debugging or exploration
shell:
	@echo "Starting interactive shell in finetune-image..."
	docker run -it --gpus all --rm --entrypoint /bin/bash finetune-image
