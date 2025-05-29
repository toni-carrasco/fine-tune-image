.PHONY: build run shell check-dir

# Build Docker image
build:
	@echo "Building Docker image finetune-image..."
	docker build -t finetune-image .

# Ensure output directory exists and is writable
check-dir:
	@echo "Checking if output directory exists and is writable..."
	@if [ ! -d "$$HOME/finetune-outputs" ]; then \
		echo "Creating output directory at $$HOME/finetune-outputs"; \
		mkdir -p $$HOME/finetune-outputs; \
	fi
	@chmod u+w $$HOME/finetune-outputs

# Run the fine-tuning script with volume mount
run: check-dir
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
	docker run -e HUGGINGFACE_TOKEN \
		--gpus all \
		--rm \
		-v $$HOME/finetune-outputs:/app/outputs \
		finetune-image \
		--model $(MODEL) --peft $(PEFT)

# Start a bash shell inside the container for debugging or exploration
shell:
	@echo "Starting interactive shell in finetune-image..."
	docker run -it --gpus all --rm --entrypoint /bin/bash finetune-image
