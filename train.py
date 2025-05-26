import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 with LoRA")
    parser.add_argument(
        "--model_name_or_path", type=str, default="gpt2", help="Model base"
    )
    parser.add_argument(
        "--dataset_name", type=str, default=None, help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="Local dataset file (one text file or JSONL)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./lora-gpt2", help="Output directory"
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=8
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=8
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4
    )
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()

    # Cargar tokenizer y modelo
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # Asegurar token de padding
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    # Configurar LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = get_peft_model(model, peft_config)

    # Cargar dataset
    if args.dataset_name:
        dataset = load_dataset(args.dataset_name)
    else:
        # Asume un archivo de texto o JSONL
        dataset = load_dataset('text', data_files={'train': args.dataset_path})

    # Tokenización
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], truncation=True, max_length=args.max_seq_length
        )

    tokenized = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset['train'].column_names
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Argumentos de entrenamiento
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
    )

    # Preparar Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Entrenar
    trainer.train()

    # Guardar adaptadores LoRA
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
