import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and run LoRA adapters on GPT-2 or LLaMA-7B'
    )
    parser.add_argument(
        'model',
        nargs='?',  # optional positional to allow manual check
        choices=['gpt-2', 'llama-7b'],
        help='Model to use: gpt-2 or llama-7b'
    )
    parser.add_argument(
        '--input', type=str, default='Explain the significance of the industrial revolution.',
        help='Inference input text'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Force model parameter
    if not args.model:
        print('Error: Debe especificar el modelo: gpt-2 o llama-7b', file=sys.stderr)
        sys.exit(1)
    
    print("Versión de torch:", torch.__version__)
    print("Versión de CUDA en torch:", torch.version.cuda)
    print("CUDA disponible:", torch.cuda.is_available())
    print("Número de GPUs detectadas:", torch.cuda.device_count())

    # Model-specific settings
    if args.model == 'gpt-2':
        hf_name = 'gpt2'
        lora_targets = ['c_attn']
        load_in_8bit = False
        output_dir = './gpt2-lora-results'
        adapter_dir = './gpt2-lora-adapters'
        use_fast_tokenizer = True
    else:
        hf_name = 'meta-llama/Llama-2-7b-hf'
        lora_targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        load_in_8bit = True
        output_dir = './llama7b-lora-results'
        adapter_dir = './llama7b-lora-adapters'
        use_fast_tokenizer = False

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(hf_name, use_fast=use_fast_tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        device_map='auto',
        torch_dtype=torch.float16,
        load_in_8bit=load_in_8bit
    )

    # LoRA config
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=lora_targets,
        lora_dropout=0.2
    )
    lora_model = get_peft_model(model, lora_config)

    # Dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    train_subset = dataset['train'].select(range(500))
    eval_subset = dataset['validation'].select(range(50))

    def preprocess_function(batch):
        tokenized = tokenizer(
            batch['text'], truncation=True,
            padding='max_length', max_length=512
        )
        tokenized['labels'] = tokenized['input_ids']
        return tokenized

    train_dataset = train_subset.map(preprocess_function, batched=True)
    eval_dataset = eval_subset.map(preprocess_function, batched=True)

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        eval_steps=10,
        save_steps=10,
        logging_steps=10,
        learning_rate=5e-5,
        fp16=True,
        report_to='none'
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.train()

    # Save LoRA adapters
    lora_model.save_pretrained(adapter_dir)

    # Inference
    # Load base + adapters
    base = AutoModelForCausalLM.from_pretrained(
        hf_name,
        device_map='auto',
        torch_dtype=torch.float16,
        load_in_8bit=load_in_8bit
    )
    loaded = PeftModel.from_pretrained(base, adapter_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded.to(device)
    loaded.eval()

    inputs = tokenizer(
        args.input,
        return_tensors='pt', padding=True,
        truncation=True, max_length=512
    ).to(device)

    outputs = loaded.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == '__main__':
    main()

