import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from utils import parse_args, load_env_vars, get_model_config

def get_lora_peft_model(hf_name, ht_token, target_modules):
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        device_map='auto',
        torch_dtype=torch.float16,
        load_in_8bit=load_in_8bit,
        use_auth_token=hf_token
    )
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.2,
        target_modules=target_modules
    )
    return get_peft_model(model, lora_config)

def get_lora_inference_model(hf_name, hf_token):
    return AutoModelForCausalLM.from_pretrained(
            hf_name,
            device_map='auto',
            torch_dtype=torch.float16,
            use_auth_token=hf_token,
            load_in_8bit=load_in_8bit
        )

def main():
    args = parse_args()

    env = load_env_vars()
    hf_token = env.hf_token
    debug_mode = env.debug_mode.lower() in ('1', 'true', 'yes')

    config = get_model_config(args.model)
    hf_name = config.hf_name
    target_modules = config.target_modules
    output_dir = config.output_dir
    adapter_dir = config.adapter_dir
    use_fast_tokenizer = config.use_fast_tokenizer

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_name,
        use_fast=use_fast_tokenizer,
        use_auth_token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    peft_model = get_lora_peft_model(hf_name, ht_token, target_modules)

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
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.train()

    # Save LoRA adapters
    peft_model.save_pretrained(adapter_dir)

    # Inference
    base = get_lora_inference_model(hf_name, hf_token)
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

