import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.bitfit import BitFitConfig
from datasets import load_dataset
from utils import parse_args, load_env_vars, get_model_config, load_model


def get_lora_model(model_name, hf_token, target_modules, bnb_config=None):
    model = load_model(model_name, hf_token, bnb_config)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.2,
        target_modules=target_modules,
        bias='none',
        task_type='CAUSAL_LM'
    )
    return get_peft_model(model, lora_config)


def get_bitfit_model(model_name, hf_token):
    model = load_model(model_name, hf_token)
    bitfit_config = BitFitConfig(
        task_type='CAUSAL_LM',
        bias='all'
    )
    return get_peft_model(model, bitfit_config)


def main():
    args = parse_args()

    env = load_env_vars()
    hf_token = env.hf_token
    debug_mode = env.debug_mode.lower() in ('1', 'true', 'yes')

    config = get_model_config(args.model, args.peft)
    hf_name = config.hf_name
    target_modules = config.target_modules
    output_dir = config.output_dir
    adapter_dir = config.adapter_dir
    use_fast_tokenizer = config.use_fast_tokenizer

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_name,
        use_fast=use_fast_tokenizer,
        token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.peft == "lora":
        peft_model = get_lora_model(args.model, hf_token, target_modules)
    elif args.peft == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        peft_model = get_lora_model(args.model, hf_token, target_modules, bnb_config)
    elif args.peft == "bitfit":
        peft_model = get_bitfit_model(args.model, hf_token)
    else:
        raise ValueError(f"Modo PEFT no soportado: {args.peft}")

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

    # Save adapters
    peft_model.save_pretrained(adapter_dir)

    # Inference
    if args.peft == "lora":
        base = load_model(args.model, hf_token)
    elif args.peft == "qlora":
        base = load_model(args.model, hf_token, bnb_config)
    elif args.peft == "bitfit":
        base = load_model(args.model, hf_token)

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
