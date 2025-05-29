import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, IA3Config, PrefixTuningConfig, get_peft_model, PeftModel
from datasets import load_dataset
from utils import parse_args, load_env_vars, get_model_config, load_model


def get_peft_model_with_lora_config(model_name, hf_token, target_modules, bnb_config):
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


def get_peft_model_with_ia3_config(model_name, hf_token, target_modules):
    model = load_model(model_name, hf_token)

    # Detectar si el modelo es tipo GPT-2 (usa 'c_attn')
    is_gpt2 = 'c_attn' in target_modules

    # Definir módulos FFN compatibles según target_modules
    candidate_ff_modules = ['mlp', 'fc1', 'gate_proj']
    feedforward_modules = [m for m in candidate_ff_modules if m in target_modules]

    ia3_config = IA3Config(
        target_modules=target_modules,
        feedforward_modules=feedforward_modules,
        task_type='CAUSAL_LM',
        fan_in_fan_out=is_gpt2
    )
    return get_peft_model(model, ia3_config)


def get_peft_model_with_prefix_config(model_name, hf_token):
    model = load_model(model_name, hf_token)
    prefix_config = PrefixTuningConfig(
        num_virtual_tokens=30,
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, prefix_config)


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

    bnb_config = None
    if args.peft == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        peft_model = get_peft_model_with_lora_config(args.model, hf_token, target_modules, bnb_config)
    elif args.peft == "lora":
        peft_model = get_peft_model_with_lora_config(args.model, hf_token, target_modules, bnb_config)
    elif args.peft == "ia3":
        peft_model = get_peft_model_with_ia3_config(args.model, hf_token, target_modules)
    elif args.peft == "prefix":
        peft_model = get_peft_model_with_prefix_config(args.model, hf_token)
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


if __name__ == '__main__':
    main()
