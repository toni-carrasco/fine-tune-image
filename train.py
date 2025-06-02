import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, EvalPrediction
from peft import LoraConfig, IA3Config, PrefixTuningConfig, PeftModel, get_peft_model
from wikisql_dataset import get_wikisql_datasets, get_tokenizer
from utils import (
    parse_args,
    load_env_vars,
    get_model_config,
    load_model,
    load_training_arguments_from_json,
    load_peft_arguments_from_json,
    start_benchmark_metrics,
    stop_benchmark_metrics
)
from callbacks import StepEvalAndEarlyStopCallback


def get_peft_model_with_qlora_config(model_name, hf_token, config):
    peft_config = load_peft_arguments_from_json("peft_qlora_configuration.json", config.output_dir)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=peft_config.pop("load_in_4bit", True),
        bnb_4bit_quant_type=peft_config.pop("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=peft_config.pop("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=torch.float16
    )
    return get_peft_model_with_lora_config(model_name, hf_token, config, bnb_config)

def get_peft_model_with_lora_config(model_name, hf_token, config, bnb_config):
    peft_config = load_peft_arguments_from_json("peft_lora_configuration.json", config.output_dir)
    model = load_model(model_name, hf_token, bnb_config)
    lora_config = LoraConfig(
        r=peft_config.pop("r", 4),
        lora_alpha=peft_config.pop("lora_alpha", 8),
        lora_dropout=peft_config.pop("lora_dropout", 0.2),
        target_modules=config.target_modules,
        bias=peft_config.pop("bias", "none"),
        task_type=peft_config.pop("task_type", "CAUSAL_LM")
    )
    return get_peft_model(model, lora_config)


def get_peft_model_with_ia3_config(model_name, hf_token, config):
    peft_config = load_peft_arguments_from_json("peft_ia3_configuration.json", config.output_dir)
    model = load_model(model_name, hf_token)

    # Detectar si el modelo es tipo GPT-2 (usa 'c_attn')
    is_gpt2 = 'c_attn' in target_modules

    # Definir módulos FFN compatibles según target_modules
    candidate_ff_modules = ['mlp', 'fc1', 'gate_proj']
    feedforward_modules = [m for m in candidate_ff_modules if m in target_modules]

    ia3_config = IA3Config(
        target_modules=config.target_modules,
        feedforward_modules=feedforward_modules,
        task_type=peft_config.pop("task_type", "CAUSAL_LM"),
        fan_in_fan_out=is_gpt2
    )
    return get_peft_model(model, ia3_config)


def get_peft_model_with_prefix_config(model_name, hf_token, config):
    peft_config = load_peft_arguments_from_json("peft_prefix_configuration.json", config.output_dir)
    model = load_model(model_name, hf_token)
    prefix_config = PrefixTuningConfig(
        num_virtual_tokens=peft_config.pop("num_virtual_tokens", 30),
        task_type=peft_config.pop("task_type", "CAUSAL_LM")
    )
    return get_peft_model(model, prefix_config)


def main():
    args = parse_args()
    config = get_model_config(args.model, args.peft)
    env = load_env_vars()
    hf_token = env.hf_token
    dataset_size_ratio = env.dataset_size_ratio
    debug_mode = env.debug_mode.lower() in ('1', 'true', 'yes')

    training_config = load_training_arguments_from_json("training_configuration.json", config.output_dir)

    # Tokenizer
    tokenizer = get_tokenizer(hf_token, config)

    bnb_config = None
    if args.peft == "qlora":
        peft_model = get_peft_model_with_qlora_config(args.model, hf_token, config)
    elif args.peft == "lora":
        peft_model = get_peft_model_with_lora_config(args.model, hf_token, config, bnb_config)
    elif args.peft == "ia3":
        peft_model = get_peft_model_with_ia3_config(args.model, hf_token, config)
    elif args.peft == "prefix":
        peft_model = get_peft_model_with_prefix_config(args.model, hf_token, config)
    else:
        raise ValueError(f"Modo PEFT no soportado: {args.peft}")

    # Dataset
    train_dataset, eval_dataset = get_wikisql_datasets(tokenizer, hf_token, dataset_size_ratio)

    # Training args
    training_args = TrainingArguments(**training_config)

    eval_steps = training_config.pop("eval_steps", 100)
    step_eval_and_early = StepEvalAndEarlyStopCallback(eval_steps=eval_steps)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[step_eval_and_early],
    )

    metrics = start_benchmark_metrics()
    trainer.train()
    stop_benchmark_metrics(metrics, config.output_dir)

    # Save adapters
    peft_model.save_pretrained(config.adapter_dir)


if __name__ == '__main__':
    main()
