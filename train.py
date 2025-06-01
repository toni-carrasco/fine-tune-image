import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, \
    EarlyStoppingCallback, TrainerCallback
from peft import LoraConfig, IA3Config, PrefixTuningConfig, PeftModel, get_peft_model
from wikisql_dataset import get_wikisql_datasets
from utils import (
    parse_args,
    load_env_vars,
    get_model_config,
    load_model,
    load_training_arguments_from_json,
    start_benchmark_metrics,
    stop_benchmark_metrics
)


class StepEvalCallback(TrainerCallback):
    """
    A callback that tells Trainer to run .evaluate() every eval_steps.
    """

    def __init__(self, eval_steps: int):
        self.eval_steps = eval_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            control.should_evaluate = True
        return control


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
    config = get_model_config(args.model, args.peft)
    env = load_env_vars()
    hf_token = env.hf_token
    dataset_size_ratio = env.dataset_size_ratio
    debug_mode = env.debug_mode.lower() in ('1', 'true', 'yes')

    training_config = load_training_arguments_from_json("training_configuration.json", config.output_dir)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_name,
        use_fast=config.use_fast_tokenizer,
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
        peft_model = get_peft_model_with_lora_config(args.model, hf_token, config.target_modules, bnb_config)
    elif args.peft == "lora":
        peft_model = get_peft_model_with_lora_config(args.model, hf_token, config.target_modules, bnb_config)
    elif args.peft == "ia3":
        peft_model = get_peft_model_with_ia3_config(args.model, hf_token, config.target_modules)
    elif args.peft == "prefix":
        peft_model = get_peft_model_with_prefix_config(args.model, hf_token)
    else:
        raise ValueError(f"Modo PEFT no soportado: {args.peft}")

    # Dataset
    train_dataset, eval_dataset = get_wikisql_datasets(tokenizer, hf_token, dataset_size_ratio)

    eval_steps = training_config.pop("eval_steps", 100)
    training_config["metric_for_best_model"] = "loss"
    training_config["greater_is_better"] = False
    training_config["load_best_model_at_end"] = True

    # Training args
    training_args = TrainingArguments(**training_config)

    step_eval_cb = StepEvalCallback(eval_steps)
    early_stopping_cb = EarlyStoppingCallback(
        early_stopping_patience=3,  # Espera 3 evaluaciones sin mejora
        early_stopping_threshold=0.0
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            step_eval_cb,
            early_stopping_cb
        ],
    )

    metrics = start_benchmark_metrics()
    trainer.train()
    stop_benchmark_metrics(metrics, config.output_dir)

    # Save adapters
    peft_model.save_pretrained(config.adapter_dir)


if __name__ == '__main__':
    main()
