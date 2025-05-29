import argparse
import os
import sys
import torch, torchvision
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from types import SimpleNamespace
from typing import Dict, Tuple, Any

def parse_args():
    parser = argparse.ArgumentParser(
        description='Entrenamiento usando QLoRA o LoRA adapters sobre GPT-2 o LLaMA-7B'
    )
    parser.add_argument(
        '--model',
        required=True,
        choices=['gpt-2', 'llama-7b'],
        help='Modelo a usar: gpt-2 o llama-7b (obligatorio)'
    )
    parser.add_argument(
        '--peft',
        required=True,
        choices=['lora', 'qlora'],
        help='Tipo de adapter PEFT a usar: lora o qlora (obligatorio)'
    )
    parser.add_argument(
        '--input', type=str,
        default='Explain the significance of the industrial revolution.',
        help='Texto de entrada para inferencia'
    )

    args = parser.parse_args()

    if args.model is None:
        print('Error: Debe especificar el modelo (gpt-2 o llama-7b)', file=sys.stderr)
        sys.exit(1)

    if args.peft is None:
        print('Error: Debe especificar el peft (lora o qlora)', file=sys.stderr)
        sys.exit(1)

    print("\n\n==================================================")
    print("Versión de torch vision:", torchvision.__version__)
    print("Versión de torch:", torch.__version__)
    print("Versión de CUDA en torch:", torch.version.cuda)
    print("CUDA disponible:", torch.cuda.is_available())
    print("Número de GPUs detectadas:", torch.cuda.device_count())
    print("LLM Model:", args.model)
    print("PEFT Mode:", args.peft)
    print("==================================================\n\n")

    return args


def load_env_vars() -> SimpleNamespace:
    specs = {
        'hf_token':   ('HUGGINGFACE_TOKEN', None, True),
        'debug_mode': ('DEBUG_MODE', 'false', False),
    }
    loaded = {}
    missing = []
    for local_name, (env_name, default, mandatory) in specs.items():
        value = os.getenv(env_name, default)
        if mandatory and (value is None or value == ''):
            missing.append(env_name)
        loaded[local_name] = value
    if missing:
        sys.stderr.write(
            'Error: faltan las siguientes variables de entorno obligatorias:\n'
            + '\n'.join(f'  - {name}' for name in missing) + '\n'
        )
        sys.exit(1)
    return SimpleNamespace(**loaded)


def get_model_config(model_name: str, peft: str) -> SimpleNamespace:
    if model_name == 'gpt-2':
        hf_name = 'gpt2'
        target_modules = ['c_attn']
        use_fast_tokenizer = True
    elif model_name == 'llama-7b':
        hf_name = 'meta-llama/Llama-2-7b-hf'
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        use_fast_tokenizer = False
    else:
        raise ValueError(f'Modelo no soportado: {model_name}')

    return SimpleNamespace(
            hf_name=hf_name,
            target_modules=target_modules,
            output_dir=f'./{model_name}-{peft}-results',
            adapter_dir=f'./{model_name}-{peft}-adapters',
            use_fast_tokenizer=use_fast_tokenizer
        )

def load_model(model_name, hf_token, quantization_config = None):
    model_kwargs = {
        "device_map": "auto"
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    if model_name == 'gpt-2':
        return AutoModelForCausalLM.from_pretrained('gpt2', **model_kwargs)

    elif model_name == 'llama-7b':
        model_kwargs["token"] = hf_token
        if quantization_config is None:
            model_kwargs["torch_dtype"] = torch.float16
        return AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', **model_kwargs)

    else:
        raise ValueError(f'Modelo no soportado: {model_name}')
