import argparse
import os
import sys
from types import SimpleNamespace
from typing import Dict, Tuple, Any

def parse_args():
    parser = argparse.ArgumentParser(
        description='Entrenamiento usando QLoRA adapters sobre GPT-2 o LLaMA-7B'
    )
    parser.add_argument(
        'model',
        nargs='?',
        choices=['gpt-2', 'llama-7b'],
        help='Modelo a usar: gpt-2 o llama-7b'
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
    else:
        print("Versión de torch:", torch.__version__)
        print("Versión de CUDA en torch:", torch.version.cuda)
        print("CUDA disponible:", torch.cuda.is_available())
        print("Número de GPUs detectadas:", torch.cuda.device_count())
        print("LLM Model:", args.model)

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


def get_model_config(model: str) -> SimpleNamespace:
    if model == 'gpt-2':
        return SimpleNamespace(
            hf_name='gpt2',
            target_modules=['c_attn'],
            output_dir='./gpt2-qlora-results',
            adapter_dir='./gpt2-qlora-adapters',
            use_fast_tokenizer=True
        )
    else:
        return SimpleNamespace(
            hf_name='meta-llama/Llama-2-7b-hf',
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            output_dir='./llama7b-qlora-results',
            adapter_dir='./llama7b-qlora-adapters',
            use_fast_tokenizer=False
        )
