import argparse
import os
import sys
import json
import time
import psutil
import threading
import torch, torchvision
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from types import SimpleNamespace
from typing import Dict, Tuple, Any
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlShutdown


def parse_args():
    parser = argparse.ArgumentParser(
        description='Entrenamiento usando QLoRA, LoRA, IA^3 o Prefix Tuning V2 sobre los modelos GPT-2 o LLaMA-7B'
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
        choices=['lora', 'qlora', 'ia3', 'prefix'],
        help='Tipo de adapter PEFT a usar: lora, qlora, ia3, prefix (obligatorio)'
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
        print('Error: Debe especificar el peft (lora, qlora, ia3, prefix)', file=sys.stderr)
        sys.exit(1)

    print("\n\n✅ Entorno de ejecucion:")
    print("Versión de torch vision:", torchvision.__version__)
    print("Versión de torch:", torch.__version__)
    print("Versión de CUDA en torch:", torch.version.cuda)
    print("CUDA disponible:", torch.cuda.is_available())
    print("Número de GPUs detectadas:", torch.cuda.device_count())
    print("LLM Model:", args.model)
    print("PEFT Mode:", args.peft)

    return args


def load_env_vars() -> SimpleNamespace:
    specs = {
        'hf_token':           ('HUGGINGFACE_TOKEN', None, True),
        'debug_mode':         ('DEBUG_MODE', 'false', False),
        'dataset_size_ratio': ('DATASET_SIZE_RATIO', None, False)
    }

    sensitive_keys = {'hf_token'}  # claves locales que deben enmascararse
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

    print("\n✅ Variables de entorno cargadas:")
    for key, value in loaded.items():
        if key in sensitive_keys and value:
            display_value = '*' * 8 + value[-4:]  # muestra últimos 4 caracteres
        else:
            display_value = value
        print(f"{key}: {display_value}")

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
            output_dir=f'/app/outputs/{model_name}-{peft}-results',
            adapter_dir=f'/app/outputs/{model_name}-{peft}-adapters',
            use_fast_tokenizer=use_fast_tokenizer
        )

def load_model(model_name, hf_token, quantization_config = None):
    model_kwargs = {
        "device_map": "auto"
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    if model_name == 'gpt-2':
        model = AutoModelForCausalLM.from_pretrained('gpt2', **model_kwargs)

    elif model_name == 'llama-7b':
        model_kwargs["token"] = hf_token
        if quantization_config is None:
            model_kwargs["torch_dtype"] = torch.float16
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', **model_kwargs)

    else:
        raise ValueError(f'Modelo no soportado: {model_name}')

    if hasattr(model, "quantization_config"):
        print("\n✅ Modelo cargado con quantization_config → Esto es QLoRA (4-bit)")
    else:
        print("\n✅ Modelo cargado sin quantization_config")

    return model

def load_training_arguments_from_json(json_path: str, output_dir: str):
    with open(json_path, 'r') as f:
        config = json.load(f)

    config['output_dir'] = output_dir

    print("\n✅ Training parameters")
    for key, value in config.items():
        print(f"{key} = {value} of type ({type(value)})")

    return config

def load_peft_arguments_from_json(json_path: str, output_dir: str):
    with open(json_path, 'r') as f:
        config = json.load(f)

    config['output_dir'] = output_dir

    print("\n✅ PEFT parameters")
    for key, value in config.items():
        print(f"{key} = {value} of type ({type(value)})")

    return config

def start_benchmark_metrics():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    process = psutil.Process()

    metrics = {
        "start_time": time.time(),
        "start_ram_mb": process.memory_info().rss / (1024 ** 2),
        "process": process,
        "gpu_handle": handle,
        "gpu_util_samples": [],
        "gpu_mem_samples": [],
        "cpu_util_samples": [],
        "ram_total_used_mb_samples": [],
        "stop_thread": False,
    }

    def sampler():
        while not metrics["stop_thread"]:
            # GPU
            util = nvmlDeviceGetUtilizationRates(handle).gpu
            mem = nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)  # MB
            metrics["gpu_util_samples"].append(util)
            metrics["gpu_mem_samples"].append(mem)

            # CPU (porcentaje de uso del sistema)
            cpu_util = psutil.cpu_percent(interval=None)
            metrics["cpu_util_samples"].append(cpu_util)

            # RAM (uso total del sistema en MB)
            ram_used_mb = psutil.virtual_memory().used / (1024 ** 2)
            metrics["ram_total_used_mb_samples"].append(ram_used_mb)

            time.sleep(1)

    metrics["thread"] = threading.Thread(target=sampler)
    metrics["thread"].start()

    return metrics

def stop_benchmark_metrics(metrics, output_dir):
    metrics["stop_thread"] = True
    metrics["thread"].join()

    end_time = time.time()
    training_duration = end_time - metrics["start_time"]

    process = metrics["process"]
    gpu_handle = metrics["gpu_handle"]

    end_ram = process.memory_info().rss / (1024 ** 2)
    ram_used = end_ram - metrics["start_ram_mb"]
    cpu_percent = process.cpu_percent(interval=1.0)

    # Listas de muestras
    gpu_util = metrics["gpu_util_samples"]
    gpu_mem = metrics["gpu_mem_samples"]
    cpu_util = metrics["cpu_util_samples"]
    ram_total_used = metrics["ram_total_used_mb_samples"]

    # Cálculos promedio
    avg_gpu_util = sum(gpu_util) / len(gpu_util) if gpu_util else 0
    avg_gpu_mem = sum(gpu_mem) / len(gpu_mem) if gpu_mem else 0
    avg_cpu_util = sum(cpu_util) / len(cpu_util) if cpu_util else 0
    avg_ram_total_used = sum(ram_total_used) / len(ram_total_used) if ram_total_used else 0

    nvmlShutdown()

    results = {
        "training_time_sec": round(training_duration, 2),
        "process_ram_used_mb": round(ram_used, 2),
        "process_cpu_percent": round(cpu_percent, 2),
        "avg_gpu_memory_used_mb": round(avg_gpu_mem, 2),
        "avg_gpu_utilization_percent": round(avg_gpu_util, 2),
        "avg_cpu_utilization_percent": round(avg_cpu_util, 2),
        "avg_system_ram_used_mb": round(avg_ram_total_used, 2)
    }


    print("\n✅ Benchmark statistics")
    print(results)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
