import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def compute_lengths(model_name: str, split: str = "train"):
    """
    Carga WikiSQL y calcula estadísticas de longitud en tokens para:
      1) El prompt "Question: {question}\nColumns: {columns}\nSQL:"
      2) La parte de la consulta "human_readable_sql"
    para el split especificado ("train", "validation" o "test").
    """
    # 1) Instanciar el tokenizer
    hf_token = os.getenv("HUGGINGFACE_TOKEN", "")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
        token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 2) Cargar el split de WikiSQL
    raw = load_dataset(
        "Salesforce/wikisql",
        trust_remote_code=True,
        token=hf_token
    )[split]

    # 3A) Función para calcular la longitud del prompt
    def prompt_length(example):
        question = example["question"]
        columns  = ", ".join(example["table"]["header"])
        prompt   = f"Question: {question}\nColumns: {columns}\nSQL:"
        tokens   = tokenizer.encode(prompt, add_special_tokens=False)
        return {"prompt_length": len(tokens)}

    # 3B) Función para calcular la longitud de la consulta SQL
    def sql_length(example):
        sql_str = example["sql"]["human_readable"]
        tokens  = tokenizer.encode(sql_str, add_special_tokens=False)
        return {"sql_length": len(tokens)}

    # 4A) Map para obtener todas las longitudes de prompt
    prompt_lengths_ds = raw.map(
        prompt_length,
        batched=False,
        remove_columns=raw.column_names
    )

    # 4B) Map para obtener todas las longitudes de SQL
    sql_lengths_ds = raw.map(
        sql_length,
        batched=False,
        remove_columns=raw.column_names
    )

    # 5) Extraer a Numpy
    prompt_lengths = np.array(prompt_lengths_ds["prompt_length"])
    sql_lengths    = np.array(sql_lengths_ds["sql_length"])

    # 6A) Estadísticas prompt
    prompt_max    = int(prompt_lengths.max())
    prompt_p99    = int(np.percentile(prompt_lengths, 99))
    prompt_p95    = int(np.percentile(prompt_lengths, 95))
    prompt_median = int(np.median(prompt_lengths))
    prompt_mean   = float(prompt_lengths.mean())

    # 6B) Estadísticas SQL
    sql_max       = int(sql_lengths.max())
    sql_p99       = int(np.percentile(sql_lengths, 99))
    sql_p95       = int(np.percentile(sql_lengths, 95))
    sql_median    = int(np.median(sql_lengths))
    sql_mean      = float(sql_lengths.mean())

    # 7) Imprimir resultados
    print(f"\n=== Estadísticas para split «{split}» de WikiSQL ===")
    print("➜ PROMPT («Question + Columns + SQL:»)")
    print(f"   Máximo tokens:      {prompt_max}")
    print(f"   Percentil 99 (P99): {prompt_p99}")
    print(f"   Percentil 95 (P95): {prompt_p95}")
    print(f"   Mediana (P50):      {prompt_median}")
    print(f"   Media:              {prompt_mean:.1f}\n")

    print("➜ SQL («human_readable_sql»)")
    print(f"   Máximo tokens:      {sql_max}")
    print(f"   Percentil 99 (P99): {sql_p99}")
    print(f"   Percentil 95 (P95): {sql_p95}")
    print(f"   Mediana (P50):      {sql_median}")
    print(f"   Media:              {sql_mean:.1f}\n")

    return {
        "prompt": {
            "max": prompt_max,
            "p99": prompt_p99,
            "p95": prompt_p95,
            "median": prompt_median,
            "mean": prompt_mean
        },
        "sql": {
            "max": sql_max,
            "p99": sql_p99,
            "p95": sql_p95,
            "median": sql_median,
            "mean": sql_mean
        }
    }

if __name__ == "__main__":
    MODEL_NAME = "gpt2"
    stats_train = compute_lengths(MODEL_NAME, split="train")
    stats_val   = compute_lengths(MODEL_NAME, split="validation")
