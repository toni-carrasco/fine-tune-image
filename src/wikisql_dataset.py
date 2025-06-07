import os
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from typing import Tuple

# Constantes recomendadas (según estadísticas calculadas previamente)
PROMPT_MAX = 96    # cubre P99 de “Question + Columns + SQL:”
SQL_MAX    = 40    # cubre P99 de “human_readable_sql”
FULL_MAX   = 128   # redondeando PROMPT_MAX+SQL_MAX (96+40=136) a múltiplo de 8

def get_tokenizer(hf_token, config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_name,
        use_fast=config.use_fast_tokenizer,
        token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer

def _preprocess_wikisql_without_tokenizer(batch: dict) -> dict:
    questions = batch["question"]
    columns   = [", ".join(tbl["header"]) for tbl in batch["table"]]
    sqls      = [entry["human_readable"] for entry in batch["sql"]]

    prompts = []
    for q, cols in zip(questions, columns):
        prompt = f"Question: {q}\nColumns: {cols}\nSQL:"
        prompts.append(prompt)

    output = {}
    output["input_ids"] = prompts
    output["labels"] = sqls

    return output


def _preprocess_wikisql(
        tokenizer: PreTrainedTokenizerBase,
        batch: dict
) -> dict:
    """
    Para cada ejemplo en el batch:
      1) Construye el prompt: "Question: {q}\nColumns: {cols}\nSQL:"
      2) Obtiene la query en SQL (human_readable_sql)
      3) Concatena prompt + " " + sql en una sola cadena
      4) Tokeniza la cadena completa a FULL_MAX tokens
      5) Genera labels = input_ids.copy() y enmascara (-100) la parte del prompt
    El resultado:
      - input_ids: (batch_size, FULL_MAX)
      - attention_mask: (batch_size, FULL_MAX)
      - labels: (batch_size, FULL_MAX) con -100 en los primeros tokens del prompt.
    """
    questions = batch["question"]
    columns   = [", ".join(tbl["header"]) for tbl in batch["table"]]
    sqls      = [entry["human_readable"] for entry in batch["sql"]]

    prompts = []
    for q, cols in zip(questions, columns):
        prompt = f"Question: {q}\nColumns: {cols}\nSQL:"
        prompts.append(prompt)

    # Construimos la lista de cadenas completas: prompt + " " + sql
    full_texts = [prompt + " " + sql for prompt, sql in zip(prompts, sqls)]

    # Tokenizamos la cadena completa con truncate/padding a FULL_MAX
    tok_full = tokenizer(
        full_texts,
        truncation=True,
        padding="max_length",
        max_length=FULL_MAX
    )
    input_ids      = tok_full["input_ids"]       # forma: (batch_size, FULL_MAX)
    attention_mask = tok_full["attention_mask"]   # forma: (batch_size, FULL_MAX)

    # Creamos los labels a partir de input_ids (copiamos para no modificar input_ids directamente)
    labels = [ids.copy() for ids in input_ids]   # lista de listas, cada una de longitud FULL_MAX

    # Para cada ejemplo, calculamos cuántos tokens ocupa el prompt en la tokenización
    # tokenizamos solo el prompt (sin la parte SQL)
    for i, prompt in enumerate(prompts):
        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=PROMPT_MAX
        )
        len_prompt = len(prompt_ids)  # número de tokens del prompt

        # Asignamos -100 en positions [0 .. len_prompt-1] para no computar pérdida en el prompt
        for j in range(len_prompt):
            labels[i][j] = -100

    # Guardamos labels en el diccionario resultante
    tok_full["labels"] = labels

    return tok_full  # contiene: input_ids, attention_mask, labels

def _print_dataset_examples(dataset, split: str = "train", n: int = 5):
    """
    Imprime n ejemplos aleatorios, mostrando: table, columns, question y human_readable_sql.
    """
    examples = dataset[split].shuffle(seed=42).select(range(n))

    print("\nDataset examples:")
    for ex in examples:
        table_name = ex["table"]["id"]  # no hay 'name', el ID es el identificador de la tabla
        columns    = ", ".join(ex["table"]["header"])
        question   = ex["question"]
        hr_sql     = ex["sql"]["human_readable"]

        print(f"table:              {table_name}")
        print(f"columns:            {columns}")
        print(f"question:           {question}")
        print(f"human_readable_sql: {hr_sql}")
        print("-" * 80)

def get_wikisql_datasets(
        tokenizer: PreTrainedTokenizerBase,
        hf_token: str,
        dataset_size_ratio: float = None
) -> Tuple:
    """
    1) Carga WikiSQL (train + validation).
    2) Muestra 5 ejemplos por consola (_print_dataset_examples).
    3) Si se pidió muestreo (dataset_size_ratio), reduce a ese % ambos splits.
    4) Mapea cada split con _preprocess_wikisql para obtener:
         - input_ids  (batch_size, FULL_MAX)
         - attention_mask (batch_size, FULL_MAX)
         - labels     (batch_size, FULL_MAX)  con -100 en la parte del prompt.
    5) Devuelve (train_dataset, eval_dataset).
    """
    print('\n✅ Loading WikiSQL...')
    raw = load_dataset(
        "Salesforce/wikisql",
        revision="5de818eb69c5385e763201d9b9abc36df69a81dc",
        token=hf_token,
        trust_remote_code=True
    )

    # Printa n ejemplos random
    _print_dataset_examples(raw)

    # Muestreo (si env variable DATASET_SIZE_RATIO está definida)
    if dataset_size_ratio is not None:
        ratio = float(dataset_size_ratio)
        if not (0 < ratio <= 100):
            raise ValueError("dataset_size_ratio must be between 0 and 100")

        train_total = len(raw["train"])
        eval_total  = len(raw["validation"])

        train_sample_size = int(train_total * ratio / 100)
        eval_sample_size  = int(eval_total  * ratio / 100)

        if ratio < 100:
            print(f"\n✅ Reducing dataset to {ratio}%")
            print(f"Train samples: {train_sample_size}")
            print(f"Eval samples:  {eval_sample_size}")

        train_raw = raw["train"].select(range(train_sample_size))
        eval_raw  = raw["validation"].select(range(eval_sample_size))
    else:
        train_raw = raw["train"]
        eval_raw  = raw["validation"]

    # Aplica map con la función de pre procesamiento
    train_dataset = train_raw.map(
        lambda batch: _preprocess_wikisql_without_tokenizer(batch),
        batched=True,
        remove_columns=train_raw.column_names
    )
    eval_dataset = eval_raw.map(
        lambda batch: _preprocess_wikisql_without_tokenizer(batch),
        batched=True,
        remove_columns=eval_raw.column_names
    )

    return train_dataset, eval_dataset
