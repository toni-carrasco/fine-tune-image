import os
import sys
from typing import List, Dict, Any
from datasets import load_dataset

def _preprocess_wikisql(tokenizer, batch):
    questions = batch["question"]
    sql_strs  = [entry["human_readable"] for entry in batch["sql"]]
    texts = [
        f"Translate to SQL:\nQuestion: {q}\nSQL: {s}"
        for q, s in zip(questions, sql_strs)
    ]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def _print_markdown_table(records: List[Dict[str, Any]]) -> None:
    headers = list(records[0].keys())
    widths = [max(len(str(r.get(h, ""))) for r in records + [dict(zip(headers, headers))]) for h in headers]

    # cabecera
    sep_header = "| " + " | ".join(h.ljust(widths[i]) for i,h in enumerate(headers)) + " |"
    sep_line   = "|-" + "-|-".join("-"*w for w in widths) + "-|"
    print(sep_header)
    print(sep_line)

    # filas
    for r in records:
        row = "| " + " | ".join(str(r.get(h, "")).ljust(widths[i]) for i,h in enumerate(headers)) + " |"
        print(row)

def get_wikisql_datasets(tokenizer, hf_token, dataset_size_ratio=None):
    # Dataset
    print('Loading WikiSQL...')
    raw = load_dataset(
        "Salesforce/wikisql",
        token=hf_token,
        trust_remote_code=True
    )

    examples = raw["train"].select(range(5))

    rows = []
    for ex in examples:
        rows.append({
            "question": ex["question"],
            "human_readable_sql": ex["sql"]["human_readable"],
            "table_name": ex["table"]["name"],
            "columns": ", ".join(ex["table"]["header"]),
        })

   #_print_markdown_table(rows)

    if dataset_size_ratio is not None:
        ratio = float(dataset_size_ratio)
        if not (0 < ratio <= 100):
            raise ValueError("dataset_size_ratio must be between 0 and 100")

        train_total = len(raw["train"])
        eval_total = len(raw["validation"])

        train_sample_size = int(train_total * ratio / 100)
        eval_sample_size  = int(eval_total  * ratio / 100)

        print(f'Reducing dataset to {ratio}%')
        print(f'Train samples: {train_sample_size}')
        print(f'Eval samples {eval_sample_size}')

        train_raw = raw["train"].shuffle(seed=42).select(range(train_sample_size))
        eval_raw  = raw["validation"].shuffle(seed=42).select(range(eval_sample_size))
    else:
        train_raw = raw["train"]
        eval_raw  = raw["validation"]

    train_dataset = train_raw.map(
        lambda batch: _preprocess_wikisql(tokenizer, batch),
        batched=True,
        remove_columns=train_raw.column_names
    )
    eval_dataset = eval_raw.map(
        lambda batch: _preprocess_wikisql(tokenizer, batch),
        batched=True,
        remove_columns=eval_raw.column_names
    )

    return train_dataset, eval_dataset
