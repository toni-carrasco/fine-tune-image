import os
import sys
from datasets import load_dataset

def _preprocess_wikisql(tokenizer, batch):
    # pull out the human_readable SQL from the nested dict
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


def get_wikisql_datasets(tokenizer, hf_token, dataset_size_ratio=None):
    # Dataset
    print('Loading WikiSQL…')
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

    print(rows)

    dataset_size_ratio = float(dataset_size_ratio)
    if dataset_size_ratio is not None:
        train_total = len(raw["train"])
        eval_total = len(raw["validation"])

        train_sample_size = int(train_total * dataset_size_ratio / 100)
        eval_sample_size  = int(eval_total  * dataset_size_ratio / 100)

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
