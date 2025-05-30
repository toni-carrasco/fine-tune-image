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


def get_wikisql_datasets(tokenizer, hf_token):
    # Dataset
    print('Loading WikiSQL…')
    raw = load_dataset(
        "Salesforce/wikisql",
        token=hf_token,
        trust_remote_code=True
    )

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
