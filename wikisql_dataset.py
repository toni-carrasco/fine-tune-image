import os
import sys
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

def _print_dataset_examples(dataset, split: str = "train", n: int = 5):
    """
    Prints n examples from the specified split of the WikiSQL dataset.
    """
    examples = dataset[split].select(range(n))

    print()
    print("Dataset examples:")
    for ex in examples:
        table_name = ex["table"]["name"]
        columns = ", ".join(ex["table"]["header"])
        question = ex["question"]
        hr_sql = ex["sql"]["human_readable"]

        print(f"table:              {table_name}")
        print(f"columns:            {columns}")
        print(f"question:           {question}")
        print(f"human_readable_sql: {hr_sql}")
        print("-" * 80)

def get_wikisql_datasets(tokenizer, hf_token, dataset_size_ratio=None):
    # Dataset
    print('Loading WikiSQL...')
    raw = load_dataset(
        "Salesforce/wikisql",
        token=hf_token,
        trust_remote_code=True
    )

    _print_dataset_examples(raw, split="train", n=5)

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
