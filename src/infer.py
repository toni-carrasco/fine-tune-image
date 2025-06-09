import torch
import time
import re
from transformers import AutoTokenizer
from peft import PeftModel
from utils import parse_args, load_env_vars, get_model_config, load_model, load_peft_arguments_from_json
from wikisql_dataset import get_wikisql_datasets, get_tokenizer
from train import get_qlora_quantification_config


def outputs_match(expected, inferred):
    def normalize_sql(sql):
        return re.sub(r'\s+', ' ', sql.strip().lower())

    def extract_sql(text: str) -> str:
        parts = text.split("\n")
        for line in parts:
            if line.startswith("SQL:"):
                return line[4:].strip()
        return ""

    def extract_sql_parts(sql):
        pattern = re.compile(r"select (.+?) from .+? where (.+)", re.IGNORECASE)
        match = pattern.search(sql)
        if not match:
            return None, None
        projection = match.group(1).strip()
        where_clause = match.group(2).strip()
        return projection, where_clause

    def compare_sql_statements(sql1, sql2):
        norm_sql1 = normalize_sql(sql1)
        norm_sql2 = normalize_sql(sql2)

        proj1, where1 = extract_sql_parts(norm_sql1)
        proj2, where2 = extract_sql_parts(norm_sql2)

        if proj1 is None or where1 is None or proj2 is None or where2 is None:
            return "Invalid SQL format"

        proj_match = proj1 == proj2
        where_match = where1 == where2

        if proj_match and where_match:
            return "Match"
        elif not proj_match and where_match:
            return "Projection mismatch"
        elif proj_match and not where_match:
            return "Filter mismatch"
        else:
            return "Both projection and filter mismatch"

    inferred_sql = extract_sql(inferred)
    match = compare_sql_statements(inferred_sql, expected)

    return match


def infer(combined_prompt, tokenizer, peft_model, device):
    inputs = tokenizer(
        combined_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    outputs = peft_model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def test_prompt(combined_prompt, expected_output, tokenizer, peft_model, device, logf):
    start_time = time.time()
    inferred_output = infer(combined_prompt, tokenizer, peft_model, device)
    end_time = time.time()

    elapsed_time = end_time - start_time
    match = outputs_match(expected_output, inferred_output)

    if match != "Match":
        log_msg = (
            f"[{match}]\n"
            f"Generated:\n{inferred_output}\n"
            f"Expected:\nSQL: {expected_output}\n"
            f"{'-'*40}"
        )
        if logf:
            logf.write(log_msg + "\n")

    return elapsed_time, match


def perform_test(tokenizer, peft_model, device, hf_token, dataset_size_ratio, output_dir):
    train_dataset, eval_dataset = get_wikisql_datasets(None, hf_token, dataset_size_ratio)

    prompts = eval_dataset["input_ids"]
    sqls = eval_dataset["labels"]

    total_time = 0.0
    total_steps = len(prompts)
    last_printed_percent = -1
    result_counters = {
        "Match": 0,
        "Projection mismatch": 0,
        "Filter mismatch": 0,
        "Both projection and filter mismatch": 0,
        "Invalid SQL format": 0
    }

    print(f"== Starting inference process for evaluation dataset ==\n"
          f"   Total samples to be evaluated: {total_steps}.\n"
          f"   Please wait while the model processes the data.")

    with open(f"{output_dir}/mismatches.log", "w", encoding="utf-8") as logf:
        for i, (p, s) in enumerate(zip(prompts, sqls), 1):
            elapsed, match = test_prompt(p, s, tokenizer, peft_model, device, logf)
            total_time += elapsed

            if match not in result_counters:
                result_counters["Invalid SQL format"] += 1
            else:
                result_counters[match] += 1

            progress = (i / total_steps) * 100
            current_percent = int(progress)
            if current_percent % 5 == 0 and current_percent != last_printed_percent:
                print(
                    f"   Progress: {progress:6.0f}% ({i:>{15}})    "
                    f"Matches: {result_counters['Match']:>{15}}    "
                    f"Mismatches: {sum(v for k, v in result_counters.items() if k != 'Match'):>{15}}    "
                    f"inference time: {total_time:.2f}"
                )
                last_printed_percent = current_percent

    print("== Global Results ==")
    print(f"  Total inference time: {total_time:.2f} seconds\n")
    for category, count in result_counters.items():
        print(f"  {category:<40}: {count}")


def main():
    args = parse_args()
    env = load_env_vars()
    config = get_model_config(args.model, args.peft)
    tokenizer = get_tokenizer(env.hf_token, config)

    if args.peft == "qlora":
        peft_config = load_peft_arguments_from_json("configs/peft_qlora_configuration.json", config.output_dir)
        bnb_config = get_qlora_quantification_config(peft_config)
    else:
        bnb_config = None

    base_model = load_model(args.model, env.hf_token, bnb_config)
    peft_model = PeftModel.from_pretrained(base_model, config.adapter_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    peft_model.to(device)
    peft_model.eval()

    print("Escribe un prompt o '/quit' para salir.")
    while True:
        question = input("Question >> ")
        if question.strip() == "/quit":
            break
        elif question == "/test":
            perform_test(tokenizer, peft_model, device, env.hf_token, env.dataset_size_ratio, config.output_dir)
            break

        columns = input("Columns (comma‐separated) >> ")
        columns = ", ".join([col.strip() for col in columns.split(",") if col.strip()])

        combined_prompt = f"Question: {question}\nColumns: {columns}\nSQL:"

        inferred_output = infer(combined_prompt, tokenizer, peft_model, device)
        print(inferred_output)

if __name__ == '__main__':
    main()
