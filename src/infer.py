import torch
import time
from transformers import AutoTokenizer
from peft import PeftModel
from utils import parse_args, load_env_vars, get_model_config, load_model, load_peft_arguments_from_json
from wikisql_dataset import get_wikisql_datasets, get_tokenizer
from train import get_qlora_quantification_config


def outputs_match(expected, inferred):

    def extract_sql(text: str) -> str:
        parts = text.split("\n")
        for line in parts:
            if line.startswith("SQL:"):
                return line[4:].strip()
    return ""

    def normalize(text):
        return " ".join(text.lower().split())

    inferred_sql = extract_sql(inferred)
    print("===")
    print(expected)
    print(inferred_sql)
    match = normalize(expected) == normalize(inferred_sql)

    if not match:
        print("== fail")
        print(f"prompt: {combined_prompt}")
        print(f"expected: {expected_output}")
        print(f"inferred: {inferred_sql}")
    else:
        print(f"inferred: {inferred_sql}")

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


def test_prompt(combined_prompt, expected_output, tokenizer, peft_model, device):
    start_time = time.time()
    inferred_output = infer(combined_prompt, tokenizer, peft_model, device)
    end_time = time.time()

    elapsed_time = end_time - start_time
    match = outputs_match(expected_output, inferred_output)

    return elapsed_time, match

def perform_test(tokenizer, peft_model, device, hf_token, dataset_size_ratio):
    train_dataset, eval_dataset = get_wikisql_datasets(tokenizer, hf_token, dataset_size_ratio)

    prompts = eval_dataset["input_ids"]
    sqls = eval_dataset["labels"]

    total_time = 0.0
    match_count = 0
    mismatch_count = 0
    total_steps = len(prompts)
    print(f"== Starting inference process for evaluation dataset ==\n"
          f"   Total samples to be evaluated: {total_steps}.\n"
          f"   Please wait while the model processes the data.")

    for i, (p, s) in enumerate(zip(prompts, sqls), 1):

        elapsed, match = test_prompt(p, s, tokenizer, peft_model, device)
        total_time += elapsed

        if match:
            match_count += 1
        else:
            mismatch_count += 1

        progress = (i / total_steps) * 100
        if progress % 5 == 0:
            print(f"   Progress: {progress:6.0f}%    Matches: {match_count:>{15}}    Mismatches: {mismatch_count:>{15}}")

    print("== Global Results ==")
    print(f"  Total inference time: {total_time:.2f} seconds")
    print(f"  Matches:   {match_count}")
    print(f"  Mismatches: {mismatch_count}")


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
            perform_test(tokenizer, peft_model, device, env.hf_token, env.dataset_size_ratio)
            break

        columns = input("Columns (comma‐separated) >> ")
        columns = ", ".join([col.strip() for col in columns.split(",") if col.strip()])

        combined_prompt = f"Question: {question}\nColumns: {columns}\nSQL:"

        inferred_output = infer(combined_prompt, tokenizer, peft_model, device)
        print(inferred_output)

if __name__ == '__main__':
    main()
