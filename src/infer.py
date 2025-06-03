import torch
from transformers import AutoTokenizer
from peft import PeftModel
from utils import parse_args, load_env_vars, get_model_config, load_model, load_peft_arguments_from_json
from wikisql_dataset import get_tokenizer
from train import get_qlora_quantification_config

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

        columns = input("Columns (comma‐separated) >> ")
        columns = ", ".join([col.strip() for col in columns.split(",") if col.strip()])

        combined_prompt = f"Question: {question}\nColumns: {columns}\nSQL:"

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

        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == '__main__':
    main()
