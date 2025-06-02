import torch
from transformers import AutoTokenizer
from peft import PeftModel
from utils import parse_args, load_env_vars, get_model_config, load_model

def main():
    args = parse_args()
    env = load_env_vars()
    config = get_model_config(args.model, args.peft)

    tokenizer = AutoTokenizer.from_pretrained(config.hf_name, token=env.hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.peft == "qlora":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        bnb_config = None

    base_model = load_model(args.model, env.hf_token, bnb_config)
    peft_model = PeftModel.from_pretrained(base_model, config.adapter_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    peft_model.to(device)
    peft_model.eval()

    print("Escribe un prompt o '/quit' para salir.")
    while True:
        prompt = input(">> ")
        if prompt.strip() == "/quit":
            break
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
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
