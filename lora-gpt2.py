from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from peft import PeftModel
import torch



print("Versión de torch:", torch.__version__)
print("Versión de CUDA en torch:", torch.version.cuda)
print("CUDA disponible:", torch.cuda.is_available())
print("Número de GPUs detectadas:", torch.cuda.device_count())




model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    #load_in_8bit=True
)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["c_attn"],
    lora_dropout=0.2
)
lora_model = get_peft_model(model, lora_config)


dataset = load_dataset("wikitext", "wikitext-2-raw-v1")


train_subset = dataset["train"].select(range(500))
eval_subset = dataset["validation"].select(range(50))
# Tokenization function
def preprocess_function(batch):
    tokenized = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized
train_dataset = train_subset.map(preprocess_function, batched=True)
eval_dataset = eval_subset.map(preprocess_function, batched=True)


training_args = TrainingArguments(
    output_dir="./gpt2-lora-results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    eval_steps=10,
    save_steps=10,
    logging_steps=10,
    learning_rate=5e-5,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()

lora_model.save_pretrained("./gpt2-lora-adapters")


base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",           # evita el get_default_device()
    torch_dtype=torch.float16,   # opcional, coincide con tu workflow fp16
)


loaded_lora_model = PeftModel.from_pretrained(base_model, "./gpt2-lora-adapters")

# Pick your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your base model + LoRA adapters
loaded_lora_model = PeftModel.from_pretrained(base_model, "./gpt2-lora-adapters")
loaded_lora_model.to(device)                     # <<< move model to device
loaded_lora_model.eval()                         # optional: set eval mode

# Prepare your inputs on the same device
input_text = "Explain the significance of the industrial revolution."
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
).to(device)                                     # <<< move inputs to device

# Generate, explicitly passing pad_token_id
outputs = loaded_lora_model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id         # avoid warning about open-end
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

