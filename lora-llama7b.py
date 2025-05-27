from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import torch

print("Versión de torch:", torch.__version__)
print("Versión de CUDA en torch:", torch.version.cuda)
print("CUDA disponible:", torch.cuda.is_available())
print("Número de GPUs detectadas:", torch.cuda.device_count())

# --- Cambia aquí al checkpoint de LLaMA-7B en HF ---
model_name = "meta-llama/Llama-2-7b-hf"

# Tokenizer (LLaMA recomienda use_fast=False)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Carga el modelo base en 16-bit y con device_map automatico
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,       # ⬅︎ opcional para ahorrar memoria
)

# Configuración de LoRA para LLaMA: aten. projections q,k,v,o
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.2,
)
lora_model = get_peft_model(model, lora_config)

# Carga dataset y prepara subconjuntos
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_subset = dataset["train"].select(range(500))
eval_subset  = dataset["validation"].select(range(50))

def preprocess_function(batch):
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

train_dataset = train_subset.map(preprocess_function, batched=True)
eval_dataset  = eval_subset.map(preprocess_function, batched=True)

# Argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="./llama7b-lora-results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    eval_steps=10,
    save_steps=10,
    logging_steps=10,
    learning_rate=5e-5,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()

# Guarda los adaptadores
lora_model.save_pretrained("./llama7b-lora-adapters")


# === Inferencia con LoRA sobre LLaMA-7B ===

# Primero carga de nuevo el base model con LoRA
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
)
loaded_lora_model = PeftModel.from_pretrained(
    base_model,
    "./llama7b-lora-adapters",
)

# Mueve a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_lora_model.to(device)
loaded_lora_model.eval()

# Prepara input y genera
input_text = "Explain the significance of the industrial revolution."
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
).to(device)

outputs = loaded_lora_model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

