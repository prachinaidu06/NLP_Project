import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.model_selection import train_test_split

# Base instruction-tuned LLaMA model
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
#merged dataset containing both summary and reply
DATA_FILE = "NLP/llama_train_merged_fixed.jsonl"

TRAIN_FILE = "NLP/train_90.jsonl"
EVAL_FILE  = "NLP/eval_10.jsonl"

LORA_DIR   = "NLP/lora_email_90_10_final"
MERGED_DIR = "NLP/llama_email_merged_90_10_final"

# Training hyperparameters
MAX_LEN = 1024
BATCH_SIZE = 4
GRAD_ACCUM = 8
EPOCHS = 2
LR = 2e-4


full_dataset = load_dataset("json", data_files=DATA_FILE)["train"]
print(f"Total samples: {len(full_dataset)}")

# Split the dataset 
train_idx, eval_idx = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.1,
    random_state=42
)

# Save the split datasets
train_ds = full_dataset.select(train_idx)
eval_ds  = full_dataset.select(eval_idx)

print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

train_ds.to_json(TRAIN_FILE)
eval_ds.to_json(EVAL_FILE)


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


from transformers import BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb
)

model.config.pad_token_id = tokenizer.pad_token_id

# Configure LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def tokenize(example):
    tok = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length"
    )
    tok["labels"] = tok["input_ids"].copy()
    return tok

train_tok = train_ds.map(tokenize, remove_columns=train_ds.column_names)
eval_tok  = eval_ds.map(tokenize, remove_columns=eval_ds.column_names)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training configuration
training_args = TrainingArguments(
    output_dir=LORA_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    bf16=True,
    logging_steps=25,
    save_strategy="epoch",
    do_eval=True,
    report_to="none"
)

# Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=data_collator
)


trainer.train()

model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)

print("LoRA saved")


print("Merging LoRA and Base")

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

merged = PeftModel.from_pretrained(base, LORA_DIR)
merged = merged.merge_and_unload()

# Saves the final merged model for inference
merged.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print("Final model saved to:", MERGED_DIR)
