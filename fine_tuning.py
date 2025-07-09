import sys
import os
from datasets import load_dataset
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
except ImportError as e:
    print("Error: The 'transformers' package is required but not installed.")
    print("Please install it with: pip install transformers")
    sys.exit(1)
import torch

# Check for accelerate package
try:
    import accelerate
except ImportError:
    print("Warning: The 'accelerate' package is not installed. Some features (like fast training or distributed training) may not be available.")
    print("You can install it with: pip install accelerate")

print("Python executable:", sys.executable)
try:
    import transformers
    print("Transformers version:", transformers.__version__)
except Exception as e:
    print("Transformers import error:", e)

# 1. Load the GSM8K dataset
gsm8k = load_dataset("gsm8k", "main")

# 2. Prepare the data
def preprocess(example):
    return {
        "text": f"Question: {example['question']}\nAnswer: {example['answer']}"
    }

gsm8k_train = gsm8k["train"].map(preprocess)
gsm8k_val = gsm8k["test"].map(preprocess)

# 3. Load tokenizer and model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# 4. Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

tokenized_train = gsm8k_train.map(tokenize_function, batched=True)
tokenized_val = gsm8k_val.map(tokenize_function, batched=True)

# 5. Set up Trainer with corrected parameter names
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Fixed parameter name
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

# 6. Fine-tune the model
trainer.train()