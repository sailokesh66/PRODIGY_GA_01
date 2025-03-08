import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

train_dataset = dataset["train"].select(range(500))
val_dataset = dataset["validation"].select(range(500))
test_dataset = dataset["test"].select(range(500))

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(data):
    return tokenizer(data["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=512)


# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set dataset format
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Initialize model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_dataloader = DataLoader(val_dataset, batch_size=2)
print(len(train_dataloader))


scaler = GradScaler()
count=0
for epoch in range(5):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        with autocast():
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        count += 1
        print(f"Step {count}, Loss: {loss.item()}")

    model.eval()
    total_eval_loss = 0
    for batch in valid_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            with autocast():
                outputs = model(**batch, labels=batch["input_ids"])
                total_eval_loss += outputs.loss.item()

    print(f"Epoch {epoch+1}, Validation Loss: {total_eval_loss / len(valid_dataloader)}")

model.save_pretrained("./gpt2-finetuned-wikitext-103_v1")
tokenizer.save_pretrained("./gpt2-finetuned-wikitext-103_v1")
