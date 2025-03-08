import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

MODEL_PATH = "../gpt2-finetuned-wikitext-103_v1"

# Try to load the fine-tuned model, fallback to default GPT-2
try:
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    print("Fine-tuned model loaded successfully.")
except Exception as e:
    print(f"Error loading fine-tuned model: {e}\nLoading default GPT-2 instead...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Set model to evaluation mode
model.eval()

# Sample text
sample_text = "Completed in 1889, the Eiffel Tower is an iron lattice tower, which means that it supports its own weight with crisscrossed material"

print(f"\nSample Text:\n{sample_text}\n")

# Tokenize input and move to the correct device
inputs = tokenizer(sample_text, return_tensors="pt")
inputs = {key: value.to(model.device) for key, value in inputs.items()}

# Generate text
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,  # Adjusted for better creativity
        top_k=50,
        top_p=0.95,
        do_sample=True
    )

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated Continuation:\n{generated_text}")
