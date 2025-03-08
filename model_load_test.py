import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("../gpt2-finetuned-wikitext-103_v1")
tokenizer = GPT2Tokenizer.from_pretrained("../gpt2-finetuned-wikitext-103_v1")

model.eval()


sample_text = "The Gregorian Tower (Italian: Torre Gregoriana) or Tower of the Winds (Italian: Torre dei Venti) is a round tower located"

print(f"Sample Text:\n{sample_text}\n")

inputs = tokenizer(sample_text, return_tensors="pt")

inputs = {key: value.to(model.device) for key, value in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.4,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated continuation:\n{generated_text}")

