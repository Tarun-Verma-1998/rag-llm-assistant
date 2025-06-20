from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Step 1: Load top chunk
with open("top_chunk.txt", "r", encoding="utf-8") as f:
    top_chunk = f.read()

# Step 2: Define user query
query = "How can Klue help sales teams perform better?"

# Step 3: Combine into a prompt
prompt = f"Context: {top_chunk}\n\nQuestion: {query}\n\nAnswer:"

# Step 4: Load Flan-T5 model (you can use 'google/flan-t5-base' or 'flan-t5-small' if system is low on RAM)
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Step 5: Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=150)

# Step 6: Decode and display
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n Final Answer:\n")
print(answer)
