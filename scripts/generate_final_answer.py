from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load top chunk selected by LLM-as-a-Judge
with open("llm_top_chunk.txt", "r", encoding="utf-8") as f:
    context = f.read()

# Original query
query = "How can Klue help sales teams perform better?"

# Prompt for generation
prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

# Load Flan-T5 model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()

# Tokenize and generate answer
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=100)

# Decode and print answer
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n Final Answer:\n")
print(answer)

# Save output to file (optional)
with open("final_answer.txt", "w", encoding="utf-8") as f:
    f.write(answer)  
