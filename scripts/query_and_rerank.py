from pymilvus import connections, Collection
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Step 1: Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")
collection = Collection("rag_chunks_t5")
collection.load()

# Step 2: Define your query
query = "How can Klue help sales teams perform better?"

# Step 3: Load T5 for reranking
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
model.eval()

# Step 4: Embed the query using encoder
inputs = tokenizer(query, return_tensors="pt", padding=True,max_length=512, truncation=True)
with torch.no_grad():
    query_embedding = model.encoder(**inputs).last_hidden_state.mean(dim=1).numpy()

# Step 5: Search Milvus for top-k chunks
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(
    data=query_embedding,
    anns_field="embedding",
    param=search_params,
    limit=3,
    output_fields=["chunk_text"]
)

# Step 6: Extract chunks
chunks = [hit.entity.get("chunk_text") for hit in results[0]]

# Step 7: Rerank using T5
# Step 7: Rerank using T5 (based on probability of 'true')
scored_chunks = []
target_token = "true"
scored_chunks = []
target_token = "true"

for chunk in chunks:
    prompt = f"Query: {query} Document: {chunk} Relevant:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    decoder_input_ids = tokenizer(target_token, return_tensors="pt").input_ids

    with torch.no_grad():
        output = model(**inputs, decoder_input_ids=decoder_input_ids)
        logits = output.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        score = probs[0, tokenizer.convert_tokens_to_ids(target_token)].item()

    scored_chunks.append((chunk, score))


# Sort and print
scored_chunks.sort(key=lambda x: x[1], reverse=True)
print("\n🔎 Reranked Results:\n")
for i, (chunk, score) in enumerate(scored_chunks, 1):
    print(f"Rank {i} | Score: {score:.4f}\n{chunk}\n\n")

# Step 10: Save top chunk for use in final answer generation
top_chunk = scored_chunks[0][0]

# Save to file (optional for next step)
with open("top_chunk.txt", "w", encoding="utf-8") as f:
    f.write(top_chunk)
