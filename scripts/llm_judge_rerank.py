from pymilvus import connections, Collection
from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
import torch

# Step 1: Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")
collection = Collection("rag_chunks_t5")
collection.load()

# Step 2: Define query
query = "How can Klue help sales teams perform better?"

# Step 3: Load T5 encoder for embedding generation
tokenizer_embed = T5Tokenizer.from_pretrained("t5-base")
encoder = T5EncoderModel.from_pretrained("t5-base")
encoder.eval()

# Step 4: Encode the query to get embedding
inputs = tokenizer_embed(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
with torch.no_grad():
    query_embedding = encoder(**inputs).last_hidden_state.mean(dim=1).numpy()

# Step 5: Milvus Search
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(
    data=query_embedding,
    anns_field="embedding",
    param=search_params,
    limit=3,
    output_fields=["chunk_text"]
)

# Step 6: Extract Chunks
chunks = [hit.entity.get("chunk_text") for hit in results[0]]

# Step 7: Load Flan-T5 for LLM-as-a-Judge
judge_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
judge_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
judge_model.eval()

# Step 8: Scoring with LLM-as-a-Judge
scored_chunks = []
for chunk in chunks:
    prompt = f"Query: {query} Answer: {chunk} Score from 1 to 10:"
    inputs = judge_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = judge_model.generate(**inputs, max_new_tokens=5)
        score_text = judge_tokenizer.decode(output[0], skip_special_tokens=True)
    try:
        score = float(score_text.strip())
    except:
        score = 0.0
    scored_chunks.append((chunk, score))

# Step 9: Sort and display
scored_chunks.sort(key=lambda x: x[1], reverse=True)
print("\n LLM-as-a-Judge Reranked Results:\n")
for i, (chunk, score) in enumerate(scored_chunks, 1):
    print(f"Rank {i} | Score: {score}\n{chunk}\n")

# Step 10: Save top chunk for final answer generation
top_chunk = scored_chunks[0][0]
with open("llm_top_chunk.txt", "w", encoding="utf-8") as f:
    f.write(top_chunk)
