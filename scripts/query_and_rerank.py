from pymilvus import connections, Collection
from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
import torch

# Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# Load Milvus collection
collection = Collection("rag_chunks_t5")
collection.load()

# Load T5 tokenizer and models
tokenizer = T5Tokenizer.from_pretrained("t5-base")
encoder = T5EncoderModel.from_pretrained("t5-base")
reranker = T5ForConditionalGeneration.from_pretrained("t5-base")
encoder.eval()
reranker.eval()

# Input query
query = "How to optimize model performance?"

# Step 1: Encode query
inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    embedding = encoder.encoder(**inputs).last_hidden_state.mean(dim=1).numpy()

# Step 2: Search in Milvus
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(
    data=embedding,
    anns_field="embedding",
    param=search_params,
    limit=3,
    output_fields=["chunk_text"]
)

# Step 3: Rerank the results using T5
chunks = [hit.entity.get("chunk_text") for hit in results[0]]

scored_chunks = []
for chunk in chunks:
    input_text = f"query: {query} document: {chunk}"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids

    with torch.no_grad():
        output = reranker.generate(
            input_ids,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=1
        )

        # Take confidence of most probable token for scoring
        token_scores = output.scores[0].softmax(dim=-1)
        confidence = token_scores.max().item()

    scored_chunks.append((confidence, chunk))


# Step 4: Sort and display
scored_chunks.sort(reverse=True)
for rank, (score, chunk) in enumerate(scored_chunks, 1):
    print(f"\nRank {rank} | Score: {score:.4f}\n{chunk}\n")
