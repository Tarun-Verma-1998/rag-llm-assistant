from pymilvus import connections, Collection
from transformers import T5Tokenizer, T5EncoderModel
import torch

# Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")


# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5EncoderModel.from_pretrained("t5-base")

# Ensure model is in eval mode and on CPU
model.eval()

# Input query
query_text = "How to optimize model performance?"

# Tokenize and encode
inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    embeddings = model.encoder(**inputs).last_hidden_state.mean(dim=1)
embedding = embeddings.numpy()

# Load collection
collection = Collection("rag_collection")
collection.load()

# Search in Milvus
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(
    data=embedding,
    anns_field="embeddings",
    param=search_params,
    limit=3,
    output_fields=["chunk"]
)

# Print results
for hits in results:
    for hit in hits:
        print(f"Score: {hit.distance:.4f} | Chunk: {hit.entity.get('chunk')}")
