from pymilvus import connections, Collection

# Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# Load collection
collection = Collection("rag_chunks_t5")

# Create index on embeddings field
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",  # Good default; others: "HNSW", "IVF_SQ8", etc.
    "params": {"nlist": 1024}
}
collection.create_index(field_name="embedding", index_params=index_params)

print("Index created successfully!")
