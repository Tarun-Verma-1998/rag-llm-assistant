from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm
from pymilvus import utility
import torch
import os

# Import your chunking logic
from chunk_documents import process_all_documents

# Step 1: Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# Step 2: Define collection name
collection_name = "rag_chunks_t5"

# Step 3: Drop collection if exists (clean slate)
if utility.has_collection(collection_name):
    Collection(collection_name).drop()

# Step 4: Define schema
fields = [
    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True, auto_id=False),
    FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields, description="RAG chunks with T5 embeddings")
collection = Collection(name=collection_name, schema=schema)

# Step 5: Load and encode chunks
chunks = process_all_documents()
texts = [c["chunk_text"] for c in chunks]
ids = [c["chunk_id"] for c in chunks]

# Load T5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5EncoderModel.from_pretrained("t5-base").to(device)
model.eval()

# Mean pooling function
def mean_pooling(output, mask):
    token_embeddings = output.last_hidden_state
    mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1)

# Encode all chunks
def encode_texts(texts):
    embeddings = []
    with torch.no_grad():
        for text in tqdm(texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            output = model(**inputs)
            emb = mean_pooling(output, inputs["attention_mask"])
            embeddings.append(emb.squeeze().cpu().numpy())
    return embeddings

print("Encoding with T5...")
embeddings = encode_texts(texts)

# Step 6: Insert into Milvus
print("Inserting into Milvus...")
collection.insert([ids, texts, embeddings])
print(f"Inserted {len(embeddings)} chunks into Milvus")
