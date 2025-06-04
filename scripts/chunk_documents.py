import os
import re

def load_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                text = f.read()
                docs.append({"filename": filename, "text": text})
    return docs

def chunk_text(text, chunk_size=200):
    # Remove extra newlines and split by sentences
    text = re.sub(r"\n+", " ", text)
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence.split()) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def process_all_documents():
    folder_path = "./data/company_docs"
    all_docs = load_documents(folder_path)

    all_chunks = []
    for doc in all_docs:
        chunks = chunk_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_text": chunk,
                "source": doc["filename"],
                "chunk_id": f"{doc['filename']}_chunk{idx}"
            })
    
    return all_chunks

if __name__ == "__main__":
    chunks = process_all_documents()
    for c in chunks:
        print(c["chunk_id"], "=>", c["chunk_text"][:80], "...")
