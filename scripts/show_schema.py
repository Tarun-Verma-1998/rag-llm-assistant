from pymilvus import connections, Collection

connections.connect("default", host="127.0.0.1", port="19530")
collection = Collection("rag_chunks_t5")
print("\n📋 Collection Schema Fields:")
for field in collection.schema.fields:
    print(f" - {field.name} ({field.dtype})")
