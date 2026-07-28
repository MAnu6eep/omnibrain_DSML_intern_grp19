from omnibrain.vectorstore.indexers import index_text_chunks

chunks = [
    "Artificial Intelligence is transforming healthcare.",
    "Qdrant stores vector embeddings.",
    "Testing OmniBrain indexing.",
]

result = index_text_chunks(chunks)

print("Indexing Result:", result)
