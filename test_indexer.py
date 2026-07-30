from omnibrain.vectorstore.indexers import index_text_chunks

chunks = [
    "Artificial Intelligence is transforming healthcare.",
    "Qdrant stores vector embeddings.",

]

result = index_text_chunks(chunks)

print(result)
