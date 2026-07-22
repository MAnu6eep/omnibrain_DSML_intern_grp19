from uuid import uuid4

from fastembed import TextEmbedding
from qdrant_client.models import PointStruct

from omnibrain.vectorstore.collections import TEXT_COLLECTION
from omnibrain.vectorstore.qdrant_client import QdrantClientWrapper

embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")


def index_text_chunks(chunks: list[str]) -> bool:
    """
    Generate BGE embeddings and upsert into the Qdrant text collection.
    """

    if not chunks:
        return False

    client = QdrantClientWrapper().client()

    embeddings = list(embedding_model.embed(chunks))

    points = []

    for text, vector in zip(chunks, embeddings):
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector.tolist(),
                payload={
                    "text": text,
                },
            )
        )

    client.upsert(
        collection_name=TEXT_COLLECTION,
        points=points,
    )

    return True