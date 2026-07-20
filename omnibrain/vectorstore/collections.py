from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

TEXT_COLLECTION = "omnibrain_text_chunks"
IMAGE_COLLECTION = "omnibrain_image_embeddings"


def create_collections(client: QdrantClient):
    """
    Create the text and image collections in Qdrant
    if they do not already exist.
    """

    collections = [
        c.name
        for c in client.get_collections().collections
    ]

    if TEXT_COLLECTION not in collections:
        client.create_collection(
            collection_name=TEXT_COLLECTION,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        )

    if IMAGE_COLLECTION not in collections:
        client.create_collection(
            collection_name=IMAGE_COLLECTION,
            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE,
            ),
        )

    print("Qdrant collections initialized successfully.")