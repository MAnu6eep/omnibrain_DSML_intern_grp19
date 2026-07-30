from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

TEXT_COLLECTION = "omnibrain_text_chunks"
IMAGE_COLLECTION = "omnibrain_image_embeddings"

TEXT_VECTOR_SIZE = 384  # bge-small-en-v1.5
IMAGE_VECTOR_SIZE = 512  # CLIP embeddings


def create_collections(client: QdrantClient) -> None:
    """
    Create the text and image collections in Qdrant
    if they do not already exist.
    """

    existing_collections = {
        collection.name for collection in client.get_collections().collections
    }

    if TEXT_COLLECTION not in existing_collections:
        client.create_collection(
            collection_name=TEXT_COLLECTION,
            vectors_config=VectorParams(
                size=TEXT_VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection: {TEXT_COLLECTION}")

    else:
        print(f"Collection already exists: {TEXT_COLLECTION}")

    if IMAGE_COLLECTION not in existing_collections:
        client.create_collection(
            collection_name=IMAGE_COLLECTION,
            vectors_config=VectorParams(
                size=IMAGE_VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection: {IMAGE_COLLECTION}")

    else:
        print(f"Collection already exists: {IMAGE_COLLECTION}")

    print("Qdrant collections initialized successfully.")
