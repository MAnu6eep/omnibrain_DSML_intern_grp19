"""
Simple utility to test multi-modal vector search in Qdrant.
"""

from fastembed import TextEmbedding

from omnibrain.vectorstore.collections import IMAGE_COLLECTION, TEXT_COLLECTION
from omnibrain.vectorstore.qdrant_client import QdrantClientWrapper


def search_text(client, query: str):
    """
    Search the text collection using a text query.
    """

    model = TextEmbedding()
    query_vector = list(model.embed([query]))[0].tolist()

    response = client.query_points(
        collection_name=TEXT_COLLECTION,
        query=query_vector,
        limit=5,
    )

    results = response.points

    print("\n========== TEXT SEARCH RESULTS ==========")

    if not results:
        print("No matching text chunks found.")
        return

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Score : {result.score:.4f}")
        print("Payload :", result.payload)


def search_image(client):
    """
    Search the image collection using a dummy 512-dimensional vector.
    Replace this with a real CLIP embedding later.
    """

    dummy_vector = [0.0] * 512

    response = client.query_points(
        collection_name=IMAGE_COLLECTION,
        query=dummy_vector,
        limit=5,
    )

    results = response.points

    print("\n========== IMAGE SEARCH RESULTS ==========")

    if not results:
        print("No matching images found.")
        return

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Score : {result.score:.4f}")
        print("Payload :", result.payload)


def main():
    """
    Run multi-modal retrieval validation.
    """

    client = QdrantClientWrapper().client()

    print("\nRunning Multi-Modal Retrieval Validation...\n")

    search_text(client, "What is Retrieval Augmented Generation?")

    search_image(client)

    print("\nValidation completed.")


if __name__ == "__main__":
    main()
