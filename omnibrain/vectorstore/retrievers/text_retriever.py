from fastembed import TextEmbedding

from omnibrain.vectorstore.collections import TEXT_COLLECTION
from omnibrain.vectorstore.qdrant_client import QdrantClientWrapper

# Initialize the embedding model (384-dimensional vectors)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Create Qdrant client
client = QdrantClientWrapper().client()


def search_text_chunks(query: str, top_k: int = 5) -> list[dict]:
    """
    Search the text collection for the most similar chunks.

    Args:
        query: User query.
        top_k: Number of results to return.

    Returns:
        List of dictionaries containing text, page, and similarity score.
    """
    try:
        # Generate embedding for the query
        query_vector = list(embedding_model.embed([query]))[0]

        # Perform similarity search
        response = client.query_points(
            collection_name=TEXT_COLLECTION,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}

            results.append(
                {
                    "text": payload.get("text", ""),
                    "page": payload.get("page", "Unknown"),
                    "score": round(point.score, 4),
                }
            )

        return results

    except Exception as e:
        print(f"Error during text retrieval: {e}")
        return []


if __name__ == "__main__":
    query = "What is reinforcement learning?"

    print(f"\nSearching for: {query}\n")

    results = search_text_chunks(query, top_k=3)

    if not results:
        print("No results found.")

    for i, item in enumerate(results, start=1):
        print("=" * 60)
        print(f"Result {i}")
        print("=" * 60)
        print(f"Score : {item['score']}")
        print(f"Page  : {item['page']}")
        print("Text:")
        print(item["text"])
        print()