from functools import lru_cache
from typing import Any

from fastembed import TextEmbedding

from omnibrain.vectorstore.collections import TEXT_COLLECTION
from omnibrain.vectorstore.qdrant_client import QdrantClientWrapper


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


@lru_cache(maxsize=1)
def get_client():
    return QdrantClientWrapper().client()


def _clean_result(payload: dict[str, Any], score: float, point_id: Any) -> dict:
    text = (payload.get("text") or "").strip()

    if not text:
        return {}

    page_number = payload.get("page_number", payload.get("page", 0))

    return {
        "chunk_id": payload.get("chunk_id", str(point_id)),
        "document": payload.get(
            "document",
            payload.get("source", "Unknown Document"),
        ),
        "page": page_number,
        "text": text,
        "score": round(float(score), 4),
        "source": payload.get("source", payload.get("document", "")),
        "modality": payload.get("modality", "text"),
        "metadata": {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "text",
                "chunk_id",
                "document",
                "page_number",
                "page",
                "source",
                "modality",
            }
        },
    }


def search_text_chunks(
    query: str,
    top_k: int = 5,
    min_score: float = 0.20,
) -> list[dict]:
    """
    Search the text collection for the most similar chunks.
    Deduplicates results by normalized text content to prevent duplicate temp file chunks.
    """

    try:
        query_vector = list(get_embedding_model().embed([query]))[0]

        # Fetch candidate pool to allow text content deduplication
        candidate_limit = max(top_k * 5, 30)

        response = get_client().query_points(
            collection_name=TEXT_COLLECTION,
            query=query_vector,
            limit=candidate_limit,
            with_payload=True,
        )

        results = []
        seen_texts = set()

        for point in response.points:
            if point.score < min_score:
                continue

            payload = point.payload or {}

            cleaned = _clean_result(
                payload,
                point.score,
                point.id,
            )

            if not cleaned:
                continue

            # Deduplicate by normalized text content (first 120 chars)
            norm_text = cleaned["text"].strip().lower()[:120]
            if norm_text in seen_texts:
                continue

            seen_texts.add(norm_text)
            results.append(cleaned)

            if len(results) >= top_k:
                break

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
        print(f"Chunk ID : {item['chunk_id']}")
        print(f"Document : {item['document']}")
        print(f"Page     : {item['page']}")
        print(f"Score    : {item['score']}")
        print("Text:")
        print(item["text"])
        print()
