from functools import lru_cache
from typing import Any

from fastembed import TextEmbedding
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from omnibrain.vectorstore.collections import TEXT_COLLECTION
from omnibrain.vectorstore.qdrant_client import QdrantClientWrapper


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    """Return the cached BGE embedding model."""

    return TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )


@lru_cache(maxsize=1)
def get_client():
    """Return the cached Qdrant client."""

    return QdrantClientWrapper().client()


def _clean_result(
    payload: dict[str, Any],
    score: float,
    point_id: Any,
) -> dict:
    """Convert a Qdrant payload into the application's result format."""

    text = (payload.get("text") or "").strip()

    if not text:
        return {}

    page_number = payload.get(
        "page_number",
        payload.get("page", 0),
    )

    return {
        "chunk_id": payload.get(
            "chunk_id",
            str(point_id),
        ),
        "document_id": payload.get(
            "document_id",
            "",
        ),
        "document": payload.get(
            "document",
            payload.get(
                "source",
                "Unknown Document",
            ),
        ),
        "page": page_number,
        "text": text,
        "score": round(
            float(score),
            4,
        ),
        "source": payload.get(
            "source",
            payload.get(
                "document",
                "",
            ),
        ),
        "modality": payload.get(
            "modality",
            "text",
        ),
        "metadata": {
            key: value
            for key, value in payload.items()
            if key not in {
                "text",
                "chunk_id",
                "document_id",
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
    document_id: str | None = None,
) -> list[dict]:
    """
    Search the Qdrant text collection.

    If document_id is provided, retrieval is strictly restricted
    to that document.

    Parameters
    ----------
    query:
        User's search query.

    top_k:
        Maximum number of results to return.

    min_score:
        Minimum cosine similarity score.

    document_id:
        Authenticated/uploaded document ID used to scope retrieval.
    """

    try:
        # ---------------------------------------------------------
        # Generate query embedding
        # ---------------------------------------------------------
        query_vector = list(
            get_embedding_model().embed([query])
        )[0]

        # ---------------------------------------------------------
        # Build document-level Qdrant filter
        # ---------------------------------------------------------
        query_filter = None

        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        ),
                    )
                ]
            )

        # ---------------------------------------------------------
        # Candidate pool
        # ---------------------------------------------------------
        candidate_limit = max(
            top_k * 5,
            30,
        )

        # ---------------------------------------------------------
        # Qdrant search
        # ---------------------------------------------------------
        response = get_client().query_points(
            collection_name=TEXT_COLLECTION,
            query=query_vector,
            query_filter=query_filter,
            limit=candidate_limit,
            with_payload=True,
        )

        results = []
        seen_texts = set()

        # ---------------------------------------------------------
        # Process results
        # ---------------------------------------------------------
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

            # -----------------------------------------------------
            # Deduplicate by normalized text
            # -----------------------------------------------------
            norm_text = (
                cleaned["text"]
                .strip()
                .lower()[:120]
            )

            if norm_text in seen_texts:
                continue

            seen_texts.add(norm_text)

            results.append(cleaned)

            if len(results) >= top_k:
                break

        return results

    except Exception as e:
        print(
            f"Error during text retrieval: {e}"
        )
        return []


if __name__ == "__main__":

    query = "What is reinforcement learning?"

    # Replace this with an actual document ID
    # when testing document-scoped retrieval.
    document_id = None

    print(
        f"\nSearching for: {query}\n"
    )

    if document_id:
        print(
            f"Document ID filter: {document_id}\n"
        )
    else:
        print(
            "Document ID filter: NOT APPLIED\n"
        )

    results = search_text_chunks(
        query,
        top_k=3,
        document_id=document_id,
    )

    if not results:
        print("No results found.")

    for i, item in enumerate(
        results,
        start=1,
    ):

        print("=" * 60)
        print(f"Result {i}")
        print("=" * 60)

        print(
            f"Chunk ID    : "
            f"{item['chunk_id']}"
        )

        print(
            f"Document ID : "
            f"{item['document_id']}"
        )

        print(
            f"Document    : "
            f"{item['document']}"
        )

        print(
            f"Page        : "
            f"{item['page']}"
        )

        print(
            f"Score       : "
            f"{item['score']}"
        )

        print("Text:")

        print(item["text"])

        print()