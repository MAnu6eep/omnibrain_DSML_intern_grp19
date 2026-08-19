from functools import lru_cache
from typing import Any

from fastembed import TextEmbedding
from qdrant_client.models import FieldCondition, Filter, MatchValue

from omnibrain.vectorstore.collections import TEXT_COLLECTION
from omnibrain.vectorstore.qdrant_client import QdrantClientWrapper


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    """Return the cached BGE embedding model."""

    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


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
            if key
            not in {
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
    source_name: str | None = None,
) -> list[dict]:
    """
    Search the Qdrant text collection.

    If document_id or source_name is provided, retrieval is strictly restricted
    to that specific document asset.
    """
    try:
        query_vector = list(get_embedding_model().embed([query]))[0]

        target = source_name or document_id
        query_filter = None

        if target:
            from pathlib import Path

            clean_target = Path(target).name
            query_filter = Filter(
                should=[
                    FieldCondition(key="source", match=MatchValue(value=clean_target)),
                    FieldCondition(
                        key="document_id", match=MatchValue(value=clean_target)
                    ),
                    FieldCondition(
                        key="document", match=MatchValue(value=clean_target)
                    ),
                ]
            )

        candidate_limit = max(top_k * 5, 30)

        client = get_client()
        response = client.query_points(
            collection_name=TEXT_COLLECTION,
            query=query_vector,
            query_filter=query_filter,
            limit=candidate_limit,
            with_payload=True,
        )

        # Fallback to unfiltered search if filtered search returned 0 results
        if not response.points and query_filter is not None:
            response = client.query_points(
                collection_name=TEXT_COLLECTION,
                query=query_vector,
                query_filter=None,
                limit=candidate_limit,
                with_payload=True,
            )

        results = []
        seen_texts = set()
        seen_chunk_ids = set()

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
            # Deduplicate by chunk_id and normalized text
            # -----------------------------------------------------
            chunk_id = cleaned.get("chunk_id")
            norm_text = cleaned["text"].strip().lower()[:120]

            if (chunk_id and chunk_id in seen_chunk_ids) or (
                norm_text and norm_text in seen_texts
            ):
                continue

            if chunk_id:
                seen_chunk_ids.add(chunk_id)
            if norm_text:
                seen_texts.add(norm_text)

            results.append(cleaned)

            if len(results) >= top_k:
                break

        return results

    except Exception as e:
        print(f"Error during text retrieval: {e}")
        return []


def get_indexed_documents() -> list[str]:
    """
    Scrolls Qdrant points to discover all unique document names currently indexed in Qdrant.
    """
    try:
        client = get_client()
        points, _ = client.scroll(
            collection_name=TEXT_COLLECTION,
            limit=500,
            with_payload=True,
            with_vectors=False,
        )
        doc_names = set()
        for pt in points:
            payload = pt.payload or {}
            doc_id = (
                payload.get("source")
                or payload.get("document_id")
                or payload.get("document")
            )
            if doc_id:
                doc_names.add(doc_id)
        return sorted(list(doc_names))
    except Exception:
        return []
