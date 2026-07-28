from typing import Any, List
from uuid import uuid4

from fastembed import TextEmbedding
from qdrant_client.models import Distance, PointStruct, VectorParams

from omnibrain.vectorstore.collections import TEXT_COLLECTION
from omnibrain.vectorstore.qdrant_client import QdrantClientWrapper

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _embedding_model


def _ensure_collection_exists(client):
    """Auto-creates text collection if it does not exist in Qdrant."""
    if not client.collection_exists(TEXT_COLLECTION):
        client.create_collection(
            collection_name=TEXT_COLLECTION,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        )


def index_text_chunks(chunks: List[Any]) -> bool:
    """Generate BGE embeddings and upsert into the Qdrant text collection."""

    if not chunks:
        return False

    client = QdrantClientWrapper().client()

    _ensure_collection_exists(client)

    model = get_embedding_model()

    text_strings = [
        (
            c.text
            if hasattr(c, "text")
            else (c.get("text") if isinstance(c, dict) else str(c))
        )
        for c in chunks
    ]

    embeddings = list(model.embed(text_strings))

    points = []

    for chunk_obj, text_str, vector in zip(chunks, text_strings, embeddings):

        # -----------------------------
        # Object-based chunk
        # -----------------------------
        if hasattr(chunk_obj, "metadata"):

            payload = {
                "text": text_str,
                "page_number": getattr(chunk_obj, "page_number", 1),
                "chunk_id": getattr(chunk_obj, "chunk_id", str(uuid4())),
                "document": getattr(
                    chunk_obj,
                    "document",
                    getattr(chunk_obj, "source", "Unknown Document"),
                ),
                "source": getattr(
                    chunk_obj, "source", getattr(chunk_obj, "document", "")
                ),
                "source_path": getattr(chunk_obj, "source_path", ""),
                "modality": getattr(chunk_obj, "modality", "text"),
                **getattr(chunk_obj, "metadata", {}),
            }

        # -----------------------------
        # Dictionary-based chunk
        # -----------------------------
        elif isinstance(chunk_obj, dict):

            payload = {
                "text": text_str,
                "page_number": chunk_obj.get("page_number", 1),
                "chunk_id": chunk_obj.get("chunk_id", str(uuid4())),
                "document": chunk_obj.get(
                    "document",
                    chunk_obj.get("source", "Unknown Document"),
                ),
                "source": chunk_obj.get("source", chunk_obj.get("document", "")),
                "source_path": chunk_obj.get("source_path", ""),
                "modality": chunk_obj.get("modality", "text"),
                **chunk_obj.get("metadata", {}),
            }

        # -----------------------------
        # Plain text
        # -----------------------------
        else:

            payload = {
                "text": text_str,
                "page_number": 1,
                "chunk_id": str(uuid4()),
                "document": "Unknown Document",
                "modality": "text",
            }

        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector.tolist(),
                payload=payload,
            )
        )

    client.upsert(
        collection_name=TEXT_COLLECTION,
        points=points,
    )

    return True
