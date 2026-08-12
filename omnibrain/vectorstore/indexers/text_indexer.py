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
        _embedding_model = TextEmbedding(
            "BAAI/bge-small-en-v1.5"
        )

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
    """
    Generate BGE embeddings and upsert text chunks into Qdrant.

    Required Qdrant payload metadata:

        - document_id
        - chunk_id
        - page_number
        - source
        - document
        - modality
        - source_path

    document_id is used to scope retrieved context to the
    uploaded document.
    """

    if not chunks:
        return False

    client = QdrantClientWrapper().client()

    _ensure_collection_exists(client)

    model = get_embedding_model()

    text_strings = [
        (
            c.text
            if hasattr(c, "text")
            else (
                c.get("text")
                if isinstance(c, dict)
                else str(c)
            )
        )
        for c in chunks
    ]

    embeddings = list(
        model.embed(text_strings)
    )

    points = []

    for chunk_obj, text_str, vector in zip(
        chunks,
        text_strings,
        embeddings,
    ):

        # =====================================================
        # Object-based chunk
        # =====================================================

        if hasattr(chunk_obj, "metadata"):

            metadata = dict(
                getattr(
                    chunk_obj,
                    "metadata",
                    {},
                )
                or {}
            )

            document_id = metadata.get(
                "document_id",
                getattr(
                    chunk_obj,
                    "document_id",
                    "",
                ),
            )

            chunk_id = getattr(
                chunk_obj,
                "chunk_id",
                metadata.get(
                    "chunk_id",
                    str(uuid4()),
                ),
            )

            page_number = getattr(
                chunk_obj,
                "page_number",
                metadata.get(
                    "page_number",
                    1,
                ),
            )

            source = getattr(
                chunk_obj,
                "source",
                metadata.get(
                    "source",
                    "",
                ),
            )

            document = getattr(
                chunk_obj,
                "document",
                source
                or metadata.get(
                    "document",
                    "Unknown Document",
                ),
            )

            modality = getattr(
                chunk_obj,
                "modality",
                metadata.get(
                    "modality",
                    "text",
                ),
            )

            source_path = getattr(
                chunk_obj,
                "source_path",
                metadata.get(
                    "source_path",
                    "",
                ),
            )

            payload = {
                # Main content
                "text": text_str,

                # =============================================
                # Task 1: Document isolation
                # =============================================
                "document_id": document_id,

                # =============================================
                # Task 2: Citation metadata
                # =============================================
                "chunk_id": chunk_id,
                "page_number": page_number,
                "source": source,

                # Additional metadata
                "document": document,
                "source_path": source_path,
                "modality": modality,

                # Preserve all existing metadata
                **metadata,
            }

            # Ensure required fields cannot be overwritten
            # by arbitrary metadata values.
            payload["document_id"] = document_id
            payload["chunk_id"] = chunk_id
            payload["page_number"] = page_number
            payload["source"] = source
            payload["document"] = document
            payload["source_path"] = source_path
            payload["modality"] = modality

        # =====================================================
        # Dictionary-based chunk
        # =====================================================

        elif isinstance(chunk_obj, dict):

            metadata = dict(
                chunk_obj.get(
                    "metadata",
                    {},
                )
                or {}
            )

            document_id = chunk_obj.get(
                "document_id",
                metadata.get(
                    "document_id",
                    "",
                ),
            )

            page_number = chunk_obj.get(
                "page_number",
                metadata.get(
                    "page_number",
                    1,
                ),
            )

            chunk_id = chunk_obj.get(
                "chunk_id",
                metadata.get(
                    "chunk_id",
                    str(uuid4()),
                ),
            )

            source = chunk_obj.get(
                "source",
                metadata.get(
                    "source",
                    chunk_obj.get(
                        "document",
                        "",
                    ),
                ),
            )

            document = chunk_obj.get(
                "document",
                chunk_obj.get(
                    "source",
                    metadata.get(
                        "document",
                        "Unknown Document",
                    ),
                ),
            )

            source_path = chunk_obj.get(
                "source_path",
                metadata.get(
                    "source_path",
                    "",
                ),
            )

            modality = chunk_obj.get(
                "modality",
                metadata.get(
                    "modality",
                    "text",
                ),
            )

            payload = {
                "text": text_str,

                # Task 1
                "document_id": document_id,

                # Task 2
                "chunk_id": chunk_id,
                "page_number": page_number,
                "source": source,

                "document": document,
                "source_path": source_path,
                "modality": modality,

                **metadata,
            }

            # Required fields take precedence
            payload["document_id"] = document_id
            payload["chunk_id"] = chunk_id
            payload["page_number"] = page_number
            payload["source"] = source
            payload["document"] = document
            payload["source_path"] = source_path
            payload["modality"] = modality

        # =====================================================
        # Plain text
        # =====================================================

        else:

            payload = {
                "text": text_str,
                "document_id": "",
                "page_number": 1,
                "chunk_id": str(uuid4()),
                "document": "Unknown Document",
                "source": "",
                "source_path": "",
                "modality": "text",
            }

        # =====================================================
        # Create Qdrant point
        # =====================================================

        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector.tolist(),
                payload=payload,
            )
        )

    # =========================================================
    # Upsert into Qdrant
    # =========================================================

    client.upsert(
        collection_name=TEXT_COLLECTION,
        points=points,
    )

    return True