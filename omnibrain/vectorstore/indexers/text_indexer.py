"""
Text vector indexing layer for Qdrant.
"""

from fastembed import TextEmbedding
from qdrant_client.models import PointStruct

from omnibrain.vectorstore.collections import TEXT_COLLECTION


class TextIndexer:
    """
    Converts text chunks into embeddings and uploads them
    into the Qdrant text collection.
    """

    def __init__(self):
        self.model = TextEmbedding()

    def index(self, client, chunks):
        """
        Convert text chunks into embeddings and store them in Qdrant.

        Expected input:

        [
            {
                "chunk_id": "page2_chunk7",
                "text": "This is sample text.",
                "page": 2,
                "source": "sample.pdf"
            }
        ]
        """

        points = []

        for chunk in chunks:
            embedding = list(
                self.model.embed([chunk["text"]])
            )[0].tolist()

            point = PointStruct(
                id=chunk["chunk_id"],
                vector=embedding,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "source": chunk["source"],
                },
            )

            points.append(point)

        client.upsert(
            collection_name=TEXT_COLLECTION,
            points=points,
        )

        print(f"Successfully indexed {len(points)} text chunks into '{TEXT_COLLECTION}'.")