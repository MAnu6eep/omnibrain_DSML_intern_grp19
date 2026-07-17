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
        # Load FastEmbed model
        self.model = TextEmbedding()

    def index(self, client, chunks):
        """
        Convert text chunks into embeddings and store them in Qdrant.

        Parameters
        ----------
        client : QdrantClient
            Initialized Qdrant client.

        chunks : list[dict]
            Example:
            [
                {
                    "chunk_id": 1,
                    "text": "This is sample text.",
                    "page_number": 2,
                    "source_pdf": "sample.pdf"
                }
            ]
        """

        points = []

        for chunk in chunks:
            # Generate embedding
            embedding = list(
                self.model.embed([chunk["text"]])
            )[0].tolist()

            # Create Qdrant point
            point = PointStruct(
                id=chunk["chunk_id"],
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "page_number": chunk["page_number"],
                    "source_pdf": chunk["source_pdf"],
                },
            )

            points.append(point)

        # Upload to Qdrant
        client.upsert(
            collection_name=TEXT_COLLECTION,
            points=points,
        )

        print(f"Successfully indexed {len(points)} text chunks.")