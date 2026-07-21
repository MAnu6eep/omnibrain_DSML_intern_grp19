"""
Image vector indexing layer for Qdrant.
"""

from qdrant_client.models import PointStruct

from omnibrain.vectorstore.collections import IMAGE_COLLECTION


class ImageIndexer:
    """
    Uploads image embeddings into the Qdrant image collection.
    """

    def index(self, client, images):
        """
        Store image embeddings in Qdrant.

        Expected input:

        [
            {
                "image_id": "page2_img1",
                "embedding": [...],   # 512-dimensional CLIP embedding
                "page": 2,
                "source": "sample.pdf"
            }
        ]
        """

        points = []

        for image in images:
            point = PointStruct(
                id=image["image_id"],
                vector=image["embedding"],
                payload={
                    "image_id": image["image_id"],
                    "page": image["page"],
                    "source": image["source"],
                },
            )

            points.append(point)

        client.upsert(
            collection_name=IMAGE_COLLECTION,
            points=points,
        )

        print(f"Successfully indexed {len(points)} image embeddings into '{IMAGE_COLLECTION}'.")