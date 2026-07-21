from uuid import uuid4

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from qdrant_client.models import PointStruct

from omnibrain.vectorstore.collections import IMAGE_COLLECTION
from omnibrain.vectorstore.qdrant_client import QdrantClientWrapper

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)


def index_image_vectors(images: list[str]) -> bool:
    """
    Generate CLIP embeddings and upsert into the Qdrant image collection.
    """

    if not images:
        return False

    client = QdrantClientWrapper().client()

    points = []

    for image_path in images:

        image = Image.open(image_path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt",
        )

        with torch.no_grad():
            embedding = model.get_image_features(**inputs)

        embedding = embedding.squeeze().cpu().numpy()

        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=embedding.tolist(),
                payload={
                    "image_path": image_path,
                },
            )
        )

    client.upsert(
        collection_name=IMAGE_COLLECTION,
        points=points,
    )

    return True