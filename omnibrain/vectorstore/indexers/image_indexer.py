from typing import Any, List
from uuid import uuid4

import torch
from PIL import Image
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import CLIPModel, CLIPProcessor

from omnibrain.vectorstore.collections import IMAGE_COLLECTION
from omnibrain.vectorstore.qdrant_client import QdrantClientWrapper

_processor = None
_model = None


def get_clip_components():
    global _processor, _model
    if _processor is None or _model is None:
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return _processor, _model


def _ensure_collection_exists(client):
    """Auto-creates image collection if it does not exist in Qdrant."""
    if not client.collection_exists(IMAGE_COLLECTION):
        client.create_collection(
            collection_name=IMAGE_COLLECTION,
            vectors_config=VectorParams(
                size=512,  # clip-vit-base-patch32 vector dimension
                distance=Distance.COSINE,
            ),
        )


def index_image_vectors(images: List[Any]) -> bool:
    """Generate CLIP embeddings and upsert into the Qdrant image collection."""
    if not images:
        return False

    client = QdrantClientWrapper().client()
    _ensure_collection_exists(client)
    processor, model = get_clip_components()

    points = []

    for img_obj in images:
        if hasattr(img_obj, "image_path"):
            image_path = img_obj.image_path
            page_num = getattr(img_obj, "page_number", 1)
            caption = getattr(img_obj, "caption", None)
        elif isinstance(img_obj, dict):
            image_path = img_obj.get("image_path", "")
            page_num = img_obj.get("page_number", 1)
            caption = img_obj.get("caption", None)
        else:
            image_path = str(img_obj)
            page_num = 1
            caption = None

        if not image_path:
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        # Unpack embedding tensor safely
        if hasattr(outputs, "image_embeds"):
            embedding = outputs.image_embeds
        elif hasattr(outputs, "pooler_output"):
            embedding = outputs.pooler_output
        else:
            embedding = outputs

        embedding_vector = embedding.squeeze().cpu().numpy().tolist()

        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=embedding_vector,
                payload={
                    "image_path": str(image_path),
                    "page_number": page_num,
                    "caption": caption,
                },
            )
        )

    if points:
        client.upsert(
            collection_name=IMAGE_COLLECTION,
            points=points,
        )
        return True

    return False
