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
        _processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        _model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    return _processor, _model


def _ensure_collection_exists(client):
    """Auto-creates image collection if it does not exist."""

    if not client.collection_exists(IMAGE_COLLECTION):
        client.create_collection(
            collection_name=IMAGE_COLLECTION,
            vectors_config=VectorParams(
                size=512,
                distance=Distance.COSINE,
            ),
        )


def _get_value(
    img_obj: Any,
    key: str,
    default: Any = None,
):
    """Safely retrieve metadata from object or dictionary."""

    if isinstance(img_obj, dict):
        return img_obj.get(key, default)

    if hasattr(img_obj, key):
        return getattr(img_obj, key)

    return default


def index_image_vectors(images: List[Any]) -> bool:
    """
    Generate CLIP embeddings and store image vectors in Qdrant.

    Required citation metadata:

        image_id
        document_id
        page_number
        source
        source_path
        image_path
        caption
        modality
    """

    if not images:
        return False

    client = QdrantClientWrapper().client()

    _ensure_collection_exists(client)

    processor, model = get_clip_components()

    points = []

    for idx, img_obj in enumerate(images):

        image_path = _get_value(
            img_obj,
            "image_path",
            "",
        )

        if not image_path:
            continue

        page_number = _get_value(
            img_obj,
            "page_number",
            1,
        )

        caption = _get_value(
            img_obj,
            "caption",
            None,
        )

        document_id = _get_value(
            img_obj,
            "document_id",
            "",
        )

        source = _get_value(
            img_obj,
            "source",
            "",
        )

        source_path = _get_value(
            img_obj,
            "source_path",
            "",
        )

        image_id = _get_value(
            img_obj,
            "image_id",
            None,
        )

        if not image_id:
            image_id = str(uuid4())

        modality = _get_value(
            img_obj,
            "modality",
            "image",
        )

        try:
            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as exc:
            print(
                f"Skipping image {image_path}: {exc}"
            )
            continue

        try:
            inputs = processor(
                images=image,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = model.get_image_features(
                    **inputs
                )

        except Exception as exc:
            print(
                f"Failed to generate CLIP embedding "
                f"for {image_path}: {exc}"
            )
            continue

        if hasattr(outputs, "image_embeds"):
            embedding = outputs.image_embeds

        elif hasattr(outputs, "pooler_output"):
            embedding = outputs.pooler_output

        else:
            embedding = outputs

        embedding_vector = (
            embedding
            .squeeze()
            .cpu()
            .numpy()
            .tolist()
        )

        payload = {
            "image_id": str(image_id),
            "document_id": str(document_id),
            "page_number": int(page_number),
            "source": str(source),
            "source_path": str(source_path),
            "image_path": str(image_path),
            "caption": caption,
            "modality": str(modality),
        }

        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=embedding_vector,
                payload=payload,
            )
        )

    if not points:
        return False

    client.upsert(
        collection_name=IMAGE_COLLECTION,
        points=points,
    )

    return True