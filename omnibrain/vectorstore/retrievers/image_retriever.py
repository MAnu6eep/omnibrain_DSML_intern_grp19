from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import torch
from transformers import CLIPModel, CLIPProcessor

from omnibrain.vectorstore.collections import IMAGE_COLLECTION
from omnibrain.vectorstore.qdrant_client import QdrantClientWrapper

_processor = None
_model = None


def get_clip_components():
    """
    Lazily loads the CLIP processor and model.
    """

    global _processor, _model

    if _processor is None or _model is None:
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        _model.eval()

    return _processor, _model


@lru_cache(maxsize=1)
def get_client():
    return QdrantClientWrapper().client()


def search_images(query: str, top_k: int = 3) -> List[Dict]:
    """
    Searches the Qdrant image collection using a
    CLIP text embedding.

    Args:
        query:
            Search query.

        top_k:
            Number of results.

    Returns:
        List of image metadata dictionaries.
    """

    if not query or not query.strip():
        return []

    processor, model = get_clip_components()

    client = get_client()

    inputs = processor(text=[query], return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

    # Handle different Transformers versions safely
    if hasattr(outputs, "text_embeds"):
        embedding = outputs.text_embeds
    elif hasattr(outputs, "pooler_output"):
        embedding = outputs.pooler_output
    else:
        embedding = outputs

    embedding = embedding.squeeze().cpu().numpy().tolist()

    results = client.query_points(
        collection_name=IMAGE_COLLECTION, query=embedding, limit=top_k
    )

    images = []

    for hit in results.points:

        payload = hit.payload or {}
        image_path = payload.get("image_path")

        if not image_path:
            continue

        if not Path(image_path).exists():
            continue

        images.append(
            {
                "image_path": image_path,
                "page_number": payload.get("page_number"),
                "caption": payload.get("caption"),
                "source": payload.get("source", ""),
                "modality": payload.get("modality", "image"),
                "metadata": {
                    key: value
                    for key, value in payload.items()
                    if key
                    not in {
                        "image_path",
                        "page_number",
                        "caption",
                        "source",
                        "modality",
                    }
                },
            }
        )

    return images


if __name__ == "__main__":

    results = search_images("Transformer architecture diagram")

    print()

    for image in results:
        print(image)
