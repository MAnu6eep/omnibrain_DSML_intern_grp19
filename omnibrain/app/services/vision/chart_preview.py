from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image


def verify_bbox(
    bbox: Optional[List[float]],
    image_width: int,
    image_height: int,
) -> bool:
    """
    Validate bbox in [x, y, width, height] format.
    """

    if not bbox or len(bbox) != 4:
        return False

    try:
        x = float(bbox[0])
        y = float(bbox[1])
        width = float(bbox[2])
        height = float(bbox[3])
    except (ValueError, TypeError):
        return False

    if width <= 0 or height <= 0:
        return False

    if x < 0 or y < 0:
        return False

    if x + width > image_width:
        return False

    if y + height > image_height:
        return False

    return True


def verify_image_quality(
    image_path: str,
    min_width: int = 200,
    min_height: int = 200,
) -> Dict[str, Any]:
    """
    Verify that an image exists and has sufficient resolution.
    """

    path = Path(image_path)

    if not path.exists():
        return {
            "valid": False,
            "width": 0,
            "height": 0,
            "message": "Image file does not exist.",
        }

    try:
        with Image.open(path) as image:
            width, height = image.size

        valid = width >= min_width and height >= min_height

        return {
            "valid": valid,
            "width": width,
            "height": height,
            "message": (
                "Resolution is sufficient."
                if valid
                else "Image resolution is too low for reliable preview."
            ),
        }

    except Exception as exc:
        return {
            "valid": False,
            "width": 0,
            "height": 0,
            "message": f"Unable to read image: {exc}",
        }


def resolve_image_path(
    image_path: str,
    cropped_image_path: Optional[str] = None,
) -> str:
    """
    Resolve the exact image path used by the chart preview.

    If a cropped image path is supplied, it takes priority.
    Otherwise the original extracted image path is used.
    """

    if cropped_image_path:
        return cropped_image_path

    return image_path


def build_chart_preview(
    image_metadata: Dict[str, Any],
    vlm_result: Optional[Dict[str, Any]] = None,
    cropped_image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Combine extracted image metadata and VLM findings
    into a frontend-ready chart preview object.
    """

    image_path = resolve_image_path(
        image_metadata.get("image_path", ""),
        cropped_image_path,
    )

    dimensions = image_metadata.get("dimensions", (0, 0))

    width = int(dimensions[0]) if len(dimensions) > 0 else 0
    height = int(dimensions[1]) if len(dimensions) > 1 else 0

    bbox = image_metadata.get("bbox")

    bbox_valid = verify_bbox(
        bbox=bbox,
        image_width=width,
        image_height=height,
    )

    resolution = verify_image_quality(image_path)

    return {
        "page_number": image_metadata.get("page_number"),
        "image_path": image_path,
        "caption": image_metadata.get("caption"),
        "dimensions": {
            "width": width,
            "height": height,
        },
        "bbox": bbox,
        "bbox_valid": bbox_valid,
        "resolution": resolution,
        "vlm": vlm_result or {},
    }


def build_chart_preview_list(
    extracted_images: List[Dict[str, Any]],
    vlm_results: Optional[List[Dict[str, Any]]] = None,
    cropped_image_paths: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Pair extracted images with their VLM findings and exact
    cropped image paths.

    Pairing is performed by list position so that the image,
    VLM result, and crop path remain aligned.
    """

    vlm_results = vlm_results or []
    cropped_image_paths = cropped_image_paths or []

    previews = []

    for index, image_metadata in enumerate(extracted_images):
        vlm_result = vlm_results[index] if index < len(vlm_results) else None

        cropped_path = (
            cropped_image_paths[index] if index < len(cropped_image_paths) else None
        )

        previews.append(
            build_chart_preview(
                image_metadata=image_metadata,
                vlm_result=vlm_result,
                cropped_image_path=cropped_path,
            )
        )

    return previews
