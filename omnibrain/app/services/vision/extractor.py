import hashlib
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import fitz


def find_caption(page, image_rect, max_distance=50):
    """Find the nearest probable figure caption below an image."""

    blocks = page.get_text("dict")["blocks"]
    caption = None
    best_distance = float("inf")

    for block in blocks:
        if block["type"] != 0:
            continue

        bbox = block["bbox"]

        # Ignore text above the image
        if bbox[1] < image_rect.y1:
            continue

        distance = bbox[1] - image_rect.y1

        # Ignore text too far below
        if distance > max_distance:
            continue

        lines = []

        for line in block["lines"]:
            spans = [
                span["text"].strip()
                for span in line["spans"]
                if span["text"].strip()
            ]

            if spans:
                lines.append(" ".join(spans))

        text = " ".join(lines).strip()

        # Accept probable figure/table captions
        if text.startswith(("Fig.", "Figure", "Table")):
            if distance < best_distance:
                caption = text
                best_distance = distance

    return caption


def extract_images(
    pdf_path: str,
    output_dir: str = "output/images",
) -> List[Dict[str, Any]]:
    """
    Extract embedded images from a PDF and save them to output_dir.

    Combines SHA256 duplicate detection, bounding boxes [x, y, w, h],
    and metadata required by Qdrant citation tracking.
    """

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted_images: List[Dict[str, Any]] = []
    seen_hashes = set()

    # Stable document identifier based on the PDF path
    document_hash = hashlib.sha256(
        str(pdf_file.resolve()).encode("utf-8")
    ).hexdigest()[:16]
    document_id = f"doc_{document_hash}"

    with fitz.open(pdf_path) as document:

        for page_index in range(len(document)):
            page = document[page_index]
            page_number = page_index + 1
            image_list = page.get_images(full=True)

            for image_index, image_info in enumerate(
                image_list,
                start=1,
            ):
                try:
                    xref = image_info[0]
                    image_data = document.extract_image(xref)

                    if not image_data:
                        continue

                    image_bytes = image_data["image"]
                    image_ext = image_data.get("ext", "png")

                    # SHA256 hash for duplicate detection
                    image_hash = hashlib.sha256(image_bytes).hexdigest()
                    if image_hash in seen_hashes:
                        continue
                    seen_hashes.add(image_hash)

                    image_id = str(uuid4())
                    image_filename = (
                        f"page_{page_number}_image_{image_index}.{image_ext}"
                    )
                    image_file = output_path / image_filename
                    image_file.write_bytes(image_bytes)

                    # Bounding box & caption extraction
                    image_rect = None
                    try:
                        rects = page.get_image_rects(xref)
                        if rects:
                            image_rect = rects[0]
                    except Exception:
                        image_rect = None

                    caption = None
                    bbox = None

                    if image_rect is not None:
                        caption = find_caption(page, image_rect)
                        bbox = [
                            float(image_rect.x0),
                            float(image_rect.y0),
                            float(image_rect.width),
                            float(image_rect.height),
                        ]

                    width = image_data.get("width", 0)
                    height = image_data.get("height", 0)

                    extracted_images.append(
                        {
                            "image_id": image_id,
                            "document_id": document_id,
                            "page_number": page_number,
                            "image_path": str(image_file),
                            "dimensions": (int(width), int(height)),
                            "bbox": bbox,
                            "source": pdf_file.name,
                            "source_path": str(pdf_file.resolve()),
                            "caption": caption,
                            "image_bytes": None,
                            "modality": "image",
                        }
                    )

                except Exception as exc:
                    print(
                        f"Failed to extract image from page {page_number}, "
                        f"image {image_index}: {exc}"
                    )

    return extracted_images