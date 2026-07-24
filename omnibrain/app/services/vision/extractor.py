import hashlib
from pathlib import Path
from typing import Any, Dict, List

import fitz


def find_caption(page, image_rect, max_distance=50):
    """Finds the nearest text block immediately below an image."""
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
                span["text"].strip() for span in line["spans"] if span["text"].strip()
            ]

            if spans:
                lines.append(" ".join(spans))

        text = " ".join(lines).strip()

        # Accept only probable figure captions
        if text.startswith(("Fig.", "Figure", "Table")):
            if distance < best_distance:
                caption = text
                best_distance = distance

    return caption


def extract_images(
    pdf_path: str, output_dir: str = "output/images"
) -> List[Dict[str, Any]]:
    """Extract embedded images from a PDF document and save them to output_dir.

    Args:
        pdf_path: Path to the PDF document.
        output_dir: Directory where extracted images will be stored.

    Returns:
        List of dictionaries containing extracted image metadata matching
        ExtractedImage schema requirements.
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extracted_images = []
    seen_hashes = set()

    pdf = fitz.open(pdf_file)

    try:
        for page_number in range(len(pdf)):
            page = pdf.load_page(page_number)
            images = page.get_images(full=True)

            for image_index, image in enumerate(images, start=1):
                xref = image[0]

                try:
                    base_image = pdf.extract_image(xref)
                except Exception:
                    continue

                image_bytes = base_image["image"]
                image_extension = base_image["ext"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # SHA256 hash for duplicate detection
                image_hash = hashlib.sha256(image_bytes).hexdigest()

                if image_hash in seen_hashes:
                    continue

                seen_hashes.add(image_hash)

                # Locate image on page for caption matching
                rects = page.get_image_rects(xref)
                image_rect = rects[0] if rects else None
                caption = find_caption(page, image_rect) if image_rect else None

                # Write file to disk
                filename = (
                    f"page_{page_number + 1}_image_{image_index}.{image_extension}"
                )
                file_path = output_path / filename

                with open(file_path, "wb") as file:
                    file.write(image_bytes)

                extracted_images.append(
                    {
                        "page_number": page_number + 1,
                        "image_path": str(file_path),
                        "dimensions": (width, height),
                        "caption": caption,
                        "image_bytes": image_bytes,
                    }
                )

    finally:
        pdf.close()

    return extracted_images


def extract_images_from_pdf(
    pdf_path: str, output_dir: str = "output/images"
) -> List[Dict[str, Any]]:
    """Standardized image extraction interface wrapper."""
    return extract_images(pdf_path, output_dir=output_dir)
