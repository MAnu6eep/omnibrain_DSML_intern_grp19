from pathlib import Path
from typing import Dict, List
import hashlib

import fitz


def find_caption(page, image_rect, max_distance=50):
    """
    Finds the nearest text block immediately below an image.
    """

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

        # Accept only probable figure captions
        if text.startswith(("Fig.", "Figure", "Table")):
            if distance < best_distance:
                caption = text
                best_distance = distance

    return caption


def extract_images(pdf_path: str, output_dir: str) -> List[Dict]:
    """
    Extract embedded images from a PDF document.

    Args:
        pdf_path: Path to the PDF document.
        output_dir: Directory where extracted images will be stored.

    Returns:
        List of dictionaries containing extracted image metadata.
    """

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extracted_images = []

    # Used to skip duplicate images
    seen_hashes = set()

    pdf = fitz.open(pdf_file)

    try:
        for page_number in range(len(pdf)):

            page = pdf.load_page(page_number)
            images = page.get_images(full=True)

            for image_index, image in enumerate(images, start=1):

                xref = image[0]

                # Skip corrupted image streams
                try:
                    base_image = pdf.extract_image(xref)
                except Exception:
                    continue

                image_bytes = base_image["image"]
                image_extension = base_image["ext"]

                # SHA256 hash for duplicate detection
                image_hash = hashlib.sha256(image_bytes).hexdigest()

                if image_hash in seen_hashes:
                    continue

                seen_hashes.add(image_hash)

                # Locate image on page
                rects = page.get_image_rects(xref)

                image_rect = rects[0] if rects else None

                caption = None

                if image_rect:
                    caption = find_caption(page, image_rect)

                filename = (
                    f"page_{page_number + 1}_"
                    f"image_{image_index}.{image_extension}"
                )

                file_path = output_path / filename

                with open(file_path, "wb") as file:
                    file.write(image_bytes)

                extracted_images.append(
                    {
                        "page": page_number + 1,
                        "image_index": image_index,
                        "xref": xref,
                        "extension": image_extension,
                        "sha256": image_hash,
                        "caption": caption,
                        "file_path": str(file_path),
                    }
                )

    finally:
        pdf.close()

    return extracted_images