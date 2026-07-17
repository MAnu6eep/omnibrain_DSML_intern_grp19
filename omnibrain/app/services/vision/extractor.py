from pathlib import Path
from typing import Dict, List

import fitz


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
                        "file_path": str(file_path),
                    }
                )

    finally:
        pdf.close()

    return extracted_images