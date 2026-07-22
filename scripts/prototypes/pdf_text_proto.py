from pathlib import Path
from typing import Any, Dict, List

import fitz


def extract_text_and_chunk(pdf_path: str) -> List[Dict[str, Any]]:
    """Parses a PDF document using PyMuPDF, extracts text page-by-page, and breaks

    paragraphs into structured text chunks with metadata tracking.
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    chunks = []
    chunk_counter = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # Split text by paragraph double breaks
        paragraphs = text.split("\n\n")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            lines = para.split("\n")
            section_heading = "General"

            # Check if first line behaves like a header title
            if len(lines[0]) < 80:
                section_heading = lines[0]
                body_text = " ".join(lines[1:]) if len(lines) > 1 else lines[0]
            else:
                body_text = " ".join(lines)

            if not body_text.strip():
                continue

            chunk_id = f"page_{page_num + 1}_chunk_{chunk_counter}"

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": body_text,
                    "page_number": page_num + 1,
                    "metadata": {
                        "source": Path(pdf_path).name,
                        "section": section_heading,
                        "chunk_id": chunk_id,
                    },
                }
            )
            chunk_counter += 1

    doc.close()
    return chunks


if __name__ == "__main__":
    # Local standalone execution/testing
    sample_path = "data/Attention_is_all_you_need.pdf"
    if Path(sample_path).exists():
        result_chunks = extract_text_and_chunk(sample_path)
        print(
            f"Done! Extracted {len(result_chunks)} chunks from {sample_path}"
            " successfully."
        )
