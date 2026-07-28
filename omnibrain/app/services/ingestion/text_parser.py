from pathlib import Path
from typing import Any, Dict, List

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_and_chunk(
    pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """Extract page text from a PDF and return chunk dictionaries with metadata."""
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_file)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: List[Dict[str, Any]] = []

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = (page.get_text("text") or "").strip()

            if not page_text:
                continue

            page_chunks = splitter.split_text(page_text)

            for chunk_index, chunk_text in enumerate(page_chunks):
                chunk_id = f"{pdf_file.stem}_p{page_num + 1}_c{chunk_index}"

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk_text.strip(),
                        "page_number": page_num + 1,
                        "source": pdf_file.name,
                        "source_path": str(pdf_file),
                        "metadata": {
                            "source": pdf_file.name,
                            "source_path": str(pdf_file),
                            "page_number": page_num + 1,
                            "chunk_index": chunk_index,
                            "chunk_id": chunk_id,
                            "modality": "text",
                            "table_extraction": "partial",
                            "tables_extracted": False,
                        },
                    }
                )
    finally:
        doc.close()

    return chunks


if __name__ == "__main__":
    pdf_path = "data/Attention_is_all_you_need.pdf"
    if Path(pdf_path).exists():
        doc = fitz.open(pdf_path)
        print("Number of pages:", len(doc))
