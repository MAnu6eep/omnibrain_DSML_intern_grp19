from pathlib import Path

import fitz
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

router = APIRouter()

STATIC_PDF_DIR = Path("static/pdfs")


@router.get("/page/{pdf_id}/{page_num}")
async def get_pdf_page(pdf_id: str, page_num: int):
    """
    Render a single PDF page and return it as a PNG image.
    """

    pdf_path = STATIC_PDF_DIR / f"{pdf_id}.pdf"

    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail="PDF not found.",
        )

    if page_num < 1:
        raise HTTPException(
            status_code=422,
            detail="Page number must be greater than or equal to 1.",
        )

    try:
        document = fitz.open(pdf_path)

        # API uses 1-based page numbering.
        page_index = page_num - 1

        if page_index >= len(document):
            document.close()
            raise HTTPException(
                status_code=404,
                detail="PDF page not found.",
            )

        page = document.load_page(page_index)

        # Render page at a reasonable resolution.
        matrix = fitz.Matrix(2, 2)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)

        png_bytes = pixmap.tobytes("png")

        document.close()

        return Response(
            content=png_bytes,
            media_type="image/png",
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to render PDF page: {str(e)}",
        )