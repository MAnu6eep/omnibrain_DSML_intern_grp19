import io
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from omnibrain.app.schemas.ingestion import UploadResponse
from omnibrain.app.services.ingestion.ingestion_service import IngestionService

STATIC_PDF_DIR = Path("static/pdfs")
STATIC_PDF_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter()
job_store = {}


def _upload_response(upload_id: str, job: dict) -> UploadResponse:
    return UploadResponse(
        upload_id=upload_id,
        status=job.get("status", "processing"),
        pages=job.get("pages", 0),
        text_chunks=job.get("text_chunks", 0),
        images=job.get("images", 0),
        message=job.get("message", ""),
        warnings=job.get("warnings", []),
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    filename = file.filename or "uploaded.pdf"
    file_suffix = Path(filename).suffix.lower()

    if (
        file.content_type
        not in {
            "application/pdf",
            "application/octet-stream",
        }
        and file_suffix != ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Malformed or unsupported file. Please upload a valid PDF.",
        )

    upload_id = str(uuid.uuid4())
    permanent_pdf_path = STATIC_PDF_DIR / f"{upload_id}.pdf"
    temp_path = None

    job_store[upload_id] = {
        "status": "processing",
        "pages": 0,
        "text_chunks": 0,
        "images": 0,
        "message": "Upload accepted. Processing started.",
        "warnings": [],
    }

    try:
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded PDF is empty.",
            )

        # Permanently store PDF for static serving and citation rendering
        with open(permanent_pdf_path, "wb") as pdf_file:
            pdf_file.write(file_bytes)

        # Temporary file for the ingestion pipeline
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp:
            temp.write(file_bytes)
            temp_path = temp.name

        service = IngestionService()
        result = service.process_pdf(
            temp_path,
            source_filename=filename,
        )

        job_store[upload_id] = {
            "status": result.status,
            "pages": result.pages_parsed,
            "text_chunks": result.text_chunks,
            "images": result.images_extracted,
            "message": result.message,
            "warnings": result.warnings,
        }

        return _upload_response(
            upload_id,
            job_store[upload_id],
        )

    except HTTPException:
        raise

    except Exception as e:
        job_store[upload_id]["status"] = "failed"
        job_store[upload_id]["message"] = f"Processing failed: {str(e)}"
        job_store[upload_id]["warnings"] = [str(e)]
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}",
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.get(
    "/upload/{upload_id}",
    response_model=UploadResponse,
)
async def get_upload_status(upload_id: str):
    if upload_id not in job_store:
        raise HTTPException(
            status_code=404,
            detail="Upload ID not found.",
        )

    return _upload_response(
        upload_id,
        job_store[upload_id],
    )


@router.get("/pdf/page/{pdf_id}/{page_num}")
async def get_pdf_page_render(pdf_id: str, page_num: int):
    """
    Renders and returns a single PDF page as a PNG image [Manav Day 4 Scope].
    """
    pdf_path = STATIC_PDF_DIR / f"{pdf_id}.pdf"
    if not pdf_path.exists():
        data_pdf = Path("data") / pdf_id
        if data_pdf.exists():
            pdf_path = data_pdf
        else:
            raise HTTPException(
                status_code=404,
                detail=f"PDF '{pdf_id}' not found.",
            )

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            raise HTTPException(
                status_code=400,
                detail=f"Page {page_num} out of bounds for PDF (Total pages: {len(doc)}).",
            )

        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        doc.close()

        return Response(content=img_bytes, media_type="image/png")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to render PDF page: {str(exc)}",
        )


@router.post("/purge")
async def purge_vector_store():
    """
    User-Authorized Purge: Permanently deletes all indexed text and image vectors from Qdrant.
    """
    try:
        from omnibrain.vectorstore.collections import purge_collections
        from omnibrain.vectorstore.qdrant_client import QdrantClientWrapper

        client = QdrantClientWrapper().client()
        purge_collections(client)

        return {
            "status": "success",
            "message": "Vector database collections purged and re-initialized successfully.",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to purge vector database: {str(exc)}",
        )


@router.get("/documents")
async def get_indexed_documents_endpoint():
    """
    Returns a list of all unique PDF document names currently indexed in Qdrant.
    """
    try:
        from omnibrain.vectorstore.retrievers.text_retriever import (
            get_indexed_documents,
        )

        docs = get_indexed_documents()
        return {"documents": docs}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve indexed documents: {str(exc)}",
        )
