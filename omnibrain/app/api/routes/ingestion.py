import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

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
        file.content_type not in {
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

    STATIC_PDF_DIR.mkdir(parents=True, exist_ok=True)

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

        # Permanently store PDF for static serving
        # and PDF page rendering.
        with open(permanent_pdf_path, "wb") as pdf_file:
            pdf_file.write(file_bytes)

        # Temporary file for the ingestion pipeline.
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
        job_store[upload_id]["message"] = (
            f"Processing failed: {str(e)}"
        )
        job_store[upload_id]["warnings"] = [str(e)]

        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}",
        )

    finally:
        # Only delete the temporary processing file.
        # The permanent PDF in static/pdfs is retained.
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