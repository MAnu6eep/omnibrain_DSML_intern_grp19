from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os
import uuid
from omnibrain.app.services.ingestion.ingestion_service import IngestionService

router = APIRouter(prefix="/api/v1/ingestion", tags=["Ingestion"])
job_store = {}

async def parse_document(file_path: str):
    """
    TODO: Replace with Charan's parser
    """
    return []


async def process_images(file_path: str):
    """
    TODO: Replace with Om's vision pipeline
    """
    return []


async def store_vectors(chunks):
    """
    TODO: Replace with Meerja's vector DB service
    """
    return {
        "status": "stored",
        "chunks": len(chunks)
    }


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Malformed or unsupported file. Please upload a valid PDF."
        )
    upload_id = str(uuid.uuid4())

    job_store[upload_id] = {
        "status": "processing",
        "pages": 0,
        "text_chunks": 0,
        "images": 0,
    }

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        # -------- Pipeline --------

        service = IngestionService()

        result = service.process_pdf(
            temp_path,
            source_filename=file.filename
        )

        job_store[upload_id] = {
             "status": result.status,
             "pages": result.pages_parsed,
             "text_chunks": result.text_chunks,
             "images": result.images_extracted,
        }

        return {
            "upload_id": upload_id,
            "status": result.status,
            "pages": result.pages_parsed,
            "text_chunks": result.text_chunks,
            "images": result.images_extracted,
        }
        
    except Exception as e:

        job_store[upload_id]["status"] = "failed"

        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
@router.get("/upload/{upload_id}")
async def get_upload_status(upload_id: str):

    if upload_id not in job_store:
        raise HTTPException(
            status_code=404,
            detail="Upload ID not found."
        )

    return {
        "upload_id": upload_id,
        "status": job_store[upload_id]["status"],
        "pages": job_store[upload_id]["pages"],
        "text_chunks": job_store[upload_id]["text_chunks"],
        "images": job_store[upload_id]["images"],
    }