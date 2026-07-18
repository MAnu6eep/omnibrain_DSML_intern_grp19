from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os

router = APIRouter(prefix="/api/v1/ingestion", tags=["Ingestion"])


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
            detail="Only PDF files are allowed."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        # -------- Pipeline --------

        text_chunks = await parse_document(temp_path)

        image_chunks = await process_images(temp_path)

        all_chunks = text_chunks + image_chunks

        result = await store_vectors(all_chunks)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks": len(all_chunks),
            "vector_store": result
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)