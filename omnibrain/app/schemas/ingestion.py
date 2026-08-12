import uuid
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ============================================================
# 1. PDF Text Page Structure
# ============================================================
class ExtractedTextPage(BaseModel):
    page_number: int = Field(
        ...,
        description="The 1-indexed page number of the PDF",
    )
    text_content: str = Field(
        ...,
        description="Raw text extracted from this page",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Headers, footers, or layout details",
    )


# ============================================================
# 2. Om & Meerja Vision Engine Output Structure (Combined)
# ============================================================
class ExtractedImage(BaseModel):
    image_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the extracted image",
    )
    document_id: str = Field(
        default="",
        description="Unique identifier of the source document",
    )
    page_number: int = Field(
        ...,
        description="The 1-indexed page number where the image was extracted",
    )
    image_path: str = Field(
        ...,
        description="Local path where the extracted image is temporarily stored",
    )
    dimensions: Tuple[int, int] = Field(
        ...,
        description="(width, height) of the image",
    )
    bbox: Optional[List[float]] = Field(
        default=None,
        description="Bounding box coordinates [x, y, width, height]",
    )
    source: str = Field(
        default="",
        description="Original source file name",
    )
    source_path: str = Field(
        default="",
        description="Original source file path",
    )
    caption: Optional[str] = Field(
        default=None,
        description="Extracted figure caption or description",
    )
    image_bytes: Optional[bytes] = Field(
        default=None,
        description="Raw binary image data",
    )
    modality: str = Field(
        default="image",
        description="Content modality",
    )


# ============================================================
# 3. Structural Text Chunk
# ============================================================
class TextChunk(BaseModel):
    chunk_id: str = Field(
        ...,
        description="Unique hash or ID for this specific chunk",
    )
    page_number: int = Field(
        ...,
        description="The page number where this chunk resides",
    )
    text: str = Field(
        ...,
        description="The chunked text snippet",
    )
    source: str = Field(
        default="",
        description="Original source file name",
    )
    source_path: str = Field(
        default="",
        description="Original source file path",
    )
    modality: str = Field(
        default="text",
        description="Content modality",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source PDF name, overlap markers, etc.",
    )


# ============================================================
# 4. API Response Contracts
# ============================================================
class IngestionResponse(BaseModel):
    task_id: str
    status: str = "processing"
    pages_parsed: int
    text_chunks: int
    images_extracted: int
    message: str
    warnings: List[str] = Field(
        default_factory=list,
    )


class UploadResponse(BaseModel):
    upload_id: str
    status: str = "processing"
    pages: int = 0
    text_chunks: int = 0
    images: int = 0
    message: str = ""
    warnings: List[str] = Field(
        default_factory=list,
    )