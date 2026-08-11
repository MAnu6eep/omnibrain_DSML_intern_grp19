from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# 1. Charan's (PDF Eng) output structure
class ExtractedTextPage(BaseModel):
    page_number: int = Field(..., description="The 1-indexed page number of the PDF")
    text_content: str = Field(..., description="Raw text extracted from this page")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Headers, footers, or layout details"
    )


# 2. Om's (Vision Eng) output structure
class ExtractedImage(BaseModel):
    page_number: int = Field(
        ...,
        description="The 1-indexed page number where the image was extracted"
    )
    image_path: str = Field(
        ...,
        description="Local path where the extracted image is temporarily stored"
    )
    dimensions: Tuple[int, int] = Field(
        ...,
        description="(width, height) of the image"
    )
    bbox: Optional[List[float]] = Field(
        default=None,
        description="Bounding box coordinates [x, y, width, height] of the extracted image"
    )
    caption: Optional[str] = Field(
        default=None,
        description="Extracted figure caption or description"
    )
    image_bytes: Optional[bytes] = Field(
        default=None,
        description="Raw binary image data"
    )


# 3. Structural chunk output after text splitting
class TextChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique hash or ID for this specific chunk")
    page_number: int = Field(
        ..., description="The page number where this chunk resides"
    )
    text: str = Field(..., description="The chunked text snippet")
    source: str = Field(default="", description="Original source file name")
    source_path: str = Field(default="", description="Original source file path")
    modality: str = Field(default="text", description="Content modality")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Source PDF name, overlap markers, etc."
    )


# 4. Manav's (Backend Eng) API Response contract
class IngestionResponse(BaseModel):
    task_id: str
    status: str = "processing"
    pages_parsed: int
    text_chunks: int
    images_extracted: int
    message: str
    warnings: List[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    upload_id: str
    status: str = "processing"
    pages: int = 0
    text_chunks: int = 0
    images: int = 0
    message: str = ""
    warnings: List[str] = Field(default_factory=list)
