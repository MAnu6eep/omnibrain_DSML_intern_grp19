from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# 1. Charan's (PDF Eng) output structure
class ExtractedTextPage(BaseModel):
    page_number: int = Field(..., description="The 1-indexed page number of the PDF")
    text_content: str = Field(..., description="Raw text extracted from this page")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Headers, footers, or layout details")

# 2. Om's (Vision Eng) output structure
class ExtractedImage(BaseModel):
    page_number: int
    image_path: str = Field(..., description="Local path where the extracted image is temporarily stored")
    dimensions: tuple[int, int] = Field(..., description="(width, height) of the image")

# 3. Structural chunk output after text splitting
class TextChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique hash or ID for this specific chunk")
    page_number: int
    text: str = Field(..., description="The chunked text snippet")
    metadata: Dict[str, Any] = Field(..., description="Source PDF name, overlap markers, etc.")

# 4. Manav's (Backend Eng) API Response contract
class IngestionResponse(BaseModel):
    task_id: str
    status: str = Field("processing", description="Current pipeline state: processing, success, failed")
    pages_parsed: int
    images_extracted: int
    message: str