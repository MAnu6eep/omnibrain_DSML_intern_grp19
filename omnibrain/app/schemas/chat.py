from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    source_name: Optional[str] = None
    top_k: int = 5


class ThoughtStep(BaseModel):
    agent: str
    action: str


class RetrievedTextChunk(BaseModel):
    chunk_id: str
    document: str = ""
    page: int | str = 0
    text: str = ""
    score: float = 0.0
    source: str = ""
    modality: str = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedImage(BaseModel):
    image_path: str
    page_number: int = 0
    caption: Optional[str] = None
    score: float = 0.0
    source: str = ""
    modality: str = "image"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    response: str
    thought_process: List[ThoughtStep]
    images: List[str] = Field(default_factory=list)
    retrieved_text: List[RetrievedTextChunk] = Field(default_factory=list)
    retrieved_images: List[RetrievedImage] = Field(default_factory=list)
    status: str = "completed"
