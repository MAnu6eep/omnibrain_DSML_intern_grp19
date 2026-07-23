from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class ThoughtStep(BaseModel):
    agent: str
    action: str


class ChatResponse(BaseModel):
    response: str
    thought_process: List[ThoughtStep]
    images: List[str] = Field(default_factory=list)
    retrieved_text: List[RetrievedTextChunk] = Field(default_factory=list)
    retrieved_images: List[RetrievedImage] = Field(default_factory=list)
    status: str = "completed"
    timestamp: Optional[str] = None