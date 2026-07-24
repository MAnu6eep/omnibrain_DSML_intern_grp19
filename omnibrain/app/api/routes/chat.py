from fastapi import APIRouter
from datetime import datetime, timezone

from omnibrain.app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ThoughtStep,
)

router = APIRouter(
    prefix="/api/v1/chat",
    tags=["Chat"],
)

def build_mock_response(message: str) -> ChatResponse:
    return ChatResponse(
        response=f"Received: {message}",
        thought_process=[
            ThoughtStep(
                agent="Retriever",
                action="Searching relevant context"
            ),
            ThoughtStep(
                agent="LLM",
                action="Generating response"
            ),
        ],
        images=[],
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # TODO: Replace build_mock_response() with LangGraph app.invoke()
    return build_mock_response(request.message)