from fastapi import APIRouter

from omnibrain.app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ThoughtStep,
)

router = APIRouter(
    prefix="/api/v1/chat",
    tags=["Chat"],
)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):

    return ChatResponse(
        response=f"Received: {request.message}",
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
    )