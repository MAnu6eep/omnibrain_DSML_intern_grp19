from datetime import datetime, timezone

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


def build_mock_response(message: str) -> ChatResponse:
    """
    Build a temporary mock response for frontend integration.

    TODO:
    Replace this helper with the compiled LangGraph workflow
    (app.invoke()) once the backend graph is available.
    """
    return ChatResponse(
        response=f"Received: {message}",
        thought_process=[
            ThoughtStep(
                agent="Retriever",
                action="Searching relevant context",
            ),
            ThoughtStep(
                agent="LLM",
                action="Generating response",
            ),
        ],
        images=[],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Temporary chat endpoint used for frontend integration.

    TODO:
    Replace the mock response with LangGraph app.invoke()
    after the compiled workflow is available.
    """
    return build_mock_response(request.message)