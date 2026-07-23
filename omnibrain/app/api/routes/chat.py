
from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from omnibrain.agents.graph import app as graph_app
from omnibrain.app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    RetrievedImage,
    RetrievedTextChunk,
    ThoughtStep,
)

router = APIRouter()


def _clean_text_results(results):
    cleaned = []
    seen = set()

    for item in results or []:
        if not isinstance(item, dict):
            continue

        text = (item.get("text") or "").strip()
        chunk_id = item.get("chunk_id") or ""

        if not text or not chunk_id or chunk_id in seen:
            continue

        seen.add(chunk_id)
        cleaned.append(
            RetrievedTextChunk(
                chunk_id=chunk_id,
                document=item.get("document", ""),
                page=item.get("page", item.get("page_number", 0)),
                text=text,
                score=float(item.get("score", 0.0) or 0.0),
                source=item.get("source", item.get("document", "")),
                modality=item.get("modality", "text"),
                metadata=item.get("metadata", {}),
            )
        )

    return cleaned


def _clean_image_results(results):
    cleaned = []
    seen = set()

    for item in results or []:
        if not isinstance(item, dict):
            continue

        image_path = (item.get("image_path") or "").strip()
        if not image_path or image_path in seen:
            continue

        seen.add(image_path)
        cleaned.append(
            RetrievedImage(
                image_path=image_path,
                page_number=int(item.get("page_number", 0) or 0),
                caption=item.get("caption"),
                score=float(item.get("score", 0.0) or 0.0),
                source=item.get("source", ""),
                modality=item.get("modality", "image"),
                metadata=item.get("metadata", {}),
            )
        )

    return cleaned


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
@router.post("", response_model=ChatResponse)
@router.post("/", response_model=ChatResponse, include_in_schema=False)
async def chat_endpoint(request: ChatRequest):

    try:

        config = {
            "configurable": {"thread_id": request.conversation_id or "default_session"}
        }

        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "retrieved_text": [],
            "retrieved_images": [],
            "thought_process": [],
            "next_node": "",
        }

        final_state = graph_app.invoke(initial_state, config=config)

        messages = final_state.get("messages", [])
        retrieved_text = _clean_text_results(final_state.get("retrieved_text", []))
        retrieved_images = _clean_image_results(final_state.get("retrieved_images", []))

        final_response = messages[-1].content if messages else "Execution completed."

        thoughts = [
            ThoughtStep(
                agent=t.get("agent", "Agent"),
                action=t.get("action", ""),
            )
            for t in final_state.get("thought_process", [])
        ]

        images = [image.image_path for image in retrieved_images]

        return ChatResponse(
            response=final_response,
            thought_process=thoughts,
            images=images,
            retrieved_text=retrieved_text,
            retrieved_images=retrieved_images,
            status="completed",
        )

    except Exception as e:
        traceback.print_exc()
        # Prints the full stack trace to the FastAPI logs
        raise HTTPException(status_code=500, detail=str(e))
