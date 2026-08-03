

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
@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = graph_app.invoke(
            {
                "messages": [HumanMessage(content=request.message)],
                "source_name": request.source_name,
                "top_k": request.top_k,
            }
        )

        return ChatResponse(
            response=result.get("response", ""),
            thought_process=[
                ThoughtStep(**step)
                for step in result.get("thought_process", [])
            ],
            images=result.get("images", []),
            retrieved_text=_clean_text_results(
                result.get("retrieved_text", [])
            ),
            retrieved_images=_clean_image_results(
                result.get("retrieved_images", [])
            ),
            status="completed",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
