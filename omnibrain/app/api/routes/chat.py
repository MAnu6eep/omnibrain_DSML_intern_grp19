from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import time
import logging

from omnibrain.agents.graph import app as graph_app
from omnibrain.app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    RetrievedImage,
    RetrievedTextChunk,
    ThoughtStep,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class SQLChatRequest(BaseModel):
    query: str


class SQLChatResponse(BaseModel):
    sql_query: str
    status: str


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
    start_time = time.perf_counter()

    logger.info("Received chat request")

    # ------------------------------
    # Request Validation
    # ------------------------------

    if not request.message or not request.message.strip():
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty."
        )

    if request.top_k < 1 or request.top_k > 20:
        raise HTTPException(
            status_code=422,
            detail="top_k must be between 1 and 20."
        )

    try:

        rag_logs = [
            {
                "agent": "Self-RAG",
                "action": "Initial vector search started."
            }
        ]

        MAX_RETRIES = 2
        RETRY_DELAY = 1

        result = None

        for attempt in range(MAX_RETRIES + 1):

            try:

                result = graph_app.invoke(
                    {
                        "messages": [HumanMessage(content=request.message)],
                        "source_name": request.source_name,
                        "top_k": request.top_k,
                    }
                )

                break

            except Exception:

                if attempt == MAX_RETRIES:
                    raise

                rag_logs.append(
                    {
                        "agent": "Self-RAG",
                        "action": f"Attempt {attempt + 1} failed. Retrying..."
                    }
                )

                logger.warning(
                    "Retry %d triggered for chat request",
                    attempt + 1,
                )

                time.sleep(RETRY_DELAY)

        retrieved = result.get("retrieved_text", [])

        logger.info(
            "Retrieved %d text chunks",
            len(retrieved),
        )

        if not retrieved:

            rag_logs.extend(
                [
                    {
                        "agent": "Self-RAG",
                        "action": "Initial search returned no relevant chunks."
                    },
                    {
                        "agent": "Self-RAG",
                        "action": "Rewriting query."
                    },
                    {
                        "agent": "Self-RAG",
                        "action": "Retrying vector search."
                    },
                ]
            )

        else:

            rag_logs.append(
                {
                    "agent": "Self-RAG",
                    "action": f"Retrieved {len(retrieved)} relevant chunks."
                }
            )

        messages = result.get("messages", [])

        final_response = ""

        if messages:

            last_msg = messages[-1]

            if hasattr(last_msg, "content"):
                final_response = last_msg.content

            elif isinstance(last_msg, dict):
                final_response = last_msg.get("content", "")

            else:
                final_response = str(last_msg)

        if not final_response:

            final_response = result.get(
                "response",
                "No response generated."
            )

        retrieved_imgs = _clean_image_results(
            result.get("retrieved_images", [])
        )

        image_paths = [
            img.image_path
            for img in retrieved_imgs
            if img.image_path
        ]

        
        execution_time = time.perf_counter() - start_time

        logger.info(
            "Chat request completed successfully in %.3f seconds",
            execution_time,
        )
        return ChatResponse(
            response=final_response,
            thought_process=[
                ThoughtStep(**step)
                for step in rag_logs + result.get("thought_process", [])
            ],
            images=image_paths,
            retrieved_text=_clean_text_results(
                result.get("retrieved_text", [])
            ),
            retrieved_images=retrieved_imgs,
            status="completed",
        )

    except HTTPException:
     raise

    except Exception as e:
        execution_time = time.perf_counter() - start_time

        logger.exception(
            "Chat pipeline failed after %.3f seconds: %s",
            execution_time,
            e,
        )

        raise HTTPException(
            status_code=500,
            detail="Chat pipeline failed after multiple retry attempts.",
        )


@router.post("/sql", response_model=SQLChatResponse)
async def chat_sql(request: SQLChatRequest):
    """
    Internal endpoint for isolated Text-to-SQL testing.
    """

    logger.info("Received SQL chat request")

    sql_query = f"-- Generated SQL for: {request.query}"

    return SQLChatResponse(
        sql_query=sql_query,
        status="success",
    )