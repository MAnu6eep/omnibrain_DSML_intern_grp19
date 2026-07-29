import os
from typing import Any, Dict

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from omnibrain.agents.state.state import AgentState


def _get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
    )


def _clean_text_results(results: list[dict]) -> list[dict]:
    cleaned = []
    seen_chunk_ids = set()

    for result in results or []:
        if not isinstance(result, dict):
            continue

        text = (result.get("text") or "").strip()
        chunk_id = result.get("chunk_id") or ""

        if not text or not chunk_id or chunk_id in seen_chunk_ids:
            continue

        seen_chunk_ids.add(chunk_id)
        cleaned.append(result)

    return cleaned


def _clean_image_results(results: list[dict]) -> list[dict]:
    cleaned = []
    seen_paths = set()

    for result in results or []:
        if not isinstance(result, dict):
            continue

        image_path = (result.get("image_path") or "").strip()
        if not image_path or image_path in seen_paths:
            continue

        seen_paths.add(image_path)
        cleaned.append(result)

    return cleaned


def generator_node(state: AgentState) -> Dict[str, Any]:
    """
    Final generation node.
    Takes retrieved text/images from previous agents and produces
    the final AI response.
    """

    # Get the original user question
    user_query = state["messages"][-1].content

    # Retrieved context
    retrieved_text = _clean_text_results(state.get("retrieved_text", []))
    retrieved_images = _clean_image_results(state.get("retrieved_images", []))

    # -------------------------
    # Build context
    # -------------------------
    context_lines = []

    for doc in retrieved_text:
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        page = (
            doc.get("page", metadata.get("page_number", "-"))
            if isinstance(doc, dict)
            else "-"
        )
        chunk_id = (
            doc.get("chunk_id", metadata.get("chunk_id", "-"))
            if isinstance(doc, dict)
            else "-"
        )
        source = (
            doc.get("source", doc.get("document", metadata.get("source", "")))
            if isinstance(doc, dict)
            else ""
        )
        text = doc.get("text", "") if isinstance(doc, dict) else str(doc)

        src = source or "Unknown"
        context_lines.append(
            f"[Text | Source: {src} | Page: {page} | Chunk: {chunk_id}] {text}"
        )

    for img in retrieved_images:
        metadata = img.get("metadata", {}) if isinstance(img, dict) else {}
        page = (
            img.get("page_number", metadata.get("page_number", "-"))
            if isinstance(img, dict)
            else "-"
        )
        source = (
            img.get("source", metadata.get("source", ""))
            if isinstance(img, dict)
            else ""
        )
        caption = img.get("caption", "") if isinstance(img, dict) else ""
        image_path = img.get("image_path", "") if isinstance(img, dict) else ""

        src = source or "Unknown"
        cap = caption or "No caption available."
        context_lines.append(
            f"[Image | Source: {src} | Page: {page} | Path: {image_path}] {cap}"
        )

    context = "\n\n".join(context_lines).strip()

    llm = _get_llm()

    if not context:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "The vector store does not contain relevant "
                        "context for this question. Please upload "
                        "or re-upload the file(s) so we can process "
                        "and add the required context into the system."
                    )
                )
            ],
            "thought_process": [
                {
                    "agent": "Generator",
                    "action": (
                        "No relevant context found in vector store; requested "
                        "file re-upload."
                    ),
                }
            ],
        }

    # -------------------------
    # Prompt
    # -------------------------
    prompt = f"""
You are OmniBrain.

Answer ONLY using the relevant retrieved context below.

If the retrieved context does NOT contain sufficient relevant
information to answer the question, clearly state:
"The vector data does not contain sufficient relevant context to answer your question.
Please upload or re-upload the relevant file(s) so we can process
them and add the required context into the system."

Retrieved Context:

{context}

User Question:

{user_query}
"""

    # -------------------------
    # LLM Generation
    # -------------------------
    if llm is None:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "The indexed context is available, but the Gemini "
                        "API key is not configured on this machine. Configure "
                        "GEMINI_API_KEY to enable LLM synthesis."
                    )
                )
            ],
            "thought_process": [
                {
                    "agent": "Generator",
                    "action": (
                        "LLM synthesis skipped because GEMINI_API_KEY " "is missing."
                    ),
                }
            ],
        }

    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "The retrieval context was prepared, but the language "
                        f"model call failed safely: {exc}"
                    )
                )
            ],
            "thought_process": [
                {
                    "agent": "Generator",
                    "action": (
                        "LLM synthesis failed and the node returned a " "safe fallback."
                    ),
                }
            ],
        }

    if isinstance(response.content, str):
        answer = response.content

    elif isinstance(response.content, list):
        answer = ""

        for part in response.content:
            if isinstance(part, dict):
                answer += part.get("text", "")
            else:
                answer += str(part)

    else:
        answer = str(response.content)

    thought = {
        "agent": "Generator",
        "action": "Synthesized final response from retrieved context.",
    }

    return {
        "messages": [AIMessage(content=answer)],
        "thought_process": [thought],
    }
