import base64
import os
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage

from omnibrain.agents.state.state import AgentState
from omnibrain.agents.prompts.prompts import SYNTHESIZER_PROMPT


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
    Takes retrieved text/images from previous agents and produces the final AI response.
    """
    messages = state.get("messages", [])
    user_query = ""
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "content"):
            user_query = (
                last_msg.content
                if isinstance(last_msg.content, str)
                else str(last_msg.content)
            )
        elif isinstance(last_msg, dict):
            user_query = last_msg.get("content", "")
        else:
            user_query = str(last_msg)

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
        chart_type = img.get("chart_type", "")
        title = img.get("title", "")
        summary = img.get("summary", "")

        values = img.get("values", [])

        value_lines = []

        for item in values:
            value_lines.append(
                f"{item.get('label', '')} = {item.get('value', '')}"
            )

        context_lines.append(
            f"""
        [Vision | Source: {src} | Page: {page}]

        Caption:
        {cap}

        Chart Type:
        {chart_type}

        Title:
        {title}

        Extracted Values:
        {chr(10).join(value_lines)}

        Summary:
        {summary}
        """.strip()
        )

    context = "\n\n".join(context_lines).strip()

    prompt_text = f"""
    {SYNTHESIZER_PROMPT}

    Retrieved Context:

    {context}

    User Question:

    {user_query}
    """

    message_content = [{"type": "text", "text": prompt_text}]

    for img in retrieved_images:
        img_path = img.get("image_path", "") if isinstance(img, dict) else str(img)
        if img_path and os.path.exists(img_path):
            try:
                ext = Path(img_path).suffix.lstrip(".").lower() or "jpeg"
                if ext == "jpg":
                    ext = "jpeg"
                with open(img_path, "rb") as image_file:
                    b64_encoded = base64.b64encode(image_file.read()).decode("utf-8")
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{ext};base64,{b64_encoded}"},
                    }
                )
            except Exception:
                pass

    prompt_payload = (
        [HumanMessage(content=message_content)]
        if len(message_content) > 1
        else prompt_text
    )

    # -------------------------
    # LLM Generation via Multi-Provider Manager
    # -------------------------
    from omnibrain.agents.llm import get_llm_response

    try:
        answer, provider_used = get_llm_response(prompt_payload)
    except Exception:
        answer = f"Retrieved Context Summary:\n{context}"
        provider_used = "Local Context Fallback"

    thought = {
        "agent": "Generator",
        "action": f"Synthesized final response using {provider_used}.",
    }

    return {
        "messages": [AIMessage(content=answer)],
        "thought_process": [thought],
    }
