import json
import os
from typing import Any, Dict

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from omnibrain.agents.prompts.prompts import SUPERVISOR_PROMPT
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


def _heuristic_route(query: str) -> tuple[str, str]:
    query_lower = query.lower()

    # Vision related queries
    if any(
        term in query_lower
        for term in ("image", "figure", "diagram", "chart", "table", "visual")
    ):
        return "vision_agent", "Heuristic routing selected the vision agent."

    # Web related queries
    if any(
        term in query_lower
        for term in ("web", "internet", "search", "latest", "current")
    ):
        return "web_agent", "Heuristic routing selected the web agent."

    # Document related queries
    document_keywords = [
        "pdf",
        "document",
        "file",
        "page",
        "chapter",
        "chunk",
        "upload",
        "uploaded",
        "summary",
        "summarize",
    ]

    if any(keyword in query_lower for keyword in document_keywords):
        return "text_agent", "Document query detected."

    # Everything else goes directly to Gemini
    return "direct_llm", "General query detected."

def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor Router Node that parses the last user message and determines
    which worker agent should handle the query next.
    """
    messages = state.get("messages", [])
    query = messages[-1].content if messages else ""

    llm = _get_llm()

    if llm is None:
        next_node, thought = _heuristic_route(query)
        return {
            "next_node": next_node,
            "thought_process": [
                {"agent": "Supervisor", "action": f"Routed to {next_node}: {thought}"}
            ],
        }

    system_msg = SystemMessage(content=SUPERVISOR_PROMPT)
    prompt_messages = [system_msg] + messages

    try:
        response = llm.invoke(prompt_messages)
    except Exception:
        next_node, thought = _heuristic_route(query)
        return {
            "next_node": next_node,
            "thought_process": [
                {"agent": "Supervisor", "action": f"Routed to {next_node}: {thought}"}
            ],
        }

    try:
        # Parse router JSON choice
        content = response.content.strip()

        content = content.replace("```json", "")
        content = content.replace("```", "").strip()

        decision = json.loads(content)
        next_node = decision.get("next_node", "direct_llm")
        thought = decision.get("thought", "Routing request...")
    except Exception:
        next_node, thought = _heuristic_route(query)

    new_thought = {"agent": "Supervisor", "action": f"Routed to {next_node}: {thought}"}

    return {"next_node": next_node, "thought_process": [new_thought]}
