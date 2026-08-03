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

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    try:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
            max_retries=0,
        )
    except Exception:
        return None


def _heuristic_route(query: str) -> tuple[str, str]:
    query_lower = query.lower()

    # 1. Vision related queries (images, figures, diagrams, charts)
    if any(
        term in query_lower
        for term in ("image", "figure", "diagram", "chart", "bar chart", "visual")
    ):
        return "vision_agent", "Visual figure/image query detected."

    # 2. SQL related queries
    if any(
        term in query_lower
        for term in (
            "sql",
            "sql query",
            "table query",
            "select ",
            "from metrics",
            "sqlite",
            "relational database",
        )
    ):
        return "sql_agent", "Structured SQL database query detected."

    # 3. Document / Vector DB related queries
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
        "vector database",
        "vector db",
        "qdrant",
        "vector",
    ]

    if any(keyword in query_lower for keyword in document_keywords):
        return "text_agent", "Document vector query detected."

    # 4. Web related queries
    if any(
        term in query_lower
        for term in (
            "web",
            "internet",
            "search the web",
            "web search",
            "latest news",
            "current news",
            "online search",
        )
    ):
        return "web_agent", "Public web search query detected."

    # Everything else goes directly to Gemini
    return "direct_llm", "General query detected."


def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor Router Node that parses the last user message and determines
    which worker agent should handle the query next.
    """
    messages = state.get("messages", [])
    query = ""
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "content"):
            query = (
                last_msg.content
                if isinstance(last_msg.content, str)
                else str(last_msg.content)
            )
        elif isinstance(last_msg, dict):
            query = last_msg.get("content", "")
        else:
            query = str(last_msg)

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
