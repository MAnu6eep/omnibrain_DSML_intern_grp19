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

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0,
    )


def _heuristic_route(query: str) -> tuple[str, str]:
    query_lower = query.lower()

    if any(
        term in query_lower
        for term in ("image", "figure", "diagram", "chart", "table", "visual")
    ):
        return "vision_agent", "Heuristic routing selected the vision agent."

    if any(
        term in query_lower
        for term in ("web", "internet", "search", "latest", "current")
    ):
        return "web_agent", "Heuristic routing selected the web agent."

    return "text_agent", "Heuristic routing selected the text agent."


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
        next_node = decision.get("next_node", "text_agent")
        thought = decision.get("thought", "Routing request...")
    except Exception:
        next_node, thought = _heuristic_route(query)

    new_thought = {"agent": "Supervisor", "action": f"Routed to {next_node}: {thought}"}

    return {"next_node": next_node, "thought_process": [new_thought]}
