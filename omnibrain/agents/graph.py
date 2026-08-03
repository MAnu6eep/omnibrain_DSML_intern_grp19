import sys

from langgraph.graph import END, StateGraph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from omnibrain.agents.direct_llm.direct_llm import direct_llm_node
from omnibrain.agents.generator.generator import generator_node
from omnibrain.agents.sql_agent.sql_agent import sql_agent_node
from omnibrain.agents.state.state import AgentState
from omnibrain.agents.supervisor.supervisor import supervisor_node
from omnibrain.agents.tools.web_search import execute_web_search
from omnibrain.vectorstore.retrievers.image_retriever import search_images
from omnibrain.vectorstore.retrievers.text_retriever import search_text_chunks


def _extract_query(state: AgentState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return ""
    last_msg = messages[-1]
    if hasattr(last_msg, "content"):
        return (
            last_msg.content
            if isinstance(last_msg.content, str)
            else str(last_msg.content)
        )
    elif isinstance(last_msg, dict):
        return last_msg.get("content", "")
    return str(last_msg)


def text_agent_node(state: AgentState):
    query = _extract_query(state)

    results = search_text_chunks(query)

    print(f"QUERY: {query} -> RETRIEVED {len(results)} CHUNKS FROM QDRANT")

    thought = {
        "agent": "Text Agent",
        "action": (
            f"Retrieved {len(results)} text chunks from Qdrant."
            if results
            else "No text chunks were retrieved from Qdrant."
        ),
    }

    return {
        "retrieved_text": results,
        "thought_process": [thought],
    }


def vision_agent_node(state: AgentState):
    query = _extract_query(state)
    results = search_images(query)
    thought = {
        "agent": "Vision Agent",
        "action": (
            f"Retrieved {len(results)} images using CLIP embeddings."
            if results
            else "No embedded images were retrieved using CLIP embeddings."
        ),
    }
    return {
        "retrieved_images": results,
        "next_node": "FINISH",
        "thought_process": [thought],
    }


def web_agent_node(state: AgentState):
    query = _extract_query(state)
    results = execute_web_search(query)
    thought = {
        "agent": "Web Agent",
        "action": (
            "Executed web search fallback; retrieved" f" {len(results)} web results."
            if results
            else "Executed web search fallback but found no usable results."
        ),
    }
    return {
        "retrieved_text": results,
        "next_node": "FINISH",
        "thought_process": [thought],
    }


def route_next(state: AgentState) -> str:
    return state.get("next_node", "FINISH")


workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("text_agent", text_agent_node)
workflow.add_node("sql_agent", sql_agent_node)
workflow.add_node("vision_agent", vision_agent_node)
workflow.add_node("web_agent", web_agent_node)
workflow.add_node("generator", generator_node)
workflow.add_node("direct_llm", direct_llm_node)
workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(
    "supervisor",
    route_next,
    {
        "text_agent": "text_agent",
        "sql_agent": "sql_agent",
        "vision_agent": "vision_agent",
        "web_agent": "web_agent",
        "direct_llm": "direct_llm",
        "FINISH": END,
    },
)

workflow.add_edge("text_agent", "generator")
workflow.add_edge("sql_agent", "generator")
workflow.add_edge("vision_agent", "generator")
workflow.add_edge("web_agent", "generator")

workflow.add_edge("generator", END)
workflow.add_edge("direct_llm", END)
app = workflow.compile()
