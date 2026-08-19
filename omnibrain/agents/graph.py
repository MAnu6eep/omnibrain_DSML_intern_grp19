import sys

from langgraph.graph import END, StateGraph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


from omnibrain.agents.direct_llm.direct_llm import direct_llm_node
from omnibrain.agents.evaluators.grader import grader_node
from omnibrain.agents.evaluators.retrieval_evaluator import retrieval_evaluator_node
from omnibrain.agents.generator.generator import generator_node
from omnibrain.agents.nodes.query_rewriter_node import query_rewriter_node
from omnibrain.agents.sql_agent.sql_agent import SQLAgent
from omnibrain.agents.state.state import AgentState
from omnibrain.agents.supervisor.supervisor import supervisor_node
from omnibrain.agents.tools.web_search import execute_web_search
from omnibrain.vectorstore.retrievers.image_retriever import search_images
from omnibrain.vectorstore.retrievers.text_retriever import search_text_chunks

sql_agent = SQLAgent()


# -----------------------------
# Extract user query
# -----------------------------


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


# -----------------------------
# Text Retrieval Agent
# -----------------------------


def text_agent_node(state: AgentState):

    query = _extract_query(state)
    source_name = state.get("source_name")
    document_id = state.get("document_id")

    results = search_text_chunks(
        query,
        source_name=source_name,
        document_id=document_id,
    )

    print(f"QUERY: {query} -> " f"RETRIEVED {len(results)} CHUNKS FROM QDRANT")

    score = results[0].get("score", 0) if results else 0

    return {
        "retrieved_text": results,
        "retrieval_relevance_score": score,
        "thought_process": [
            {
                "agent": "Text Agent",
                "action": f"Retrieved {len(results)} chunks from Qdrant",
            }
        ],
    }


# -----------------------------
# Second Retrieval After Rewrite
# -----------------------------


def requery_text_agent_node(state: AgentState):

    query = state.get("rewritten_query", "")
    source_name = state.get("source_name")
    document_id = state.get("document_id")

    results = search_text_chunks(
        query,
        source_name=source_name,
        document_id=document_id,
    )

    score = results[0].get("score", 0) if results else 0

    print(f"REQUERY: {query} -> " f"SCORE {score}")

    return {
        "retrieved_text": results,
        "second_score": score,
        "thought_process": [
            {
                "agent": "Text Agent Retry",
                "action": f"Retrieved {len(results)} chunks after rewrite",
            }
        ],
    }


# -----------------------------
# Vision Agent
# -----------------------------


def vision_agent_node(state: AgentState):

    query = _extract_query(state)

    results = search_images(query)

    return {
        "retrieved_images": results,
        "next_node": "FINISH",
        "thought_process": [
            {"agent": "Vision Agent", "action": f"Retrieved {len(results)} images"}
        ],
    }


# -----------------------------
# Web Agent
# -----------------------------


def web_agent_node(state: AgentState):

    query = _extract_query(state)

    results = execute_web_search(query)

    return {
        "retrieved_text": results,
        "next_node": "FINISH",
        "thought_process": [
            {"agent": "Web Agent", "action": f"Retrieved {len(results)} web results"}
        ],
    }


def sql_agent_node(state: AgentState):
    statement = _extract_query(state)

    result = sql_agent.execute(statement)

    thought = {
        "agent": "SQL Agent",
        "action": f"Generated SQL Query: {result.get('sql_query', '')}",
    }

    return {
        "sql_query": result.get("sql_query", ""),
        "sql_result": result.get("sql_result", []),
        "retrieved_text": result.get("retrieved_text", []),
        "next_node": "generator",
        "thought_process": [thought],
    }


# -----------------------------
# Routing
# -----------------------------


def route_next(state: AgentState):

    return state.get("next_node", "FINISH")


def evaluator_route(state: AgentState):

    if state.get("needs_requery", False):
        return "query_rewriter"

    return "generator"


def route_after_grader(state: AgentState) -> str:
    score = state.get("retrieval_relevance_score", "no")
    retry_count = state.get("retry_count", 0)

    if score == "yes" or (isinstance(score, (int, float)) and score >= 0.70):
        return "generator"
    elif retry_count < 3:
        return "query_rewriter"
    else:
        return "web_agent"


# -----------------------------
# Build Graph
# -----------------------------

workflow = StateGraph(AgentState)


# Nodes

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("text_agent", text_agent_node)
workflow.add_node("retrieval_evaluator", retrieval_evaluator_node)
workflow.add_node("grader", grader_node)
workflow.add_node("query_rewriter", query_rewriter_node)


workflow.add_node("requery_text_agent", requery_text_agent_node)


workflow.add_node("sql_agent", sql_agent_node)


workflow.add_node("vision_agent", vision_agent_node)


workflow.add_node("web_agent", web_agent_node)


workflow.add_node("generator", generator_node)


workflow.add_node("direct_llm", direct_llm_node)


# Entry

workflow.set_entry_point("supervisor")


# Supervisor Routing

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


# Text Self-RAG Loop

workflow.add_edge("text_agent", "retrieval_evaluator")


workflow.add_conditional_edges(
    "retrieval_evaluator",
    evaluator_route,
    {
        "query_rewriter": "query_rewriter",
        "generator": "generator",
    },
)


workflow.add_edge("query_rewriter", "requery_text_agent")


workflow.add_edge("requery_text_agent", "generator")


def blocked_guardrail_node(state: AgentState):
    """
    Fallback graph node executed when a user query is flagged as out-of-scope,
    toxic, or violating document domain guardrails.
    """
    thought = {
        "agent": "NeMo Guardrails",
        "action": (
            "Query flagged as out-of-scope or violating domain safety boundaries."
        ),
    }
    return {
        "next_node": "FINISH",
        "thought_process": [thought],
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "🛡️ **Query Blocked**: I can only answer questions directly "
                    "grounded in the provided PDF documents or historical "
                    "stock database. Please reframe your query within scope."
                ),
            }
        ],
    }


# Other Agents

workflow.add_node("blocked_guardrail", blocked_guardrail_node)

workflow.add_edge("sql_agent", "generator")
workflow.add_edge("text_agent", "generator")
workflow.add_edge("sql_agent", "generator")
workflow.add_edge("vision_agent", "generator")
workflow.add_edge("web_agent", "generator")


workflow.add_edge("vision_agent", "generator")


workflow.add_edge("web_agent", "generator")


workflow.add_edge("generator", END)


workflow.add_edge("direct_llm", END)

workflow.add_edge("blocked_guardrail", END)

app = workflow.compile()
