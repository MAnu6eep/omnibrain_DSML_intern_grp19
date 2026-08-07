import sys

from langgraph.graph import END, StateGraph

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


from omnibrain.agents.direct_llm.direct_llm import direct_llm_node
from omnibrain.agents.generator.generator import generator_node
from omnibrain.agents.sql_agent.sql_agent import sql_agent_node
from omnibrain.agents.state.state import AgentState
from omnibrain.agents.supervisor.supervisor import supervisor_node

from omnibrain.agents.evaluators.retrieval_evaluator import (
    retrieval_evaluator_node
)

from omnibrain.agents.nodes.query_rewriter_node import (
    query_rewriter_node
)

from omnibrain.agents.tools.web_search import execute_web_search

from omnibrain.vectorstore.retrievers.image_retriever import search_images
from omnibrain.vectorstore.retrievers.text_retriever import search_text_chunks
from omnibrain.agents.sql_agent.sql_agent import SQLAgent
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

    results = search_text_chunks(query)

    print(
        f"QUERY: {query} -> "
        f"RETRIEVED {len(results)} CHUNKS FROM QDRANT"
    )


    score = (
        results[0].get("score", 0)
        if results
        else 0
    )


    return {

        "retrieved_text": results,

        "retrieval_relevance_score": score,

        "thought_process": [
            {
                "agent": "Text Agent",
                "action":
                    f"Retrieved {len(results)} chunks from Qdrant"
            }
        ],
    }



# -----------------------------
# Second Retrieval After Rewrite
# -----------------------------

def requery_text_agent_node(state: AgentState):

    query = state.get(
        "rewritten_query",
        ""
    )


    results = search_text_chunks(query)


    score = (
        results[0].get("score", 0)
        if results
        else 0
    )


    print(
        f"REQUERY: {query} -> "
        f"SCORE {score}"
    )


    return {

        "retrieved_text": results,

        "second_score": score,

        "thought_process": [
            {
                "agent": "Text Agent Retry",
                "action":
                    f"Retrieved {len(results)} chunks after rewrite"
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
            {
                "agent": "Vision Agent",
                "action":
                    f"Retrieved {len(results)} images"
            }
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
            {
                "agent": "Web Agent",
                "action":
                    f"Retrieved {len(results)} web results"
            }
        ],
    }

def sql_agent_node(state: AgentState):
    statement = state["messages"][-1].content

    result = sql_agent.execute(statement)

    thought = {
        "agent": "SQL Agent",
        "action": "Generated SQL query."
    }

    return {
        "sql_result": result,
        "next_node": "FINISH",
        "thought_process": [thought],
    }


# -----------------------------
# Routing
# -----------------------------

def route_next(state: AgentState):

    return state.get(
        "next_node",
        "FINISH"
    )



def evaluator_route(state: AgentState):

    if state.get(
        "needs_requery",
        False
    ):
        return "query_rewriter"


    return "generator"



# -----------------------------
# Build Graph
# -----------------------------

workflow = StateGraph(AgentState)


# Nodes

workflow.add_node(
    "supervisor",
    supervisor_node
)


workflow.add_node(
    "text_agent",
    text_agent_node
)


workflow.add_node(
    "retrieval_evaluator",
    retrieval_evaluator_node
)


workflow.add_node(
    "query_rewriter",
    query_rewriter_node
)


workflow.add_node(
    "requery_text_agent",
    requery_text_agent_node
)


workflow.add_node(
    "sql_agent",
    sql_agent_node
)


workflow.add_node(
    "vision_agent",
    vision_agent_node
)


workflow.add_node(
    "web_agent",
    web_agent_node
)


workflow.add_node(
    "generator",
    generator_node
)


workflow.add_node(
    "direct_llm",
    direct_llm_node
)



# Entry

workflow.set_entry_point(
    "supervisor"
)



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

workflow.add_edge(
    "text_agent",
    "retrieval_evaluator"
)


workflow.add_conditional_edges(

    "retrieval_evaluator",

    evaluator_route,

    {

        "query_rewriter":
            "query_rewriter",

        "generator":
            "generator",
    },
)



workflow.add_edge(
    "query_rewriter",
    "requery_text_agent"
)


workflow.add_edge(
    "requery_text_agent",
    "generator"
)



# Other Agents

workflow.add_edge(
    "sql_agent",
    "generator"
)


workflow.add_edge(
    "vision_agent",
    "generator"
)


workflow.add_edge(
    "web_agent",
    "generator"
)



workflow.add_edge(
    "generator",
    END
)


workflow.add_edge(
    "direct_llm",
    END
)



app = workflow.compile()