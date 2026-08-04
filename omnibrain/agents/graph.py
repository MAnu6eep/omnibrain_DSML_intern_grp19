from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from omnibrain.agents.generator.generator import generator_node
from omnibrain.agents.state.state import AgentState
from omnibrain.agents.supervisor.supervisor import supervisor_node
from omnibrain.agents.tools.web_search import execute_web_search
from omnibrain.vectorstore.retrievers.image_retriever import search_images
from omnibrain.vectorstore.retrievers.text_retriever import search_text_chunks
from omnibrain.agents.sql_agent.sql_agent import SQLAgent
sql_agent = SQLAgent()


def text_agent_node(state: AgentState):
    query = state["messages"][-1].content

    print("QUERY:", query)

    results = search_text_chunks(query)

    print("=" * 80)
    print("TEXT RETRIEVAL RESULTS")
    print(results)
    print("NUMBER OF RESULTS:", len(results))
    print("=" * 80)

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
    query = state["messages"][-1].content
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
    query = state["messages"][-1].content
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

def sql_agent_node(state: AgentState):
    statement = state["messages"][-1].content

    result = sql_agent.execute(state, statement)

    thought = {
        "agent": "SQL Agent",
        "action": "Generated SQL query."
    }

    return {
        "sql_result": result,
        "next_node": "FINISH",
        "thought_process": [thought],
    }

def route_next(state: AgentState) -> str:
    return state.get("next_node", "FINISH")


workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("text_agent", text_agent_node)
workflow.add_node("vision_agent", vision_agent_node)
workflow.add_node("web_agent", web_agent_node)
workflow.add_node("sql_agent", sql_agent_node)
workflow.add_node("generator", generator_node)
workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_next,
    {
        "text_agent": "text_agent",
        "vision_agent": "vision_agent",
        "web_agent": "web_agent",
        "sql_agent": "sql_agent",
        "FINISH": END,
    },
)

workflow.add_edge("text_agent", "generator")
workflow.add_edge("vision_agent", "generator")
workflow.add_edge("web_agent", "generator")
workflow.add_edge("sql_agent", "generator")

workflow.add_edge("generator", END)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)
