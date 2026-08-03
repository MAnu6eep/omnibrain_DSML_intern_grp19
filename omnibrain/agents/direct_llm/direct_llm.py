from langchain_core.messages import AIMessage

from omnibrain.agents.llm import get_llm_response
from omnibrain.agents.state.state import AgentState


def direct_llm_node(state: AgentState):
    query = state["messages"][-1].content

    answer, provider_used = get_llm_response(query)

    return {
        "messages": [AIMessage(content=answer)],
        "thought_process": [
            {
                "agent": "Direct LLM",
                "action": f"Answered query using {provider_used}.",
            }
        ],
    }
