import os

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from omnibrain.agents.state.state import AgentState


def _get_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        google_api_key=api_key,
        temperature=0,
    )


def direct_llm_node(state: AgentState):

    query = state["messages"][-1].content

    llm = _get_llm()

    if llm is None:
        return {
            "messages": [
                AIMessage(content="Gemini API Key not configured.")
            ],
            "thought_process": [
                {
                    "agent": "Direct LLM",
                    "action": "API key missing."
                }
            ]
        }

    response = llm.invoke(query)

    return {
        "messages": [
            AIMessage(content=response.content)
        ],
        "thought_process": [
            {
                "agent": "Direct LLM",
                "action": "Answered directly using Gemini without vector retrieval."
            }
        ]
    }