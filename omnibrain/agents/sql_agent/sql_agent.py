from dataclasses import dataclass


@dataclass
class SQLAgent:
    name: str = "sql_agent"

    def execute(self, statement: str) -> dict[str, str]:
        return {
            "agent": self.name,
            "statement": statement,
            "status": "placeholder",
        }


def sql_agent_node(state):
    """
    Placeholder SQL agent node for LangGraph workflow.
    """

    agent = SQLAgent()

    query = ""

    if "messages" in state and state["messages"]:
        last = state["messages"][-1]

        if hasattr(last, "content"):
            query = last.content
        else:
            query = str(last)

    result = agent.execute(query)

    state["sql_result"] = result

    return state