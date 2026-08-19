def build_nodes() -> dict[str, str]:
    """
    Returns registered agentic nodes for the LangGraph orchestrator.
    """
    return {
        "supervisor": "supervisor_node",
        "search": "web_agent_node",
        "vision": "vision_agent_node",
        "sql": "sql_agent_node",
    }
