from omnibrain.agents.nodes.query_rewriter import QueryRewriter


rewriter = QueryRewriter()


def query_rewriter_node(state):

    messages = state.get("messages", [])

    if not messages:
        return {
            "needs_requery": False
        }

    last_message = messages[-1]

    if hasattr(last_message, "content"):
        query = last_message.content
    else:
        query = str(last_message)

    retrieved_docs = state.get(
        "retrieved_text",
        []
    )

    rewritten_query = rewriter.rewrite(
        query,
        retrieved_docs
    )

    print(
        f"Original query: {query}"
    )

    print(
        f"Rewritten query: {rewritten_query}"
    )

    return {
        "rewritten_query": rewritten_query,
        "needs_requery": False,
        "thought_process": [
            {
                "agent": "Query Rewriter",
                "action": (
                    f"Rewritten query: {rewritten_query}"
                )
            }
        ]
    }