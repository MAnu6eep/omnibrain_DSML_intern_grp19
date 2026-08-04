import operator
from typing import Annotated, Any, List, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # Conversation
    messages: Annotated[List[BaseMessage], operator.add]

    # Graph routing
    next_node: str | None

    # Retrieved context
    retrieved_text: List[dict]
    retrieved_images: List[dict]

    # Agent reasoning
    thought_process: List[dict]

    # Self-RAG Evaluation
    retrieval_relevance_score: float
    retry_count: int

    # Text-to-SQL
    sql_query: str | None
    sql_result: Any