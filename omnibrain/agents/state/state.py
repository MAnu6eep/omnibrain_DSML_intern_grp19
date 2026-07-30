import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_node: str | None  # "text_agent", "vision_agent", "web_agent", or "FINISH"
    retrieved_text: List[dict]
    retrieved_images: List[dict]
    thought_process: List[dict]  # [{"agent": "...", "action": "..."}]
