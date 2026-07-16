from dataclasses import dataclass, field


@dataclass
class AgentState:
    active_agent: str = "supervisor"
    messages: list[dict[str, str]] = field(default_factory=list)
