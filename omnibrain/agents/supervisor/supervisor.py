from dataclasses import dataclass, field


@dataclass
class Supervisor:
    agent_names: list[str] = field(default_factory=lambda: ["search_agent", "vision_agent", "sql_agent"])

    def route(self, task: str) -> str:
        del task
        return self.agent_names[0]
