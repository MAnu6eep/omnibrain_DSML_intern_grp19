from dataclasses import dataclass


@dataclass
class SearchAgent:
    name: str = "search_agent"

    def run(self, query: str) -> dict[str, str]:
        return {"agent": self.name, "query": query, "status": "placeholder"}
