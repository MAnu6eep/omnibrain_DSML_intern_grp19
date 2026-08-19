from dataclasses import dataclass

from omnibrain.agents.tools.web_search import execute_web_search


@dataclass
class SearchAgent:
    name: str = "search_agent"

    def execute(self, query: str) -> dict:
        """
        Executes web search using DuckDuckGo search engine.
        """
        results = execute_web_search(query)
        return {
            "agent": self.name,
            "query": query,
            "status": "completed",
            "results": results,
        }
