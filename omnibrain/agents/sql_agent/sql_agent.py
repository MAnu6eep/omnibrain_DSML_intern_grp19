from dataclasses import dataclass


@dataclass
class SQLAgent:
    name: str = "sql_agent"

    def execute(self, statement: str) -> dict[str, str]:
        return {"agent": self.name, "statement": statement, "status": "placeholder"}
