from dataclasses import dataclass
import os

from langchain_google_genai import ChatGoogleGenerativeAI


@dataclass
class SQLAgent:
    name: str = "sql_agent"

    def _get_llm(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return None

        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0,
        )

    def execute(self, state: dict, statement: str) -> dict[str, str]:
        """
        Converts a natural language database request into
        an SQL query using Gemini.
        """

        llm = self._get_llm()

        if llm is None:
            return {
                "agent": self.name,
                "status": "Gemini API key not configured.",
                "sql": ""
            }

        prompt = f"""
            Convert the following request into a valid SQL query.

            Request:
            {statement}

            Return only the SQL query.
            """

        try:
            response = llm.invoke(prompt)

            return {
                "agent": self.name,
                "status": "success",
                "sql": response.content.strip()
            }

        except Exception as e:
            return {
                "agent": self.name,
                "status": f"error: {str(e)}",
                "sql": ""
            }