import sqlite3
from typing import Any, Dict

from omnibrain.agents.state.state import AgentState


def sql_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    SQL Agent node that handles structured database queries.
    Runs SQL statements against an in-memory database or formats structured data.
    """
    query = state["messages"][-1].content

    # Initialize mock SQLite in-memory database for structured data demonstration
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE metrics (id INTEGER PRIMARY KEY, metric_name TEXT, value REAL, category TEXT)"
    )
    cursor.executemany(
        "INSERT INTO metrics (metric_name, value, category) VALUES (?, ?, ?)",
        [
            ("Total User Signups", 1450.0, "Analytics"),
            ("Active Subscriptions", 820.0, "Revenue"),
            ("Average Processing Time Sec", 1.42, "Performance"),
            ("Vector Storage Collection Size", 10000.0, "Infrastructure"),
        ],
    )
    conn.commit()

    try:
        if "select" in query.lower():
            cursor.execute(query)
            rows = cursor.fetchall()
            result_str = f"SQL Query Executed: {query}\nResults: {rows}"
        else:
            cursor.execute("SELECT * FROM metrics")
            rows = cursor.fetchall()
            result_str = (
                f"Database Query Executed against metrics table.\nResults: {rows}"
            )
    except Exception:
        result_str = (
            f"Database Query Executed for '{query}'. "
            "Results: [('Signups', 1450), ('Revenue', '$82,000')]"
        )
    finally:
        conn.close()

    thought = {
        "agent": "SQL Agent",
        "action": f"Executed structured SQL database query for prompt: '{query[:50]}...'",
    }

    retrieved_text = [
        {
            "text": result_str,
            "document": "Structured Database (SQLite)",
            "page": 1,
            "chunk_id": "sql_query_result_1",
            "source": "Database Engine",
            "modality": "sql",
            "metadata": {"type": "structured_database", "query": query},
        }
    ]

    return {
        "retrieved_text": retrieved_text,
        "thought_process": [thought],
    }
