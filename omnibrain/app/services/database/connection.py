import os
import sqlite3

try:
    import psycopg2
except ImportError:
    psycopg2 = None


class DatabaseConnection:
    def __init__(self):
        self.db_type = os.getenv("DB_TYPE", "sqlite").lower()

    def connect(self):
        if self.db_type == "sqlite":
            db_path = os.getenv("SQLITE_DB_PATH", "stocks.db")
            return sqlite3.connect(db_path)

        elif self.db_type == "postgres":
            if psycopg2 is None:
                raise ImportError("psycopg2 is not installed.")

            return psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=os.getenv("POSTGRES_PORT", "5432"),
                database=os.getenv("POSTGRES_DB", "stocks"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            )

        raise ValueError(f"Unsupported database type: {self.db_type}")