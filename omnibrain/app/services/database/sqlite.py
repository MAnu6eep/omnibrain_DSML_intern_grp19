from .connection import DatabaseConnection


class SQLiteSchema:
    def __init__(self):
        self.conn = DatabaseConnection().connect()
        self.cursor = self.conn.cursor()

    def create_tables(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            trade_date TEXT,
            open_price REAL,
            high_price REAL,
            low_price REAL,
            close_price REAL,
            volume INTEGER
        );
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS financial_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            fiscal_year INTEGER,
            revenue REAL,
            net_income REAL,
            eps REAL
        );
        """)

        self.conn.commit()

    def close(self):
        self.conn.close()