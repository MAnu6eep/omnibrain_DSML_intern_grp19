from .connection import DatabaseConnection


class PostgresSchema:
    def __init__(self):
        self.conn = DatabaseConnection().connect()
        self.cursor = self.conn.cursor()

    def create_tables(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS historical_stock_data (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) NOT NULL,
            trade_date DATE,
            open_price DOUBLE PRECISION,
            high_price DOUBLE PRECISION,
            low_price DOUBLE PRECISION,
            close_price DOUBLE PRECISION,
            volume BIGINT
        );
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS financial_data (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(20) NOT NULL,
            fiscal_year INT,
            revenue DOUBLE PRECISION,
            net_income DOUBLE PRECISION,
            eps DOUBLE PRECISION
        );
        """)

        self.conn.commit()

    def close(self):
        self.conn.close()