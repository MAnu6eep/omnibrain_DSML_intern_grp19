import json
import os
import sqlite3
from typing import Any, Dict, List

DB_PATH = os.path.join("data", "omnibrain_stock.db")


def _ensure_db_populated():
    """
    Initializes SQLite tables for PDF chunk metadata registry and document history.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table 1: PDF Document Metadata Registry
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT UNIQUE,
            filename TEXT NOT NULL,
            total_pages INTEGER NOT NULL,
            text_chunk_count INTEGER NOT NULL,
            image_count INTEGER NOT NULL,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Table 2: PDF Chunk Index Metadata Pointer Table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pdf_chunk_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE,
            document_id TEXT NOT NULL,
            source TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            text_preview TEXT NOT NULL,
            modality TEXT DEFAULT 'text',
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()


def record_pdf_ingestion_in_sqlite(
    filename: str,
    document_id: str,
    total_pages: int,
    text_chunks: list,
    images: list,
):
    """
    Writes PDF chunk pointers and document metadata directly into SQLite database.
    """
    _ensure_db_populated()
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Insert / Replace Document Registry Entry
        cursor.execute(
            """
            INSERT OR REPLACE INTO documents
            (document_id, filename, total_pages, text_chunk_count, image_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (document_id, filename, total_pages, len(text_chunks), len(images)),
        )

        # Insert Chunk Index Pointers
        chunk_rows = []
        for chunk in text_chunks:
            chunk_id = getattr(chunk, "chunk_id", str(chunk.get("chunk_id", "")))
            page_num = getattr(chunk, "page_number", int(chunk.get("page_number", 1)))
            text = getattr(chunk, "text", str(chunk.get("text", "")))
            source = getattr(chunk, "source", filename)

            if chunk_id and text:
                chunk_rows.append((chunk_id, document_id, source, page_num, text[:300]))

        cursor.executemany(
            """
            INSERT OR REPLACE INTO pdf_chunk_metadata
            (chunk_id, document_id, source, page_number, text_preview)
            VALUES (?, ?, ?, ?, ?)
            """,
            chunk_rows,
        )

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging PDF metadata to SQLite: {e}")


class SQLAgent:
    name: str = "sql_agent"

    def __init__(self):
        _ensure_db_populated()

    def execute(self, user_query: str) -> Dict[str, Any]:
        """
        Translates user queries into SQL to query SQLite metadata registry tables,
        retrieving relevant matching PDF chunk pointers to attach to the LLM context.
        """
        _ensure_db_populated()
        query_lower = user_query.lower().strip()

        # Determine target search keywords or document names from query
        target_source = None
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT source FROM pdf_chunk_metadata")
        existing_sources = [r[0] for r in cursor.fetchall()]

        for src in existing_sources:
            if (
                src.lower() in query_lower
                or src.lower().replace(".pdf", "") in query_lower
            ):
                target_source = src
                break

        # Generate SQL Query against SQLite PDF Metadata Index
        if target_source:
            sql_query = (
                f"SELECT chunk_id, source, page_number, text_preview "
                f"FROM pdf_chunk_metadata WHERE source = '{target_source}' LIMIT 5;"
            )
        elif (
            "list" in query_lower
            or "show all" in query_lower
            or "document" in query_lower
        ):
            sql_query = (
                "SELECT filename, total_pages, text_chunk_count, image_count, upload_timestamp "
                "FROM documents ORDER BY id DESC;"
            )
        else:
            # General keyword search in SQLite metadata preview
            keywords = [
                w
                for w in query_lower.split()
                if len(w) > 3
                and w not in ["show", "find", "select", "where", "what", "from"]
            ]
            if keywords:
                search_term = keywords[0]
                sql_query = (
                    f"SELECT chunk_id, source, page_number, text_preview "
                    f"FROM pdf_chunk_metadata WHERE text_preview LIKE '%{search_term}%' LIMIT 5;"
                )
            else:
                sql_query = (
                    "SELECT chunk_id, source, page_number, text_preview "
                    "FROM pdf_chunk_metadata ORDER BY id DESC LIMIT 5;"
                )

        # Execute query against SQLite DB
        rows = []
        columns = []
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql_query)
            fetched = cursor.fetchall()
            if fetched:
                columns = fetched[0].keys()
                rows = [dict(r) for r in fetched]
            conn.close()
        except Exception as e:
            rows = [{"error": str(e)}]

        # Format retrieved SQLite context passages for Generator LLM
        retrieved_passages = []
        for r in rows:
            if "text_preview" in r:
                passage_text = (
                    f"[SQLite Index Metadata] Source: {r.get('source')}, Page {r.get('page_number')}, "
                    f"Chunk: {r.get('chunk_id')} | Preview: {r.get('text_preview')}"
                )
                retrieved_passages.append(
                    {
                        "text": passage_text,
                        "source": r.get("source", "SQLite Metadata"),
                        "page": r.get("page_number", 1),
                        "chunk_id": r.get("chunk_id", "sql_meta"),
                    }
                )
            elif "filename" in r:
                passage_text = (
                    f"[SQLite Document Registry] File: {r.get('filename')}, Pages: {r.get('total_pages')}, "
                    f"Chunks: {r.get('text_chunk_count')}, Images: {r.get('image_count')}"
                )
                retrieved_passages.append(
                    {
                        "text": passage_text,
                        "source": r.get("filename", "Document Registry"),
                        "page": 1,
                        "chunk_id": "doc_registry",
                    }
                )

        return {
            "sql_query": sql_query,
            "sql_result": rows,
            "retrieved_text": retrieved_passages,
            "columns": list(columns),
        }
