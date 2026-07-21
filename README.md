# OmniBrain

OmniBrain is an enterprise-style Agentic Multi-Modal RAG Orchestrator. This repository currently contains the production-ready scaffold only: package boundaries, starter entrypoints, container wiring, and integration placeholders for later implementation.

## Top-Level Layout

- `omnibrain/app/` - FastAPI backend, API wiring, core config, services, models, schemas, and utilities.
- `omnibrain/agents/` - LangGraph-style orchestration placeholders for supervisor, search, vision, SQL, memory, state, nodes, tools, and prompts.
- `omnibrain/vectorstore/` - Qdrant client wrapper and retrieval scaffolding for text chunks and image embeddings.
- `frontend/streamlit/` - Streamlit chat UI placeholder for query and citation display.
- `evaluation/langfuse/` - Observability and evaluation placeholders.
- `guardrails/` - NeMo Guardrails configuration placeholders.
- `data/` - Upload, extraction, chunking, and image storage locations.
- `scripts/` - Prototype and utility scripts.
- `tests/` - Future automated tests.
- `docker/` - Docker support files and deployment helpers.
- `docs/` - Project documentation.

## Included Starter Components

- FastAPI application with a `/health` route.
- Streamlit chat-style placeholder UI.
- Environment-backed application settings.
- Placeholder Qdrant integration layer.
- Placeholder agent modules for multi-agent routing.
- Docker and Docker Compose support for FastAPI, Streamlit, and Qdrant.

## Quick Start

1. Copy `.env.example` to `.env` and fill in the required values.
2. Install dependencies with `uv`, Poetry, or `pip` using `pyproject.toml` or `requirements.txt`.
3. Run the backend with `uvicorn omnibrain.app.main:app --reload`.
4. Run the Streamlit UI with `streamlit run frontend/streamlit/app.py`.

## Team Members

- Anudeep
- Charan
- Om
- Manav
- Abhilash
- Meerja
## Backend Setup

Run the backend server locally using:

```bash
uvicorn apps.main:app --reload

## 🚀 Quick Start Runbook

### 1. Prerequisites
* Docker Desktop (running)
* Python 3.11+