# OmniBrain

OmniBrain is a multimodal RAG workspace with a FastAPI backend, a Streamlit frontend, PDF ingestion, image extraction, text chunking, embeddings, Qdrant storage, and a Week 2 agentic chat scaffold.

## Supported local setup

- Python 3.11.x
- Qdrant via Docker
- FastAPI backend via Uvicorn
- Streamlit frontend via the Streamlit CLI

## Environment

1. Copy `.env.example` to `.env`.
2. Fill in `GEMINI_API_KEY` if you want live LLM synthesis.
3. Keep `QDRANT_URL=http://localhost:6333` for local development.
4. Leave the Langfuse variables empty unless you have those credentials.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Recommended local run procedure

1. Start Qdrant.

```bash
docker compose up -d qdrant
```

2. Start the backend.

```bash
uvicorn omnibrain.app.main:app --reload --host 0.0.0.0 --port 8000
```

3. Start the frontend in a second terminal.

```bash
streamlit run frontend/streamlit/app.py --server.port 8501
```

4. Open the frontend at `http://localhost:8501`.

## Docker alternative

If you want to run all services through compose instead of mixing local and container processes:

```bash
docker compose up --build
```

That starts Qdrant, FastAPI, and Streamlit with the compose-defined environment variables.

## Verification commands

Run the basic backend tests:

```bash
pytest omnibrain/app/core/test
```

Run the ingestion validation smoke test:

```bash
python scripts/validate_pipeline.py --input data/Attention_is_all_you_need.pdf
```

Check the backend health endpoint:

```bash
curl http://localhost:8000/health
```

Check the chat endpoint scaffold:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What is in the PDF?\"}"
```

## Upload and ingestion flow

1. Use the Streamlit sidebar upload control or call `POST /api/v1/ingestion/upload` directly.
2. The backend stores text chunks and extracted images in Qdrant.
3. The chat endpoint reads sanitized retrieval results and falls back safely when nothing useful is found.

## Current partial areas

- Table extraction is intentionally partial. The pipeline continues with text and image extraction, and the response includes a warning when no table path is available.
- The agentic chat flow is scaffolded for Week 2 and can run in a deterministic fallback mode when `GEMINI_API_KEY` is not configured.

## Hardware-aware workload guidance

Keep the default workload conservative on teammate laptops:

- 20–50 pages: 20–30 PDFs
- 100–200 pages: 5–10 PDFs
- 300–500 pages: 2–5 PDFs

If ingestion starts to lag, reduce batch size before increasing chunk size or embedding throughput.

## Project layout

- `omnibrain/app/` - FastAPI backend, schemas, services, and tests.
- `omnibrain/agents/` - LangGraph-style chat routing and synthesis.
- `omnibrain/vectorstore/` - Qdrant collections, indexers, and retrievers.
- `frontend/streamlit/` - Streamlit chat and upload UI.
- `scripts/validate_pipeline.py` - Local ingestion smoke test.

## Notes

- The codebase targets Python 3.11 to keep the dependency set consistent across machines.
- The backend can start and respond even when Gemini credentials are missing; the generator falls back to a safe message instead of failing during import.
