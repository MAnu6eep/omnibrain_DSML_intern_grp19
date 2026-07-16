# Contributing

## Local Setup

1. Create a virtual environment.
2. Install dependencies from `pyproject.toml` or `requirements.txt`.
3. Copy `.env.example` to `.env`.
4. Run the backend with `uvicorn omnibrain.app.main:app --reload`.
5. Run the Streamlit UI with `streamlit run omnibrain/frontend/streamlit/app.py`.

## Repository Expectations

- Keep changes small and focused.
- Preserve the package layout under `omnibrain/`.
- Add tests for any future behavior changes.
