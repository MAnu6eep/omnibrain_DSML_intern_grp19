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

1. Start Qdrant.(open docker before hand to avoid errors)

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
Each folder has differnet and many files which work in a diffenent way.
Database folder inside service now contains connection.py,postgress.y and sqlite.py for the sql commands to run on our project smoothly.
## Notes

- The codebase targets Python 3.11 to keep the dependency set consistent across machines.
- The backend can start and respond even when Gemini credentials are missing; the generator falls back to a safe message instead of failing during import.

## 🐙 Git Branching Architecture & Workflow Protocols

To maintain codebase stability and prevent merge conflicts, all team members must adhere to the following Git standards.

### 🌿 1. Branch Strategy

```
  main          ──[ Production Releases Only ]──────────────────────────►
                   ▲
  develop       ───┼──[ Daily Team Integration Branch ]─────────────────►
                   │          ▲                    ▲
  feature/*     ───┴──────────┴─[ Individual Developer Feature Branches ]
```

- **`main`**: Production-ready release branch. Directly locked.
- **`develop`**: Main team integration branch. All features are merged here via Pull Requests.
- **`feature/<feature-name>`**: Isolated developer workspace (e.g., `feature/vlm-reasoner`, `feature/sql-agent`).

---

### 🔄 2. Standard Daily Development Workflow

#### Step A: Start New Work
Always branch off the latest `develop`:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

#### Step B: Commit Changes
Write clear, scoped commit messages using conventional prefixes:
```bash
git add .
git commit -m "feat(vision): add VLM chart numerical parser"
```
> **Commit Prefixes**: `feat:` (new feature), `fix:` (bug fix), `docs:` (documentation), `test:` (tests), `refactor:` (code cleanup), `build:` (dependencies/docker).

#### Step C: Push & Open Pull Request
Push your feature branch and open a Pull Request (PR) into `develop`:
```bash
git push -u origin feature/your-feature-name
```

---

### 🛠️ 3. Scenario-Based Troubleshooting Runbook

#### 📍 Scenario 1: You have uncommitted local work and need to pull remote changes
**Goal**: Safely pull `origin/develop` without losing your uncommitted edits.

- **Approach A (Commit first - Recommended)**:
  ```bash
  git add .
  git commit -m "wip: save local progress"
  git pull origin develop --no-rebase
  git push origin develop
  ```
- **Approach B (Stash temporary work)**:
  ```bash
  git stash                              # Save uncommitted work to stash
  git pull origin develop                 # Fetch remote updates
  git stash pop                          # Re-apply your uncommitted work
  ```

---

#### 📍 Scenario 2: Pre-Commit Hooks Abort Your Commit (`black` / `flake8` / `isort` failures)
**Cause**: Code formatting or line length violations triggered by `.pre-commit-config.yaml`.

- **Fix Steps**:
  1. Inspect pre-commit errors (e.g., `line too long` or `reformatted file`).
  2. If files were reformatted by `black` or `isort`, simply re-stage them:
     ```bash
     git add .
     ```
  3. Re-run your commit command:
     ```bash
     git commit -m "feat(scope): your descriptive commit message"
     ```

---

#### 📍 Scenario 3: Resolving Merge Conflicts (`<<<<<<< HEAD` vs `>>>>>>> origin/develop`)
**Cause**: Both you and a teammate edited the same file lines.

- **Fix Steps**:
  1. Open the conflicting file. Locate conflict markers:
     ```text
     <<<<<<< HEAD (Your local changes)
     model: gpt-4o-mini
     =======
     model: gpt-3.5-turbo
     >>>>>>> origin/develop (Teammate's remote changes)
     ```
  2. Keep the correct code block and delete the marker lines (`<<<<<<<`, `=======`, `>>>>>>>`).
  3. Save the file, stage it, and complete the merge:
     ```bash
     git add path/to/resolved_file.py
     git commit -m "fix: resolve merge conflict in resolved_file.py"
     ```

---

#### 📍 Scenario 4: Updating Your Feature Branch with Latest `develop`
**Goal**: Keep your long-running feature branch updated with recent team merges.

```bash
git checkout feature/your-feature-name
git fetch origin
git merge origin/develop
```
*(Resolve any conflicts if prompted, then test your code locally).*

---

#### 📍 Scenario 5: Made Edits on the Wrong Branch
**Goal**: Move your uncommitted work from `develop` to a new `feature` branch.

```bash
git stash                              # Stash your current edits
git checkout -b feature/correct-branch # Create & switch to target branch
git stash pop                          # Apply stashed edits to the new branch
```

---

#### 📍 Scenario 6: Discard Unwanted Local Changes
- **Discard edits in a single file**:
  ```bash
  git checkout -- path/to/file.py
  ```
- **Reset all untracked files & local changes back to last commit**:
  ```bash
  git reset --hard HEAD
  git clean -fd
  ```
  *(⚠️ Warning: This permanently deletes uncommitted local changes).*

---

### ⛔ Golden Rules for Team Collaboration

1. **NEVER force push (`git push -f`) to `main` or `develop`**.
2. **Always pull before pushing**: Run `git pull origin develop` before submitting PRs.
3. **Keep PRs focused**: One feature or bug fix per Pull Request.
4. **Never commit API Keys or Secret Tokens**: Store credentials strictly inside `.env` (which is listed in `.gitignore`).


Case 1: You have NO uncommitted changes on develop
If your git status is clean, simply run:

git checkout develop
git pull origin develop


🟡 Case 2: You HAVE uncommitted local changes on develop
If you have edited files locally that are not yet committed:

Option A: Commit your local work first (Recommended)

git add .
git commit -m "docs: update README with git operations guide"
git pull origin develop --no-rebase
git push origin develop

Option B: Temporary Stash (If you don't want to commit yet)

git stash                              # Temporarily hide uncommitted work
git pull origin develop                # Pull latest team updates
git stash pop                          # Re-apply your local edits
Current status:
Upload the pdf in get pdf checkpoint and get a upload it by this u can access your pdf in browser and also can go to specific pages by typing the page number in checkpoint.