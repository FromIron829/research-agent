# Stage 2 — Deployment Plan (sketch)

Goal: ship the Stage 2 agent as a **stateless, read-only HTTP API** on AWS — a clickable portfolio demo, and the deployment skeleton Stage 3 will extend. Deliberately right-sized (no Kubernetes).

## What the service needs at runtime (grounded in the code)

- **Deps:** `anthropic` (agent LLM), `openai` (query embeddings — `text-embedding-3-small`), `chromadb`, `rank_bm25`, `python-dotenv`, plus `fastapi` + `uvicorn` for serving. (`pymupdf4llm`/`langchain-text-splitters` are build-time only — not needed to *serve* Stage 2; can trim for a leaner image.)
- **Two secrets:** `ANTHROPIC_API_KEY` (agent) **and** `OPENAI_API_KEY` (retrieval embeds the query). People forget the second one — retrieval breaks without it.
- **Two data artifacts baked into the image:**
  - `stage_1/data/chunks.json` — tracked in git; BM25 corpus + chunk text.
  - `stage_1/data/chroma_db/` — **gitignored** (regenerable). It must still be copied into the image: copy the prebuilt local index; **do NOT regenerate it in the Docker build** (that re-pays embedding cost and isn't reproducible). ⚠️ Make sure `.dockerignore` does *not* exclude it (gitignore ≠ dockerignore).
- **Startup cost:** `hybrid.py` builds BM25 and opens Chroma **at import** (`_bm25 = BM25Okapi(...)`, `_collection = PersistentClient(...)`). So process start tokenizes the whole corpus + loads the index — a few seconds. Matters for cold starts (below).

## 1. API layer (new, thin)

A small FastAPI wrapper over `agent.answer()`:
- `POST /ask` `{ "question": "..." }` → `{ answer, citations, iterations, timing, usage }`
- `GET /health` → 200 (for the platform health check)
- `GET /docs` → FastAPI's Swagger UI **for free** — a usable demo surface with zero frontend work.
- (Optional) `POST /ask/stream` (SSE) — you already have streaming in the agent.

Import `agent` once at module load (pays the BM25/Chroma cost at boot, not per request). Each request is independent → trivially horizontally scalable.

## 2. Containerize

```dockerfile
FROM python:3.13-slim
WORKDIR /app
# uv for installs (project already uses uv + uv.lock)
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY stage_1/ stage_1/         # code + chunks.json + chroma_db (ensure not in .dockerignore)
COPY stage_2/ stage_2/
ENV PORT=8080
CMD ["uv", "run", "uvicorn", "stage_2.api:app", "--host", "0.0.0.0", "--port", "8080"]
```
- Add a `.dockerignore` (exclude `.venv`, `__pycache__`, `paper_pdfs/`, `test.py`, `*.md` if desired) **but keep `stage_1/data/`**.
- Sanity-check image size; the index + chunks should be tens–low-hundreds of MB.

## 3. Secrets — never in the image

- Store `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in **AWS Secrets Manager** (or platform env vars), injected at runtime.
- `.env` stays gitignored and is *not* copied into the image.

## 4. AWS target — App Runner (recommended)

Simplest path that still says "I ship to AWS":
1. Build image → push to **ECR**.
2. **App Runner** service from the ECR image: set port 8080, health check `/health`, inject the two secrets, **min instances = 1** (avoids cold-start on the import-time index load), small CPU/mem.
3. Get the auto-provided HTTPS URL → that's your demo link.

Alternatives: **ECS Fargate** (more control, more setup) · **Lambda container** (scale-to-zero = cheapest, but the import-time BM25/Chroma load makes cold starts slow — workable but fiddlier). Skip EKS.

## 5. ⚠️ Cost & abuse control (the #1 real-world concern)

A public endpoint where each call costs **~$0.15** in Anthropic+OpenAI tokens is a **standing bill liability** — one scraper and you wake up to a large invoice. Do at least one before going public:
- A shared **API key / secret header** the caller must send (simplest), or basic auth on the endpoint, or
- **Rate limiting** (per-IP) and/or a **daily request cap**, and
- An **AWS Budget alarm** on the account regardless.

Also keep `MAX_ITERATIONS` and per-turn `max_tokens` bounded (already are) to cap per-request cost.

## 6. Optional polish (nice signals, not required)

- A single static `index.html` calling `/ask` for a friendlier demo than Swagger.
- **CI/CD:** GitHub Actions → build → push ECR → App Runner auto-deploy on push to `main`. Good "I automate deploys" signal.
- CloudWatch logs + a basic latency/error dashboard.

## Suggested sequence

1. Write `stage_2/api.py` (FastAPI) + run locally (`uvicorn`) and hit `/ask`.
2. Add `Dockerfile` + `.dockerignore`; build and run the container **locally**; confirm retrieval works inside the container (this is where a missing `chroma_db` or absent `OPENAI_API_KEY` shows up).
3. Push to ECR; create the App Runner service with secrets + health check.
4. Add abuse/cost guard + budget alarm **before** sharing the URL.
5. (Optional) static frontend + CI/CD.

## Open questions

- App Runner min=1 (always-on, ~$15-25/mo, no cold start) vs. Lambda scale-to-zero (cheap, slow cold start)? For a low-traffic demo that must feel responsive when clicked, min=1 wins.
- Public vs. gated demo (the cost guard decides this).
- Stage 3 will need this to become **writable + stateful** (hosted vector store + checkpoint DB) — keep the API layer and Dockerfile structured so that swap is localized.
