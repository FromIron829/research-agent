# Research Agent — Production-Grade Corrective RAG

A retrieval-augmented research agent over **77 arXiv papers on efficient LLM inference**, built as a **graph-structured Corrective-RAG system** on LangGraph and deployed to AWS. It answers questions with cited, fabrication-checked answers; when the corpus can't answer, it proposes a paper, asks the user to approve, ingests it live, and retries.

The repository is also a **deliberate engineering log**: every feature was shipped through a *Setup → Hypothesis → Result → Interpretation → Decision* experiment with the hypothesis committed before the result. **22 experiments** are recorded in [`stage_3/results/EXPERIMENTS.md`](stage_3/results/EXPERIMENTS.md) — including the bugs found, the null results, and the things that didn't work.

**🔗 Live demo:** https://re-7a9f0af6e9724f82928962179aa42f09.ecs.us-east-2.on.aws
**Demo API key** (shared, rate-limited, rotatable): `ra_5e0ybrccg2Gqe2OMJZL4yU-2mOISil87`
> The page asks for the key once. Try: *"How does FlashAttention reduce memory I/O?"*, then a follow-up like *"summarize that"* (watch it answer from memory, no retrieval), then something out-of-corpus like *"What is the Mamba architecture?"* to see the corrective loop propose adding a paper.

---

## What it does

Ask a question and watch the agent's reasoning stream live, node by node:

```
Checked conversation memory → Classified intent → Planned retrieval → Retrieved sources
→ Graded source relevance → Drafted an answer → Verified groundedness → Finalized
```

- **Corrective retrieval loop** — grades whether retrieved sources are sufficient; if not, rewrites the query and retries; if still not, proposes ingesting a new paper from arXiv.
- **Human-in-the-loop ingestion** — out-of-corpus questions trigger an approval prompt; on *approve*, the agent downloads the PDF, chunks, embeds, and re-retrieves — then answers from the freshly-added paper. The pause/resume survives a server restart (durable checkpoint).
- **Two-layer groundedness gate** — a deterministic citation check (a cited paper not in the retrieved set is fabricated) *plus* an LLM grader that distinguishes fabricated specifics from reasonable synthesis. Ungrounded drafts are surgically rewritten or the least-fabricated draft is kept.
- **Three-tier memory** — in-session history, a rolling summary that bounds context growth, and cross-session episodic recall ("what did I ask last week?") that is RAG over your own conversation history.
- **Multi-tenant** — each API key gets its own corpus overlay and its own memory; one user's ingested papers and chat history are invisible to everyone else.

---

## Architecture

A LangGraph state machine where every node is an independently evaluable decision point.

```
                          ┌───────────┐
                  START → │ summarize │  (fold evicted turns into a rolling summary)
                          └─────┬─────┘
                                ▼
                          ┌─────────────┐
                          │ route_intent│  corpus? follow-up? memory recall?
                          └──┬───┬───┬──┘
            memory_recall ───┘   │   └─── followup
                   ▼             │ corpus          ▼
              ┌────────┐         ▼            ┌──────────────────┐
              │ recall │   ┌────────────┐     │answer_from_history│
              └───┬────┘   │ plan_query │     └─────────┬─────────┘
                  │        └─────┬──────┘               │
                  ▼              ▼                       ▼
                 END     ┌──────────────┐ ◄──────┐     END
                         │   retrieve   │        │ refine_query
                         └──────┬───────┘        │ (rewrite, retries left)
                                ▼                │
                       ┌─────────────────┐ ──────┘ insufficient
                       │ grade_relevance │ ──────┐ insufficient + retries exhausted
                       └────────┬────────┘       ▼
                       relevant │         ┌──────────────────┐  search arXiv,
                                │         │ propose_ingestion│  LLM-name the paper
                                ▼         └────────┬─────────┘
                         ┌──────────┐              ▼
                         │ generate │      « interrupt: approve ingestion? »
                         └────┬─────┘        approve │        │ deny
                              ▼                       ▼        ▼
                     ┌────────────────────┐      ┌────────┐   END
                     │ grade_groundedness │      │ ingest │ ──┐
                     └────────┬───────────┘      └────────┘   │ → retrieve (loop)
                  grounded    │ regenerate (capped)           │
                              ▼  ◄────────────────────────────┘
                         ┌─────────┐
                         │ respond │ → END   (keep-best draft + persist turn)
                         └─────────┘
```

Per-node evaluation is the differentiator: the relevance gate, the planner, the groundedness gate, the router, and the keep-best fallback each have their own labeled eval harness (`stage_3/eval/`), reported with an error-type split (dangerous vs. safe) rather than bare accuracy.

---

## Production engineering (deployed)

Built demo → production through four phases, each item verified locally **and** in the cloud:

| Area | What shipped |
|---|---|
| **Durable, session-aware state** | LangGraph checkpointer on **RDS Postgres**; `/ask` honors a client session id; `/resume` completes a parked approval — conversations survive both server restarts and cloud redeploys. |
| **Observability** | **LangSmith** tracing (env-var graph spans + wrapped SDK calls): per-node latency, token, and cost; thread-grouped traces; dev/prod project split. |
| **Cost governance** | Per-request token budget + circuit breaker that degrades to a best-effort answer; the in-state meter matches LangSmith's independent count to the token. |
| **Resilience** | SDK retry/timeout config + per-node failure semantics (graders fail open, generation fails honestly); provider outage → `503`, never a crash. |
| **Latency & streaming** | Server-Sent-Events progress stream (time-to-first-feedback 20 s → 0.03 s); parallelized multi-entity retrieval. |
| **Prompt caching** | Canonical shared prefix across the three heavy LLM nodes → **~30–70% cheaper per turn** (before/after A/B; cache reads verified byte-identical in prod). |
| **AuthN/Z** | API keys (SHA-256 at rest), thread ownership (closes a corpus-poisoning hole and session hijack), per-identity rate limits. |
| **Secrets** | All credentials in **AWS Secrets Manager**, injected via task-def `valueFrom`. |
| **Multi-tenancy** | Frozen shared base corpus + per-tenant overlay collections; tenant-scoped episodic memory (closes a cross-user leak). |
| **Prompt-injection hardening** | Untrusted retrieved text and history are fenced and sanitized (spotlighting + breakout-token stripping); the agent ignores instructions embedded in ingested papers. |

**Stack:** Python 3.13 · LangGraph · Anthropic Claude Sonnet 4.6 · OpenAI `text-embedding-3-small` · ChromaDB · FastAPI · slowapi · LangSmith · AWS (ECS Fargate "Express" + RDS Postgres + Secrets Manager + ECR) · Docker.

---

## How it was built — the methodology

The project's premise is that **the depth of the evaluation, not the pipeline, is the asset.** Four stages, built in isolation and validated before wiring together:

| Stage | Focus | Headline result (reported honestly) |
|---|---|---|
| **0** | LLM API fluency | Five primitives: chat, structured output, streaming, tool use, agent loop. |
| **1** | [Retrieval in isolation](stage_1/README.md) | Vector / +Cohere-rerank / +BM25-RRF compared on a 50-question hand-curated eval. **recall@5 = 0.96** (all three statistically tied at ±3.5% CI); **recall@10 = 0.98** for hybrid. A documented **null result**: reranking didn't beat vector — it made *different* errors. |
| **2** | [Agent + answer-quality eval](stage_2/README.md) | ReAct agent (Claude) with an LLM-as-judge (GPT-4.1, cross-family). The judge **saturated at 12/12 on all 30 questions** — reported as a **ceiling effect in the instrument, not a flawless agent**; a human spot-check (the binding signal) caught a groundedness overstatement the judge scored inconsistently. |
| **3** | [Corrective-RAG agent + productionization](stage_3/) | The LangGraph rewrite, the memory layer, and Phases 0–3 above. 22 experiments. |

Recurring lessons that show up across the log: a perfect score usually means a saturated metric, not a perfect system; forced-path tests catch the bugs happy-path demos hide (keep-best silently degenerating to keep-first; an empty answer passing the groundedness gate; an auth fence missing one of four endpoints); and a self-mutating corpus silently rots a static eval set.

---

## Run it locally

Requires Python 3.13 and [`uv`](https://github.com/astral-sh/uv). You'll need `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in a `.env` file.

```bash
uv sync

# Reproduce the vector index (see "Corpus" below for the PDFs), then:
uv run python stage_3/graph.py        # interactive REPL — watch the graph reason

# Or run the API + web UI:
uv run uvicorn api:app --app-dir stage_3 --port 8000
#   ADMIN_KEY=<choose-one> bootstraps an admin key; POST /admin/keys to mint more.
#   Without DATABASE_URL it uses local SQLite for checkpoints + auth; with it, Postgres.
```

Per-node evals (each is a standalone script):

```bash
uv run python stage_3/eval/eval_router.py            # intent classifier
uv run python stage_3/eval/eval_planner.py           # query decomposition + omission
uv run python stage_3/eval/eval_grounding_synthesis.py  # synthesis vs. fabrication
uv run python stage_3/eval/test_keep_best.py         # degenerate-draft fallback
uv run python stage_3/eval/eval_episodic.py          # memory routing + recall@k
```

---

## Repository layout

```
stage_0/   API-fluency primitives
stage_1/   RAG pipeline + retrieval eval (manifest → extract → chunk → embed → 3-way comparison)
stage_2/   ReAct agent + LLM-as-judge answer-quality eval + first AWS deploy
stage_3/   LangGraph Corrective-RAG agent (the production system)
  graph.py        the state machine: nodes, edges, corrective loop, caching, injection defense
  api.py          FastAPI: /ask, /ask/stream (SSE), /resume, /admin/keys, auth, rate limiting
  auth.py         API-key + thread-ownership store
  memory.py       short-term history + rolling summarization
  episodic.py     cross-session episodic memory (RAG over conversations)
  static/         single-file web UI (live reasoning timeline)
  eval/           per-node evaluation harnesses
  results/EXPERIMENTS.md   the full 22-experiment log
ROADMAP.md   the demo→production task list, checked off phase by phase
```

---

## Corpus reproduction

The 77 PDFs aren't committed (arXiv-redistributable, ~80 MB). The arXiv IDs are in `stage_1/data/manifest.json`; download each from `https://arxiv.org/pdf/<id>.pdf` into `paper_pdfs/<id>.pdf`, then run the Stage 1 pipeline (`extract.py` → `chunk.py` → `embed_and_index.py`) to rebuild the ChromaDB index.

---

## Status

Phases 0–3 complete and deployed. **Phase 4** (online eval + feedback loop, prompt/model versioning, a managed vector store for durable overlays, corpus lifecycle) is next — tracked in [`ROADMAP.md`](ROADMAP.md).
