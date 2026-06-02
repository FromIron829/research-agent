# Stage 3 — Architecture Sketch (planning, not yet built)

Re-architects the Stage 2 ReAct agent as a **graph-structured Corrective-RAG agent** with dynamic arXiv ingestion (human-in-the-loop), a memory layer, and explicit graded nodes — likely on **LangGraph** (see [[stage-3-future-work]] for the framework rationale). Each node is independently evaluable, reusing the Stage 2 judge.

## The graph

```
   user turn
      │
      ▼
┌──────────────┐   load in-session summary (short-term) +
│ recall_memory│   relevant past episodes (long-term, RAG over history)
└──────┬───────┘
       ▼
┌──────────────┐   classify: corpus Q? explicit "add paper"? memory recall? chit-chat?
│    route     │
└──┬────────┬──┘
   │        │ (corpus question)
   │        ▼
   │  ┌──────────────┐ ◄───────────────────────────┐
   │  │   retrieve   │                              │
   │  └──────┬───────┘                              │ refine_query
   │         ▼                                      │ (rewrite, retries left)
   │  ┌────────────────┐                            │
   │  │ grade_relevance│──── insufficient ──────────┘
   │  └──┬──────────┬──┘
   │     │ relevant │ insufficient & retries exhausted
   │     │          ▼
   │     │   ┌──────────────────┐   search arXiv for candidate(s)
   └────►│   │ propose_ingestion│   (explicit user "add X" routes here directly)
 (explicit └────────┬─────────┘
  add-paper)         ▼
                « interrupt: approve ingestion? »  ← skip if pre-authorized by user
                 approve │        │ deny
                         ▼        ▼ (answer with "not in corpus" caveat)
                  ┌──────────┐
                  │  ingest  │  arXiv API → pymupdf4llm → chunk → embed → upsert ChromaDB
                  └────┬─────┘
                       └────► retrieve (loop with the new paper available)
       │ relevant
       ▼
┌──────────────┐  cited synthesis, ===ANSWER=== sentinel (Stage 2 generation logic)
│   generate   │
└──────┬───────┘
       ▼
┌────────────────────┐  is every claim supported by a retrieved chunk?
│ grade_groundedness │
└──┬──────────────┬──┘
   │ grounded     │ ungrounded & retries → generate (or retrieve)
   ▼
┌──────────────┐  persist salient episode + facts to long-term store
│ write_memory │
└──────┬───────┘
       ▼
      END (respond)
```

## State (the graph's shared object)

| Field | Purpose |
|-------|---------|
| `messages` | conversation history (short-term) |
| `memory_context` | recalled long-term episodes/facts injected this turn |
| `question` | current user question (possibly rewritten) |
| `retrieved` | current retrieved chunks |
| `relevance_grade` | structured verdict from `grade_relevance` |
| `answer`, `groundedness_grade` | draft answer + its support verdict |
| `pending_paper` | arXiv candidate awaiting ingestion/approval |
| `retries` | per-stage caps (retrieve-refine, regenerate) |

## Nodes

- **recall_memory** — read short-term summary + semantic search over long-term episodic store; inject into context.
- **route** — classify the turn and pick the entry path (corpus / explicit-add / recall / chit-chat). Explicit "add paper X" = **pre-authorization** for ingestion (skips the approval interrupt).
- **retrieve** — Stage 1 hybrid search (vector + BM25 + RRF) over ChromaDB.
- **grade_relevance** — structured grader (LLM or reranker threshold): are the chunks sufficient to answer? Routes: relevant→generate · insufficient+retries→refine_query · insufficient+exhausted→propose_ingestion. *(This is the CRAG corrective trigger.)*
- **refine_query** — rewrite the query; loop to retrieve.
- **propose_ingestion** — search arXiv for the best candidate paper(s); build an ingestion plan.
- **« interrupt »** — human-in-the-loop approval gate before any write. Skipped when pre-authorized.
- **ingest** — arXiv API → pymupdf4llm extract → chunk → embed → **upsert** into ChromaDB; loop back to retrieve. (Idempotent on `arxiv_id`; handle v1/v2; arXiv-only for security.)
- **generate** — cited synthesis with the `===ANSWER===` sentinel (lifted from Stage 2).
- **grade_groundedness** — self-critique: is every claim supported by a retrieved chunk? Routes: grounded→write_memory · ungrounded+retries→regenerate. *(The deferred Stage 2 self-critique, now a node.)*
- **write_memory** — persist the salient episode + extracted facts to the long-term store.

## Memory layer

- **Short-term:** LangGraph **checkpointer** (thread-scoped state persistence across turns) + a rolling **summarization** node when history grows.
- **Long-term:** a separate ChromaDB collection = **RAG over conversation history**. Episodic (past turns w/ timestamp + embedding → "what paper did I ask about days ago") + semantic/profile facts. `recall_memory` reads; `write_memory` writes; bound growth via dedup/summarize/decay.

## Eval plan (the differentiator)

Each node is independently testable — reuse the Stage 2 judge muscle:
- `grade_relevance` — labeled set of (query, chunks) → correct sufficient/insufficient calls (precision/recall of the corrective trigger).
- `grade_groundedness` — answers with seeded unsupported claims → does it catch them?
- `ingest` — given an out-of-corpus question, is the *right* paper fetched?
- `generate` — the existing 30-question answer-quality eval.
- **End-to-end** — task success on questions that *require* ingestion or memory recall.

## Deployment implications (affects Stage 2 vs Stage 3 ordering)

- **Stage 2 is read-only & stateless** → vector index can be baked into the image or a read-only volume. Simpler to deploy.
- **Stage 3 writes** (ingestion mutates the index) and is **stateful** (memory, checkpoints) → needs a **writable, durable, possibly shared** vector store (hosted Qdrant/Weaviate/pgvector, or a persistent volume with concurrency care) + a checkpoint store (Postgres/SQLite) + async/long-running handling for ingestion + a UI/async channel for the approval interrupt.
- Implication: the read-only deployment learned on Stage 2 is the foundation; Stage 3 forces the writable/stateful upgrade. Natural difficulty ramp, not wasted work.

## Open questions to resolve before building

- Relevance grading: LLM grader vs. reranker-score threshold (cost/latency vs. accuracy)?
- Approval UX in a deployed setting (the `interrupt` needs an async channel — websocket/polling).
- Long-term memory write policy: every turn vs. agent-decided salience?
- Concurrency on the shared index when ingestion writes during others' reads.
