# Stage 3 — LangGraph Corrective-RAG Agent: Implementation Log

## Setup

- **LangGraph** for graph-structured orchestration — chosen over a hand-rolled `while` loop because explicit nodes/edges are observable, debuggable, and let each node be tested in isolation.
- **Reused Stage 1 primitives inside nodes** (so ingestion mirrors the original build exactly): `retrieve_hybrid` (vector + BM25 + RRF), `strip_references`, the chunker (`RecursiveCharacterTextSplitter`, 800/100 tokens, `cl100k_base`), `text-embedding-3-small`, and the live `papers` Chroma collection.

## Design — a graded Corrective-RAG loop

- **Nodes** are plain functions: `State -> partial state update`. LangGraph merges the returned keys into the shared state.
- **Edges** wire the control flow. **Conditional edges** branch on the state — a routing function inspects the state and returns which node runs next. This is what turns a linear pipeline into a decision-making agent.

The graph (production patterns: **Self-RAG / Corrective RAG / Reflexion**):

```
retrieve → grade_relevance → (refine ↺ retrieve | generate | propose_ingestion)
generate → grade_groundedness → (regenerate ↺ generate | respond)
propose_ingestion → approval (interrupt) → (ingest ↺ retrieve | deny)
```

- **Relevance gate** — grade the retrieved chunks; if weak, refine the query and retry (capped); if still weak after refinement, take the corrective branch.
- **Groundedness gate** — after generating, check every claim against the sources; regenerate with the critique if unsupported (capped).
- **Corrective ingestion** — when the corpus can't answer, search arXiv, ask the human, ingest the approved paper, loop back to retrieve.

## Challenges (and how I solved them)

1. **A self-critique loop can reward-hack its own grader.** My first version told the LLM to "rewrite so everything is grounded." I realized that invites two disasters: it can **fabricate citations** to pass the check, or **delete correct-but-hard-to-verify content** — both worse than the original answer. Three fixes:
   - **Constrained rewrite**: per flagged claim, *either* add a citation that's actually in the sources *or* remove just that claim — never invent a citation, never touch other claims.
   - **Deterministic citation check** (`verify_citations`, ported from my Stage 2 judge): parse every `[Paper (page N)]` and confirm the cited paper is among the retrieved chunks. A fabricated citation doesn't resolve → caught by *code*, which the LLM can't fool. This is the real defense against fabrication.
   - **Keep-best-not-last**: if the regenerate loop hits its cap still ungrounded, return the *original* draft, not the degraded rewrite — so the loop can only help, never hurt.

2. **The groundedness grader was over-firing** — flagging supported claims as unsupported, triggering pointless (and destructive) rewrites. Root cause: I was grading against **truncated** chunk text, so the grader literally couldn't see the support. Fixes: grade against the **full** source text the generator used, and require the grader to **quote the exact unsupported sentence** — if it can't quote one, treat the answer as grounded.

3. **arXiv search quality.** Passing the user's raw question (e.g., "How does Adam work?") returned irrelevant recent papers, not the foundational one. Fix: have the LLM rewrite the question into a focused paper-finding query (key terms / likely title).

4. **arXiv is flaky and rate-limited** — hit HTTP 301 redirects, then 429s, then read-timeouts in sequence. Lessons:
   - Replaced the `arxiv` library with a direct `requests` call (follows redirects, full control) + a descriptive `User-Agent`.
   - **Resilience**: wrap every external call in a node with try/except + retry/backoff and **graceful degradation** — a node must never crash the whole graph because an API hiccupped.
   - **Staying under the limit (proactive)** — this is the part I first got wrong: a **disk cache** (never re-request a query I've already searched) and a **self-throttle** (enforce ≥3s between *my own* requests so I never burst past arXiv's rate). These *prevent* the 429.
   - **Handling the limit (reactive)** — when a 429 still happens, honor the `Retry-After` header and back off before retrying. *(I'd initially conflated this with the throttle — they're different: the throttle stops me earning the 429; the backoff handles it when it slips through.)*

5. **A write-capable agent needs a human gate.** Ingestion mutates the knowledge base, so it can't happen silently. Used LangGraph's `interrupt()` — the graph **pauses** mid-run, surfaces the candidate paper, and only proceeds on my approval (resumed via `Command(resume=...)`). This requires a **checkpointer** (`MemorySaver`) plus a `thread_id` so the paused state can be persisted and restored.

6. **Pipeline drift** (caught late). My ingest skipped the `strip_references` cleaning that Stage 1 applied, so newly-added papers carried reference-list noise — making the index heterogeneous (clean old papers, noisy new ones). Lesson: the ingestion path must **exactly mirror** the original corpus-build pipeline; ideally they share the same function. Fixed by reusing `strip_references`.

7. **Termination guards.** A graph can cycle, so every loop needs a cap: `MAX_ATTEMPTS` (refine), `MAX_GEN` (regenerate), and an `ingested` flag so the agent can't loop propose → ingest → propose forever. Also a conditional edge after `propose_ingestion` so a failed/empty arXiv lookup ends gracefully instead of crashing into the approval node (a missing-`candidate` `KeyError`).

8. **Anthropic `tool_use` doesn't enforce `required` schema fields** (unlike OpenAI strict outputs) — the model dropped a required `issues` field. Lesson: read structured tool output defensively with `.get(default)`, and default grades to "pass" so a missing grade doesn't trap the loop.

## Outcome

- **In-corpus questions** — the agent answers from the 77-paper corpus with cited, groundedness-checked answers. *(Answer quality not yet formally evaluated — see next steps.)*
- **Out-of-corpus questions** — the agent detects the knowledge gap, proposes the right arXiv paper, pauses for my approval, ingests it (download → strip references → chunk → embed → upsert), loops back, and answers from the freshly-acquired paper. **Verified end-to-end** on "How does Adam work?": ingested `1412.6980`, then produced a correct, fully-cited explanation of Adam — sourced from a paper that wasn't in the corpus 60 seconds earlier.

## Known limitations / next steps

- **BM25 is stale.** Ingested papers are added to the vector store only; the BM25 index is built once at import, so new papers are found via vector similarity (+ RRF) but get no lexical boost. Proper fix: persist new chunks and rebuild BM25, or use a dynamic index.
- **arXiv search is currently behind a temporary stub** (Adam paper) while rate-limited — restore the real LLM-query + `search_arxiv` call.
- **Single-paper, synchronous ingestion** (~30-60s, blocks the request). Production design: decouple to a background queue processed at arXiv's allowed rate.
- **Per-node quality not yet evaluated.** The relevance gate, groundedness gate, and ingestion each deserve their own eval (reusing the Stage 2 LLM-as-judge) — e.g., does the relevance gate fire on out-of-corpus questions and *not* on in-corpus ones? This is the planned rigor capstone (Stage 3.5).
