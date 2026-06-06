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
- **Corrective ingestion** — when the corpus can't answer, find the right paper, ask the human, ingest the approved paper, loop back to retrieve.

## Challenges (and how I solved them)

1. **A self-critique loop can reward-hack its own grader.** My first version told the LLM to "rewrite so everything is grounded." I realized that invites two disasters: it can **fabricate citations** to pass the check, or **delete correct-but-hard-to-verify content** — both worse than the original answer. Three fixes:
   - **Constrained rewrite**: per flagged claim, *either* add a citation that's actually in the sources *or* remove just that claim — never invent a citation, never touch other claims.
   - **Deterministic citation check** (`verify_citations`, ported from my Stage 2 judge): parse every `[Paper (page N)]` and confirm the cited paper is among the retrieved chunks. A fabricated citation doesn't resolve → caught by *code*, which the LLM can't fool. This is the real defense against fabrication.
   - **Keep-best-not-last**: if the regenerate loop hits its cap still ungrounded, return the *original* draft, not the degraded rewrite — so the loop can only help, never hurt.

2. **The groundedness grader was over-firing** — flagging supported claims as unsupported, triggering pointless (and destructive) rewrites. Root cause: I was grading against **truncated** chunk text, so the grader literally couldn't see the support. Fixes: grade against the **full** source text the generator used, and require the grader to **quote the exact unsupported sentence** — if it can't quote one, treat the answer as grounded.

3. **Finding the right paper is harder than it looks — and the LLM is more reliable than search.** Passing the raw question to arXiv search returned irrelevant recent papers. My first fix: have the LLM rewrite the question into a focused query, fetch the top-5, and re-rank them with the LLM. This *still* failed — arXiv's `all:` relevance search frequently doesn't surface the canonical paper at all. Searching "Attention Is All You Need" returns papers that *cite* it (its title appears in their related-work sections), not the original; the re-rank then picks the closest-but-wrong candidate. The real fix was to **flip the priority**: the out-of-corpus gaps that trigger ingestion are almost always *foundational* papers (Transformer, Adam, BERT), and the LLM knows their exact arXiv IDs more reliably than keyword search surfaces them. So **LLM proposal from parametric knowledge became the primary source**, with arXiv search as a *secondary* fallback for long-tail papers. A data-driven architecture change — made after watching the search repeatedly fail.

   Because the candidate's ID now comes from the LLM (and an arXiv ID is the single most error-prone token to ask a model for), I added **four defense-in-depth guards before any write** to the knowledge base: (1) the LLM can **abstain** (`known=false`) instead of fabricating; (2) **human approval** via `interrupt`; (3) a nonexistent ID **fails the PDF download** (404); (4) a valid-but-wrong ID downloads a *different* paper, caught by a **page-1 title check** against the proposed title. Only a paper that clears all four gets upserted.

4. **arXiv is two services, and only one is the problem.** I burned real time on what looked like one bug — 301s, then 429s, then read-timeouts in sequence — before isolating it:
   - **Two endpoints.** The *search* API (`export.arxiv.org/api/query`) is aggressively rate-limited; the *PDF download* (`arxiv.org/pdf/<id>`) is a different host and is reliable. Ingestion only needs the download given an ID — so making the LLM propose the ID (above) avoids the rate-limited endpoint entirely.
   - **The 429/timeout was per-IP/network, not my code.** Confirmed by elimination: `curl` to the search API timed out on my wifi but returned 200 on a phone hotspot; the deployed AWS server hit timeouts/429s only *under test load*, never a code error. arXiv's limit is per-IP, escalates, and applies across machines I control — so switching networks doesn't fix it.
   - **Client-side hardening:** a disk **cache** (never re-request a query), a **persistent file-based throttle** (≥3.5s between requests, surviving restarts — an in-memory one resets every run and immediately re-violates), exponential **backoff** honoring `Retry-After`, and **graceful degradation** (a failed search returns `[]` and funnels to the fallback, never crashing the graph).
   - **The request-path lesson:** I first gave the search a *patient* backoff (up to ~4 min). In the deployed service that blocked the HTTP request past the load balancer's timeout → **504**, and clogged the single worker. Fix: in the request path, **fail fast** (short timeout, degrade in seconds); the patient backoff belongs in a *background job*, not a live request.

5. **A write-capable agent needs a human gate.** Ingestion mutates the knowledge base, so it can't happen silently. Used LangGraph's `interrupt()` — the graph **pauses** mid-run, surfaces the candidate paper, and only proceeds on my approval (resumed via `Command(resume=...)`). This requires a **checkpointer** (`MemorySaver`) plus a `thread_id` so the paused state can be persisted and restored.

6. **Pipeline drift** (caught late). My ingest skipped the `strip_references` cleaning that Stage 1 applied, so newly-added papers carried reference-list noise — making the index heterogeneous (clean old papers, noisy new ones). Lesson: the ingestion path must **exactly mirror** the original corpus-build pipeline; ideally they share the same function. Fixed by reusing `strip_references`.

7. **Termination guards.** A graph can cycle, so every loop needs a cap: `MAX_ATTEMPTS` (refine), `MAX_GEN` (regenerate), and an `ingested` flag so the agent can't loop propose → ingest → propose forever. Also a conditional edge after `propose_ingestion` so a failed/empty arXiv lookup ends gracefully instead of crashing into the approval node (a missing-`candidate` `KeyError`).

8. **Anthropic `tool_use` doesn't enforce `required` schema fields** (unlike OpenAI strict outputs) — the model dropped a required `issues` field. Lesson: read structured tool output defensively with `.get(default)`, and default grades to "pass" so a missing grade doesn't trap the loop.

9. **A paper I just ingested couldn't be retrieved — "dynamic ingestion" is a keep-two-indexes-in-sync problem.** After ingesting the Transformer paper, the next retrieve *still* couldn't find it. Cause: hybrid retrieval has two indexes and ingestion updated only one. The **vector** half (ChromaDB) was live and saw the new paper; the **BM25** half was a snapshot built once at import from `chunks.json` and never updated. In RRF, an existing chunk earns points from *both* lists while a freshly-ingested chunk earns from *only* the vector list — a structural penalty that buries it, worst of all for a query whose topic saturates the corpus. Two fixes:
   - **Targeted boost** — after ingesting paper X to answer a question, retrieve from X directly via a metadata-filtered vector query (`where={"arxiv_id": X}`), guaranteeing it's in context for the loop.
   - **Dynamic BM25** — on ingest, append the new chunks to `chunks.json` *and* rebuild the BM25 index (idempotent by `chunk_id`, which must equal the Chroma ID so RRF fuses correctly), so *future-session* queries find it too.

10. **Every failure mode of an external call must funnel to one graceful path.** `propose_ingestion` had several early `return`s for the ways arXiv could fail (exception, empty results, re-rank found nothing). I fixed them one at a time — and each fix just exposed the next dead-end, because each was a separate exit that skipped the LLM fallback. The structural fix: compute a candidate, and if it's `None` for *any* reason, fall through to *one* fallback with *one* terminal give-up. Lesson: when the same bug keeps reappearing in different spots, the spots aren't the problem — the control flow has too many exits. Collapse N early-returns into one decision point and the whole class is fixed at once.

11. **Deploying a stateful, write-capable agent.** Stage 2 was stateless and read-only; Stage 3 has a checkpointer (for `interrupt`) and mutates the vector store. The HTTP layer needs two endpoints — `/ask` (returns the approval prompt when the graph pauses) and `/resume` (continues the same run via `Command(resume=...)`), matched by `thread_id`. A subtle bug: I reused a fixed `thread_id` across test requests, so overlapping requests (one still running server-side after its client 504'd) collided in the checkpointer and corrupted each other's state — an in-corpus question even inherited another request's refined query. Fix: generate a unique `thread_id` per request. Deeper takeaway: a synchronous external fetch in the request path is an anti-pattern (it caused both the 504s and a corpus-uptime dependency); the production design decouples ingestion to an **async background worker** with `/ask` returning immediately.

## Per-node evaluation

I evaluated the two novel decision nodes in isolation (full write-up in `results/EXPERIMENTS.md`):
- **Relevance gate** — 12/13, **0 dangerous missed-gaps**. Its one "error" (GPTQ graded insufficient) was a *retrieval* gap it correctly flagged — the right paper was in the corpus but the retrieved chunks were its intro/results, not its method section — not a grader miss.
- **Groundedness gate** — 5/5, with the **deterministic citation floor** and the **LLM semantic grader** each verified independently on its own failure mode (fabricated citation vs. validly-cited-but-unsupported claim).

Per-node isolation (call each node function on controlled inputs, no live arXiv) is the right eval pattern for a graph agent: reproducible, and it tells you *which* node failed rather than just that the end-to-end answer was wrong.

## Outcome

- **In-corpus questions** — answered from the corpus with cited, groundedness-checked answers (answer quality evaluated separately in Stage 2's judge framework).
- **Out-of-corpus questions** — the agent detects the gap, **proposes the right paper** (LLM-first, e.g. *Attention Is All You Need* → `1706.03762`), pauses for approval, **verifies and ingests** it (download → page-1 title check → `strip_references` → chunk → embed → upsert into *both* the vector and BM25 indexes), boosts it into the next retrieval, and answers from the freshly-acquired paper — grounded and cited. Verified end-to-end on *Attention Is All You Need* and *Adam*.
  - *Honest note:* an earlier "verified end-to-end" run used a **hardcoded candidate** to exercise the ingestion pipeline while the search endpoint was rate-limited — the pipeline (download→clean→chunk→embed→upsert) was real, but the *discovery* step was stubbed. Discovery is now the real LLM-first proposal.
- **Deployed** on AWS ECS Express (`api.py` → Docker → ECR → ECS, single task): in-corpus Q&A is reliable; out-of-corpus degrades gracefully when arXiv is unavailable.

## Known limitations / next steps

- **Discovery depends on the model knowing the paper.** LLM-first is reliable for foundational papers (and their exact IDs); long-tail papers fall back to arXiv keyword search, which is flaky. The page-1 title check is a heuristic, not authoritative verification.
- **Synchronous, single-paper ingestion** (~30-60s, blocks the request). Production design: decouple to an **async background worker** processed at arXiv's allowed rate, with `/ask` returning immediately — this also removes the request-path timeout/504 risk entirely.
- **Deployed storage is ephemeral.** The in-container Chroma upserts and in-memory checkpointer don't survive a container restart. Production needs a **persistent vector store** + a **checkpoint DB** (e.g. SqliteSaver/Postgres) + a writable volume.
- **Dynamic BM25 rebuilds the whole index on each ingest.** Cheap at this scale (a few thousand chunks); for a much larger corpus, switch to an incremental search index (Tantivy / `bm25s` / Elasticsearch).
- **No interrupt-aware frontend yet.** The two-step `/ask` → `/resume` approval flow is exercised via `curl`; a small UI that surfaces the approval prompt is the remaining polish.
