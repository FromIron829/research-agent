# Production Readiness Roadmap

Canonical task list for taking the Stage 3 Corrective-RAG agent from **demo-grade** to
**production-grade**. We work this list top-to-bottom; phases are ordered by leverage and
dependency. Check items off as they land.

**How we work it:**
- Each feature follows the project's experiment discipline: **Setup → Hypothesis → Result →
  Interpretation → Decision**, logged in `stage_3/results/EXPERIMENTS.md`. Hypothesis committed
  *before* the result.
- Each decision node / classifier gets a **per-node eval** (the Exp 1 pattern) before it's
  considered done — not just a happy-path demo.
- Build in isolation, verify, then wire in. Commit when stable.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Phase 0 — Close out the memory layer (current feature) ✅ COMPLETE

The short-term memory + plan-and-execute reasoning works (Exp 2). Finish the feature with the
rigor that is the project's differentiator, then add long-term recall.
**Status: all six items done (Exp 3-8). Three memory tiers — short-term history, in-session
summary, cross-session episodic — built and per-node validated.**

- [x] **0.1 Eval the intent router** as a classifier — labeled set incl. new-entity follow-ups
  (the failure that started Exp 2: corpus vs followup, later vs memory_recall). Report error-type
  split, not just accuracy. **Done (Exp 3):** `eval_router.py`, n=17 + 2 taxonomy-gap; found 1
  dangerous miss on same-entity *deepening*, fixed with an information-sufficiency tiebreaker →
  DANGEROUS=0, error relocated to the safe column. Taxonomy gap (meta/chitchat) deferred to 0.6.
- [x] **0.2 Eval the planner** (`plan_query_node`) — does it produce the right `sub_queries`, and
  does it correctly *omit* topics already covered in history? Decomposition correctness + omission
  precision. **Done (Exp 4):** `eval_planner.py`, entity-coverage metric (should_fetch=DANGEROUS /
  should_omit=WASTEFUL), n=9 + 1 gap; 9/9 clean, 0 dangerous, over-omission traps STABLE-PASS ×4.
  Planner is sufficiency-aware by construction; "fetch-nothing when history fully covers" left
  untested (gap fixture too thin).
- [x] **0.3 Comparison-grounding eval** — verify synthesis survives while fabricated specifics are
  caught (quantify the synthesis-vs-fabrication boundary from Exp 2; does it ever wave through a
  real fabrication framed as "synthesis"?). **Done (Exp 5):** `eval_grounding_synthesis.py`, 5 cases
  ×N=3; 0 false positives (synthesis spared), 0 false negatives (fabrications caught), stable 15/15.
- [x] **0.4 Adversarial keep-best test** — a deliberately twice-failing groundedness case to
  exercise the untested `best_answer`/`best_n_issues` fallback path. **Done (Exp 6):** `test_keep_best.py`
  found a real bug — keep-best silently degenerated to keep-FIRST (issue count was always 1).
  Fixed with a structured `n_fabrications` count in GROUND_TOOL; re-verified 6/6.
- [x] **0.5 History summarization** — bound growth (`MAX_TURNS=6` currently just truncates);
  rolling summary/compaction so long sessions don't lose early context or blow the token budget.
  **Done (Exp 7):** `summarize_node` at graph entry + `summary`/`n_summarized` state; incremental
  fold (no re-summarize), first turns free, end-to-end verified. (Storage still unbounded — context
  only; summary *fidelity* eval deferred.)
- [x] **0.6 Long-term episodic memory** — conversations Chroma collection (past turns +
  timestamp + embedding) + a `memory_recall` intent for cross-session "what did I ask last week."
  Decisions: when-to-write, read = recency + relevance, bound growth (dedup/decay).
  **Done (Exp 8):** `episodic.py` + `recall_node` + write path; `eval_episodic.py` caught two real
  bugs (L2-vs-cosine metric, silent same-second id collision) + an embed A/B (Q+A → recall@1 2/5→5/5).
  Routing 7/8 (0 hijacks). Cross-session recall verified. Deferred: recency filter, per-user
  isolation, growth bounds, followup/memory_recall tiebreaker.

## Phase 1 — Make it real: durable, session-aware state (Tier 1) ✅ COMPLETE

The highest-leverage gap. Today `MemorySaver` is in-memory and `/ask` mints a throwaway thread
id, so the memory layer **does not work deployed**.
**Status: all four items done (Exp 9-12). Durable × session-aware × resumable × deployed —
conversations survive server restarts AND cloud redeploys.**

- [x] **1.1 Durable checkpointer** — replace `MemorySaver` with a persisted backend
  (LangGraph Postgres/Redis checkpointer); state survives restart/redeploy/crash.
  **Done (Exp 9):** `SqliteSaver` (`checkpoints.db`, `check_same_thread=False` for FastAPI threads);
  kill-restart test passed — follow-up in a new process answered from the prior session's history.
  Postgres deferred to 1.4 (deploy); WAL sidecars gitignored via `checkpoints.db*`.
- [x] **1.2 Session-aware API** — `/ask` honors a client-supplied (authenticated) session id
  instead of overwriting it with a fresh uuid; conversations persist across requests.
  **Done (Exp 10):** honor-or-mint thread_id (default None, returned on both branches) +
  `fresh_turn()` per-turn reset (latent stale-state bug activated by thread reuse). 4-step live
  test passed incl. conversation surviving a server restart. Auth deferred to Phase 3.
- [x] **1.3 Complete the approval flow** — add the missing `/resume` endpoint; verify
  interrupt → approve → resume works against the durable checkpointer.
  **Done (Exp 11):** `/resume` with a `get_state(.).next` guard (409 on no-pending/nonexistent
  threads). Headline test passed: approval parked → server killed + restarted → resumed in the
  new process. Decline-path tested (approve→ingest verified earlier); /resume auth → Phase 3.
- [x] **1.4 Stateful infra** — provision the state store in AWS; confirm the single-task
  constraint can be relaxed once state is external.
  **Done (Exp 12):** RDS Postgres 16 (db.t4g.micro, Single-AZ, ~$14/mo — caught a $132/mo
  template default), SG-to-SG networking, env-driven `_make_checkpointer` (DATABASE_URL →
  PostgresSaver, else SqliteSaver), immutable tag stage3-v4, task-def rev 9. Cloud kill-test
  passed: conversation survived a forced redeploy. Single-task constraint NOT relaxed —
  Chroma/BM25 still on container disk (unblocks at 4.3). Secrets → Phase 3; rate-limit gap → 2.2.

## Phase 2 — Operate it: observability, cost, resilience, latency (Tier 2) ✅ COMPLETE

**Status: all five items done (Exp 13-17). Rate-limited, traced, budget-capped, outage-resilient,
progress-streaming. Deployed through stage3-v8.**

- [x] **2.0 Hotfix: restore rate limiting** (regression found in Exp 12 — stage-3 api.py never
  carried over Stage 2's slowapi). **Done (Exp 13):** XFF-keyed 5/min;50/day on `/ask` AND
  `/resume` (ingest = most expensive route); verified 429 locally and on the deployed service
  (stage3-v5, rev 10). Per-user quotas need Phase 3 auth; external counter store needed at 4.3.
- [x] **2.1 Observability** — replace `print()` with structured logging + tracing
  (LangSmith or OpenTelemetry); per-node latency / token / cost capture; request correlation ids.
  **Done (Exp 14):** LangSmith — env-var graph tracing + `wrap_anthropic(client)` for the raw SDK
  calls; dev/prod project split; verified locally (per-node tree, token splits, thread grouping)
  and on Fargate via the API (full corpus trace: 18.6s / 27.8K tokens / $0.0907). First measured
  cost baselines: ~$0.01 followup vs ~$0.09 corpus turn. New deploy lesson: draining old task
  served the first verification request (RUNNING ≠ gateway switched). Embeddings unwrapped;
  prints kept as log breadcrumbs.
- [x] **2.2 Cost governance** — per-request token/cost budget + circuit breaker (one query fans
  out to 6–10 LLM calls today); surface cost per request in traces.
  **Done (Exp 15):** `tokens_used` meter in state (plain int, NOT a reducer — reset trap), 8 nodes
  metered, breaker in both routers (over budget → generate-with-what-we-have / respond-with-best,
  reusing the 0.4 keep-best degrade). Budget 60K ≈ happy path + 1 refine + 1 regen. 10/10 checks;
  meter matches LangSmith **to the token** (27,651 = 27,651). Worst case ~$0.35/request. Deploy
  batched with 2.3. Per-user budgets need Phase 3 auth.
- [x] **2.3 LLM-call resilience** — retry-with-backoff on Anthropic/OpenAI calls, timeout
  handling, fallback model; a provider 429/529 must not crash the graph mid-run.
  **Done (Exp 16):** SDK-level retry config (timeout 600s→60s, max_retries 4) + per-node failure
  semantics (graders fail open, router→corpus, planner→passthrough; groundedness keeps the
  deterministic citation floor during outages; generate-class raises honestly) + `APIError`→503
  at the API boundary. Forced-outage suite 10/10 — caught 2 real outage-only bugs before ship.
  Deferred: cross-request circuit breaker, per-subclass (401 vs 529) handling, fallback model.
  **2.2+2.3 deployed together as stage3-v7 (task-def rev 12); prod meter verified against the
  prod LangSmith trace to the token (25,308 = 25,308).**
- [x] **2.4 Streaming + latency** — stream the final answer to the user (restore Stage 2
  streaming); parallelize independent work (sub-query retrieval, where safe).
  **Done (Exp 17):** progress-not-tokens design (groundedness gate is anti-token-streaming by
  construction — stream what the agent is DOING, ship only the verified answer). SSE `/ask/stream`
  via `graph.stream(stream_mode="updates")`: time-to-first-feedback 20.28s → 0.03s, interrupt
  traverses the stream as `approval_needed`. Parallel sub-query retrieval: identical chunks,
  1.70s → 0.43s. The stream narrates the CRAG loop live (demo artifact).

## Phase 3 — Secure & multi-user (Tier 3)

- [ ] **3.1 AuthN/Z** — user identity + auth on endpoints; rate-limit per user, not just per IP.
- [ ] **3.2 Multi-tenancy & corpus isolation** — ingestion currently writes to ONE global
  corpus; isolate per-user/tenant so one user's ingested paper can't pollute another's retrieval.
- [ ] **3.3 Injection & poisoning defenses** — treat retrieved chunks and conversation history as
  untrusted input (they're interpolated into prompts today); sanitize/segregate; validate ingested
  content beyond the arXiv-id/title check.

## Phase 4 — Quality lifecycle & scale (Tier 4)

- [ ] **4.1 Online eval + feedback loop** — capture user feedback (👍/👎), sample live traffic for
  review, monitor answer quality in prod; eval-gated CI that blocks a deploy on regression.
- [ ] **4.2 Prompt/model versioning** — persist `prompt_version` + model id per response
  (flagged unresolved since Stage 2); A/B framework so prompt changes are provably non-regressive.
- [ ] **4.3 Retrieval at scale** — managed vector store (pgvector/Pinecone/Weaviate) + incremental
  sparse index; today BM25 does a full O(n) rebuild on every ingest.
- [ ] **4.4 Corpus lifecycle** — paper version handling (v1→v2), dedup at scale, size bounds /
  eviction.

---

**Current focus:** Phase 0. Highest single-fix leverage once Phase 0 closes: **1.1–1.3**
(durable + session-aware state) — that's what makes the memory layer actually usable in production.
