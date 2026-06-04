# Stage 3 — Experiment Log

The Stage 3 agent is a LangGraph Corrective-RAG graph (build narrative in `stage_3/LOG.md`, code in `stage_3/graph.py`). Stages 1-2 measured **retrieval recall** and **end-to-end answer quality**; Stage 3's novel surface is its **decision nodes** — the two graders that drive the corrective loop. This log evaluates them.

---

## Experiment 1 — Per-node evaluation of the CRAG decision gates

**Date:** 2026-06-04
**Harness:** `stage_3/eval/eval_relevance.py`, `stage_3/eval/eval_groundedness.py`

### Setup

- **Method: per-node, in isolation.** Each gate is tested by calling its node function directly on controlled inputs and checking the output against a label — no full graph run, no live arXiv. This isolates the unit under test and is reproducible (a live, rate-limited external API is the wrong eval dependency; external calls are mocked/fixtured).
- **Why these two nodes:** `retrieve` was evaluated in Stage 1 (recall@k) and `generate` answer-quality in Stage 2 (LLM-as-judge). The **relevance** and **groundedness** gates are new to Stage 3 and are the decisions that drive the corrective loop, so they are what needs measuring.
- **`grade_relevance`:** 13 questions — 7 in-corpus (answerable from the 77 papers) and 6 out-of-corpus (Mamba, CNNs, BERT MLM, the original Transformer, diffusion models, batch norm). For each: `retrieve_hybrid(q)` → `grade_relevance_node` → compare verdict to label.
- **`grade_groundedness`:** 5 hand-crafted (answer, fixture-chunks) cases — 2 grounded, 3 ungrounded — designed to target the gate's **two distinct defenses separately**: a fabricated citation (paper not retrieved), an unsupported claim with a *valid* citation, and a fully fabricated paper + claim.

### Hypothesis

- **Relevance gate:** it correctly separates in- vs out-of-corpus, and any errors are biased toward **over-triggering** (sufficient retrieval flagged insufficient → safe: extra refine/ingest) rather than **missing gaps** (junk flagged sufficient → dangerous: answer with no real sources). A corrective gate should fail safe.
- **Groundedness gate:** both defenses fire on their own failure mode — the **deterministic** `verify_citations` floor catches fabricated citations regardless of the LLM, and the **LLM semantic grader** catches unsupported-but-validly-cited claims; clean answers pass with no false positives.

### Result

**Relevance gate — 12/13 (92%)**

| Class | n | Correct |
|-------|---|---------|
| In-corpus → sufficient | 7 | 6 |
| Out-of-corpus → insufficient | 6 | 6 |

- **Missed gaps (out-of-corpus graded sufficient): 0.** The dangerous error is zero.
- **Over-triggers (in-corpus graded insufficient): 1 — GPTQ.**

**Groundedness gate — 5/5** (ungrounded caught 3/3, false positives 0/2)

| Case | Defense responsible | Outcome |
|------|--------------------|---------|
| Clean answer ×2 | — (should pass) | ✅ grounded, no false positive |
| Fabricated citation `[GPTQ]` (not retrieved) | deterministic `verify_citations` | ✅ `[citation-check]` flagged it |
| Unsupported claim, *valid* citation ("100× speedup on every GPU") | LLM semantic grader | ✅ grader quoted the claim; no citation-check |
| Fabricated paper + claim `[MadeUpPaper]` | deterministic floor | ✅ flagged |

### Interpretation

1. **The relevance gate fails safe — as hypothesized.** 0 missed gaps: it never let an out-of-corpus question answer from irrelevant chunks (the hallucination failure the corrective branch exists to prevent). Its lone error is the safe kind.

2. **The one "over-trigger" is a *retrieval* gap, not a grader error.** GPTQ *is* in the corpus, but the grader's stated reason was that the retrieved chunks held GPTQ's intro/results/comparisons but **not its method section** (Sec. 3-4) — so those chunks genuinely cannot answer a "how does it work" question, and grading them insufficient was correct. This exposes a subtlety in the labels: they encode **corpus membership**, but the gate's real job is **"are *these retrieved chunks* sufficient?"** — and the two diverge exactly when retrieval misses. Reframed, the gate is correct on all 13; GPTQ is a *retrieval* finding. It also surfaces the **chunk-level** retrieval gap Stage 1 could not measure (its gold was paper-level; `gold_chunk_ids` was empty). In the full loop this routes to `refine_query` — the correct corrective action for "right paper, wrong chunks" — not to a wasteful ingestion.

3. **Both groundedness defenses are independently verified.** The fabricated-citation cases were caught deterministically (a fake citation can't be argued past); the unsupported-but-cited claim was caught *only* by the LLM grader (no citation-check line fired). Splitting the cases by responsible defense — rather than reporting one combined number — is what makes the result informative: a failure would tell us exactly *which* layer broke.

### Decision

- **Both decision gates are validated** for the current corpus/fixtures.
- **Follow-ups:** (a) investigate the GPTQ chunk-level retrieval gap (why does top-10 surface intro/results over the method section for "how" questions?) — a *retriever* improvement, distinct from the gate; (b) expand both eval sets (n is small; groundedness fixtures are hand-crafted) before treating the percentages as precise; (c) relabel the relevance set as "retrieval-sufficient" rather than "corpus-member" to avoid the GPTQ-style proxy mismatch.
- **Methodology note:** per-node isolation (call the node on controlled inputs) is the right eval pattern for a graph agent — it tests each decision independently and reproducibly, and reuses the Stage 2 judge discipline (structured verdicts, honest error-type splitting rather than a single headline number).

### Known limitations

- Small n (13 relevance, 5 groundedness) — directional, not precise.
- Groundedness fixtures are hand-authored, not sampled from real agent outputs.
- Relevance labels use corpus-membership as a proxy for retrieval-sufficiency (the GPTQ case shows where that breaks).
- The `ingest`/`search` nodes are exercised via the end-to-end run (Stage 3.3, verified on the Adam paper) and fixtures, not a labeled eval; post-ingestion answer quality is not separately scored.
