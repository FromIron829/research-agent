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

---

## Experiment 2 — Memory-aware reasoning layer: history-aware retrieval + plan-and-execute

**Date:** 2026-06-06
**Harness:** interactive REPL (`python stage_3/graph.py`), single canonical multi-turn case. *Not* a scripted eval — this was feature development plus hypothesis-driven debugging. See "Known limitations" for what that costs.

### Setup

- **What's new:** the agent gained a short-term memory layer (`history` in state, persisted by the checkpointer) and, on top of it, three reasoning behaviors this experiment validates:
  1. **History-aware query rewriting** — resolve pronouns/references against the conversation before retrieval (`plan_query_node`, replacing the earlier `rewrite_query_node`).
  2. **Plan-and-execute decomposition** — split a question into one `sub_query` per topic, but *only* for topics not already covered in history; the ReAct "Thought" step deciding what to fetch.
  3. **History-aware grading** — `grade_relevance` and `grade_groundedness` (and the deterministic `verify_citations` floor) judge the answer against history **+** retrieved chunks, not chunks alone.
- **Test method:** one canonical cross-entity follow-up, run interactively:
  - Turn 1: *"What is FlashAttention?"* (corpus question, populates history)
  - Turn 2: *"How does it compare to GPTQ?"*
- **Why this case:** it simultaneously stresses every new behavior — pronoun resolution ("it"), the known-vs-missing boundary (FlashAttention is in history; GPTQ is not), cross-entity synthesis (no single paper compares the two), and the corrective branches (GPTQ is in-corpus but may retrieve thin, and could fall through to ingestion). One question, maximum surface area. This is the complement to Experiment 1's per-node isolation: end-to-end, but n=1.
- **Agent model:** `claude-sonnet-4-6` throughout (the `/model` switch mid-session changed the *coding assistant*, not the agent's hardcoded `MODEL`).

### Hypothesis

The design bet, committed before running turn 2: plan-and-execute + history-aware grading will answer the comparison **grounded**, by (a) routing to `corpus` (not `followup`), (b) rewriting "it" → FlashAttention, (c) fetching **only** GPTQ (FlashAttention already known from history), (d) grading the retrieval sufficient on history+chunks combined, and (e) passing groundedness on the synthesized comparison.

Sub-prediction (honest, and as it turned out wrong in its optimism): failures, if any, would surface in the **grading gates** — the LLM-judge surface validated in Exp 1 — not in routing or plumbing.

### Result

The design did **not** work on the first attempt. Five distinct failure modes surfaced and were fixed in sequence; only then did the case converge. The sequence *is* the finding.

| # | Observed failure | Root cause | Fix |
|---|------------------|-----------|-----|
| 1 | "compare to GPTQ?" routed to `followup` → answered from parametric knowledge, ungrounded, no retrieval | `ROUTE_TOOL` didn't distinguish "operates on prior content" from "introduces a new entity" | followup desc: any **new** entity/paper/technique → `corpus`. Tiebreaker: when in doubt → corpus (spurious retrieval is cheap; spurious followup is silently ungrounded) |
| 2 | Plan retrieved **both** FA and GPTQ; grader then rejected on "'it' is ambiguous" | plan didn't reason about what history already covered; graders saw raw `question`, not the rewrite | plan Step 2 skips history-covered topics → `sub_queries=['GPTQ ...']`; graders use `rewritten_query` + injected history |
| 3 | After ingestion loop-back, query reverted to raw "How does it compare to GPTQ?" | `ingest_node` reset `query` to `state["question"]`; rewrite was never stored | new state fields `rewritten_query`/`sub_queries`, preserved across the loop |
| 4 | FlashAttention citation flagged as **fabricated** | `verify_citations` checked only current `chunks` (GPTQ); FA was in history | added `history` param — papers cited in prior assistant turns are accepted (validated when first generated) |
| 5 | **Groundedness flagged "FlashAttention benefits both training and inference"** — a reasonable inference, not a fabrication; loop never converged, hit the cap | a binary grounded/not grader punishes synthesis; each regeneration produced a *different* over-generalization | `GROUND_TOOL` flags **only** specific fabricated facts/numbers/results, not high-level synthesis ("comparisons synthesize by nature") |

**Final converged run (turn 2):**

```
[route] intent=corpus
[plan] rewritten='How does FlashAttention compare to GPTQ?'
[plan] sub_queries=['GPTQ quantization method for large language models']
[grade] sufficient=True (attempt 1)
[groundedness] grounded=True (gen attempt 1) — none
```

The comparison answered grounded on the first generation: pronoun resolved, only the missing entity fetched, synthesis claims preserved ("complementary techniques," "could be applied simultaneously"), every GPTQ number correctly cited, and — critically — the prior fabrication (a "4–8× size reduction" figure that appeared in an earlier draft) absent.

### Interpretation

1. **The sub-prediction was half right.** Failures did cluster in the grading gates (#2, #4, #5) — but routing (#1) and state plumbing (#3) failed too, which the hypothesis didn't anticipate. The honest read: adding a new *context source* (history) is not a local change. **Every node that consumes or judges the answer must be re-threaded to see it** — router, planner, both graders, the citation floor, and the state that survives loops. Four of the five bugs are the same systemic omission viewed from different nodes.

2. **Headline finding — groundedness verification is in tension with synthesis (#5).** A strict grounded-or-not grader treats every inferential leap as ungrounded, so it punishes the model precisely for doing a comparison well. The fix is a *distinction*, not a threshold: fabricated specifics (a number/benchmark absent from sources) vs. reasonable characterization (a statement that follows from combining sources). That line kept the real catch (the "4–8×" fabrication) while letting synthesis pass. This generalizes beyond comparisons — any multi-source answer synthesizes.

3. **Plan-and-execute is the ReAct "Thought" the earlier loop lacked.** The prior graph was conditional routing with implicit grading; the agent never reasoned about *what it already knew*. `plan_query_node` Step 2 ("skip topics already in history") makes that reasoning explicit and observable in the trace — and it's what produced the single-`sub_query` fetch instead of redundantly re-retrieving FlashAttention.

### Decision

- **Ship the memory-aware reasoning layer** — verified on the canonical case end-to-end.
- **Convert this n=1 arc into labeled evals** (the real next step, mirroring Exp 1's per-node discipline): (a) a **router** classification set (corpus vs followup vs — later — memory_recall), explicitly including new-entity follow-ups, the failure that started this; (b) a **planner** decomposition set (does it produce the right sub_queries, and does it correctly *omit* history-covered topics?); (c) **comparison-grounding** cases that check synthesis survives while fabrications are caught.
- **`verify_citations` history-acceptance is a deliberate trust boundary:** we trust a paper cited in a prior turn because it was groundedness-checked then. Documented so it isn't mistaken for a hole.

### Known limitations

- **n=1, interactive, not scripted.** One test case, run by hand. Directional evidence that the path works — not a measured success rate. The Exp 1 per-node harnesses are the model to follow; this arc has not yet been turned into one.
- **Hypotheses partly reconstructed during debugging.** The top-level design bet was committed before turn 2, but the five fix-level hypotheses were formed *as* each failure appeared — this is debugging, not pre-registered experimentation. Disclosed rather than dressed up as foresight.
- **Keep-best fallback is untested.** `respond_node` now returns the lowest-issue draft (`best_answer`/`best_n_issues`), but the final run converged on gen-attempt-1, so the fallback path never fired. Needs a deliberately twice-failing groundedness case.
- **The synthesis-vs-fabrication boundary is a prompt heuristic**, not a measured one — its precision/recall (does it ever wave through a real fabrication framed as "synthesis"?) is exactly what eval (c) above must quantify.

---

## Experiment 3 — Per-node eval of the intent router (Roadmap Phase 0.1)

**Date:** 2026-06-06
**Harness:** `stage_3/eval/eval_router.py`

First experiment under the production-readiness roadmap (`ROADMAP.md`). The intent router (`route_intent_node`) is the classifier that broke in Exp 2 — a new-entity follow-up was mis-routed to `followup` and answered ungrounded from parametric knowledge. Exp 2 fixed the prompt; this experiment *validates* the fix with a labeled per-node eval (the Exp 1 discipline).

### Setup

- **Method: per-node, in isolation** — call `route_intent_node({"history", "question"})` on controlled inputs, compare verdict to label. No full graph, no retrieval.
- **What makes this classifier different from Exp 1's gates:** its input includes the *conversation*. The same message ("How does it affect accuracy?") is `corpus` or `followup` depending on what history already contains. So every fixture is a **(history, message, label)** triple, with assistant history-content deliberately scoped to make labels defensible.
- **Labeled set (n=17 scored):** A — first-turn standalone (corpus ×3); B — pure transform/recall of the prior answer (followup ×4); C — **new-entity follow-ups phrased conversationally** (corpus ×5, the Exp 2 failure class, flagged SAFETY-CRITICAL); D — same-entity *deepening* where the asked aspect is NOT in history (corpus ×3); E — same-entity but answerable from history (followup ×2). Plus 2 **taxonomy-gap** cases ("What can you do?", "Thanks!") reported but **NOT scored** — neither class fits.
- **Safety asymmetry (the metric that matters):**
  - **DANGEROUS = corpus→followup.** Answers from history with no retrieval → silently ungrounded. The `followup` path is `answer_from_history → END` — **no relevance/groundedness gate, no corrective net**. Drive to 0.
  - **SAFE = followup→corpus.** Wasteful extra retrieval, but the answer is still grounded.
- **Design payload — rows D and E are the same grammatical shape, opposite labels** ("how does it [aspect]?"), flipping only on whether history covers the aspect. This tests whether the router reasons about *information-sufficiency*, not just question form.

### Hypothesis

Committed before the run: after Exp 2's `ROUTE_TOOL` new-entity fix, **`DANGEROUS` is empty** — every SAFETY-CRITICAL new-entity follow-up routes to corpus; residual errors, if any, fall in `SAFE`.

### Result

**Run 1 — partially falsified. 16/17 (94%), DANGEROUS = 1, SAFE = 0.**

- **5/5 SAFETY-CRITICAL new-entity cases passed** (GPTQ, PagedAttention, SmoothQuant, Medusa, Mamba) — the Exp 2 fix held; the main sub-claim is confirmed.
- **1 DANGEROUS error:** *"How do these techniques reduce latency?"* (after a KV-cache history that lists techniques but no latency mechanism) → routed `followup`. Its row-D sibling *"What speedup numbers does it achieve?"* and its row-E contrast *"How does it affect accuracy?"* both routed correctly — so the miss is specifically on a same-entity-deepening question the router judged answerable from history when it wasn't.
- (A cleanup preceded this: the 2 taxonomy-gap cases were initially double-counted — scored in `LABELED` *and* reported in `GAP_CASES`. Removed from scoring → denominator is the 17 genuinely-labeled cases.)

**Fix — information-sufficiency tiebreaker.** Extended the `ROUTE_TOOL` `intent` description: for a topic already discussed, if the question asks about an aspect/detail/number/mechanism **not actually stated in the prior answer**, classify `corpus`; *when unsure whether the conversation fully covers the answer, choose corpus.*

**Run 2 — 16/17 (94%), DANGEROUS = 0, SAFE = 1.** Same accuracy, but **the error relocated from the dangerous column to the safe column.** The latency case now routes `corpus`; the cost is the row-E accuracy case flipping `followup→corpus` (a safe error).

### Interpretation

1. **The router reasoned about entity-novelty but not information-sufficiency.** It reliably routes a *new named entity* to corpus (5/5), but a same-entity question whose answer isn't in history looked like a followup. The D/E paired contrast isolates this exactly: identical surface form, label set by history content. This is the Exp 1 "GPTQ" pattern — the single error is the boundary teaching us something, not classifier noise.

2. **The tiebreaker is the fail-safe principle made concrete.** Accuracy didn't move; *safety* did. Every residual error is now recoverable (a wasteful retrieval) rather than unrecoverable (a silently ungrounded answer). Justified by the asymmetry: the `followup` path has no corrective net, so ambiguity must resolve toward the path that does.

3. **The lone SAFE error is barely an error.** "How does it affect accuracy?" *is* answerable from history, but retrieving real accuracy data instead of leaning on a one-line history mention yields a grounded, likely richer answer. We pay one round-trip; we don't lose grounding.

4. **Severity nuance, stated honestly:** even the original DANGEROUS miss was less harmful than Exp 2's GPTQ case, because `answer_from_history_node` is constrained to "use ONLY the conversation" — worst case is a *thin* answer, not a confident fabrication from parametric knowledge. Still worth fixing (it should retrieve), which it now does.

### Decision

- **Router validated** for the current fixtures: 0 dangerous errors, all errors fail-safe.
- **Taxonomy gap logged, not patched:** "What can you do?" / "Thanks!" fit neither class. Defer to a dedicated meta/chitchat route — naturally folded into **Roadmap 0.6**, where the router becomes 3-class with `memory_recall` anyway. The harness already separates scored vs gap cases so adding a label is trivial.
- **0.1 complete.** Next: **0.2** — eval the planner (`plan_query_node`) for sub-query correctness and history-covered-topic omission.

### Known limitations

- **Small n (17 scored).** Directional, not precise — especially the safe/dangerous split rests on a handful of boundary cases.
- **Single-pass, stochastic.** The router is a forced-tool LLM call; one pass per case (matches the Exp 1 convention). A boundary case could flip run-to-run; SAFETY-CRITICAL cases should be run ×N and reported worst-case before any strong robustness claim.
- **Labels for rows D/E are history-relative by construction.** The corpus/followup boundary for same-entity questions genuinely depends on what the prior answer contained — the fixtures encode one defensible reading, not a universal ground truth.
- **Accuracy is the secondary metric.** The verdict is `DANGEROUS == 0`; the percentage is reported but should not be optimized in isolation (the tiebreaker proves the point — better behavior, identical accuracy).
