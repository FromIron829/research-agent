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

---

## Experiment 4 — Per-node eval of the query planner (Roadmap Phase 0.2)

**Date:** 2026-06-06
**Harness:** `stage_3/eval/eval_planner.py` (+ a throwaway robustness re-run on the boundary cases)

The planner (`plan_query_node`) is the ReAct "Thought" step from Exp 2: it rewrites a follow-up into a standalone question and decomposes it into `sub_queries` — one per topic that needs retrieval, **omitting** topics already answered in history. This experiment validates both jobs.

### Setup

- **Method: per-node, in isolation** — call `plan_query_node({"history", "question"})`, inspect `sub_queries`.
- **Why the metric is different from the router's:** the output is **structured** (a list), not a single label, and many phrasings are valid. So we don't check string equality — we measure **entity coverage**: which entities the plan will actually retrieve. Each fixture labels two sets:
  - **`should_fetch`** — entities that MUST appear in `sub_queries`; a miss = **DANGEROUS** (never retrieved → ungrounded/incomplete).
  - **`should_omit`** — history-covered entities that should NOT appear; presence = **WASTEFUL** (redundant re-retrieval, but grounded).
- **Tolerant matcher:** `_hit()` normalizes case/spacing and checks a per-entity alias list ("FlashAttention" / "Flash Attention" / "flashattention"). The matcher can only false-*fail* (an alias gap), never false-pass — so a clean result can't be a matcher artifact.
- **Set (n=9 scored + 1 gap):** A passthrough (empty history); B canonical comparison (one entity known → fetch the new, omit the known); C new-entity follow-ups; D multi-decomposition (two NEW entities → both must appear); **E over-omission traps ×3** — entity *named* in history but the needed aspect *not* present, so it must be refetched ("what speedup numbers?", "how does H2O evict?", "how do these reduce latency?"). Gap: both entities known.

### Hypothesis

Committed before the run: **`DANGEROUS = 0`**, and if a dangerous miss exists it will be in the **over-omission traps (E)** — the same information-sufficiency boundary that bit the router, applied to omission (mention ≠ coverage).

### Result

**9/9 clean, DANGEROUS = 0, WASTEFUL = 0.** Hypothesis confirmed — cleanly (contrast Exp 3, which was partially falsified).

- **The three over-omission traps all passed** for the right reason: each refetched the named-but-unexplained entity (`FlashAttention speedup numbers`, `H2O ... eviction policy`, `KV cache compression ... latency`) instead of omitting it as "already discussed."
- **Canonical case** textbook: "How does it compare to GPTQ?" → rewrite resolved *it→FlashAttention*, `sub_queries=['GPTQ quantization method']`, FA omitted.
- **Multi-decomposition** split correctly: "GPTQ and AWQ" → two sub-queries.
- **Robustness re-run** (the 3 traps ×3 more each, on top of the original): all **STABLE-PASS** — 4 clean samples per trap, no flips. So the clean sweep is not a single-sample fluke on the stochastic boundary cases.

### Interpretation

1. **The clean sweep is real, and explainable — not saturation.** A 9/9 warrants the same suspicion the Stage 2 judge earned. Two checks defuse it: the matcher can only false-fail (passes are genuine), and the planner *prompt* bakes in sufficiency — *"skip topics already explained **in detail** … only sub-queries for topics NOT yet covered."* The planner was **built** sufficiency-aware, which is exactly the property the router *lacked* and had to be patched for (Exp 3). Same principle, opposite starting point.

2. **The planner never over-omits — even when baited.** The traps were designed to trick it into dropping a mentioned entity; it refetched all three. It errs toward fetching when coverage is uncertain — the fail-safe direction.

3. **The GAP case flipped my framing — and strengthened the result.** I expected "both entities known → ~0 sub_queries (redundant fetch)." Instead, for *"which is more memory-efficient?"* over a thin `MULTI_HIST` (which names FA and GPTQ but says nothing about memory efficiency), the planner fetched **both** — correctly, because the *asked aspect* isn't in history. So the fixture never actually tested "fetch nothing": its history was too thin to make the entities' relevant aspect "known." That's a **fixture-design limitation, not a planner defect** — and the observed behavior is the *consistent* sufficiency logic, not over-fetching.

4. **Cross-node consistency (the portfolio point):** router (Exp 3) and planner (Exp 4) both resolve ambiguity by *information-sufficiency* — does history actually contain the answer, not merely mention the topic. The router needed a tiebreaker to get there; the planner had it by construction. The agent reasons the same way at both decision points.

### Decision

- **Planner validated** for coverage + omission: 0 dangerous misses, 0 redundant fetches, traps stable across 4 samples.
- **One case remains genuinely untested:** "history *fully covers* the answer → planner emits ~0 sub-queries." The current gap fixture's history is too thin to trigger it. Logged for a future rich-history fixture (also relevant once 0.5 summarization changes what history contains).
- **0.2 complete.** Next: **0.3** — comparison-grounding eval (does synthesis survive while fabricated specifics are caught — the Exp 2 headline, now measured).

### Known limitations

- **Small n (9 + 9 robustness samples).** Directional; the boundary rests on a few well-chosen traps.
- **Only the coverage dimension is scored.** Rewrite quality (pronoun resolution) and decomposition *count* (`len(sub_queries) ≥ 2` for comparisons) are observed in the logs but not asserted — a clean coverage result implies the rewrite is fine (a broken rewrite usually surfaces as a missing fetch), but they aren't independently measured.
- **Matcher depends on a hand-maintained alias list** — a new entity with an unlisted alias would false-fail; read the printed `sub_queries` before trusting any future DANGEROUS flag.
- **The true "fetch-nothing" omission case is not yet tested** (gap fixture too thin), so the *upper* bound of the omission optimization is unverified — only that it never *under*-fetches.

---

## Experiment 5 — Synthesis-vs-fabrication discrimination in the groundedness gate (Roadmap Phase 0.3)

**Date:** 2026-06-06
**Harness:** `stage_3/eval/eval_grounding_synthesis.py` (N=3 per case)

Exp 2's headline was that a binary groundedness grader punishes synthesis, fixed by teaching it to flag *fabricated specifics* but spare *reasonable inference*. That was a fix, asserted on one example. This experiment **measures** the distinction adversarially.

### Setup

- **Isolation:** two papers (FlashAttention + GPTQ), **both "retrieved"**, so `verify_citations` never fires — this isolates the *LLM grader's* judgment, the part Exp 2 changed.
- **5 cases, two must-spare / two must-catch / one control:**
  - SYNTHESIS ×2 — cross-source synthesis + a reasonable inference ("complementary", "memory-centric philosophy") not verbatim in any single source → expect **grounded**.
  - GROUNDED-SPECIFICS ×1 (control) — specific numbers that *are* in the sources → expect **grounded**.
  - FABRICATION ×2 — a fabricated number ("4-8x at 2-bit") embedded *amid valid synthesis*, and a fabricated benchmark ("50% lower perplexity") *beside a real number* → expect **ungrounded**.
- **Two error types:** FALSE POSITIVE (synthesis wrongly flagged — the Exp 2 regression) and FALSE NEGATIVE (fabrication passed — DANGEROUS).
- **N=3 per case** because this boundary is the stochastic one; report per-case stability.

### Hypothesis

0 false positives **and** 0 false negatives. The discriminating cases (fabrication embedded in valid synthesis) are the most likely to fail — the grader could either get "distracted" by surrounding grounded content (miss the fabrication) or over-react and flag the synthesis.

### Result

**5/5, all stable across N=3 (15 samples). FALSE POSITIVES 0/3, FALSE NEGATIVES 0/2.** Confirmed.

The two discriminating cases are the result that matters: the grader caught the fabricated number *while leaving the surrounding synthesis intact*, and caught the fabricated benchmark *sitting next to a legitimate one*. No flips across samples.

### Interpretation

1. **The Exp 2 distinction is real and measured, not anecdotal.** The grader separates "a specific fact/number absent from sources" from "a characterization that follows from combining sources" — and does so stably.
2. **It isn't fooled by context.** A fabrication embedded in otherwise-grounded synthesis is still caught (no false negative), and valid synthesis sitting next to a real number isn't dragged down (no false positive). The grader evaluates claims, not vibes.
3. **Stability matters as much as the verdict** — a single-pass 5/5 on a stochastic boundary would be weak; 15/15 makes the claim credible.

### Decision

- **Synthesis/fabrication boundary validated** for the current fixtures. 0.3 complete.
- This is the third validated gate (relevance Exp 1, router Exp 3, groundedness Exp 1+5) — the agent's LLM-judge surfaces are now all measured, not assumed.

### Known limitations

- Hand-crafted fixtures, n=5, single corpus-pair — directional. The boundary is a prompt heuristic; subtler fabrications (e.g., a plausible-but-wrong number close to a real one) would stress it harder and tighten the precision/recall estimate.
- All fabrications here are *specific numbers*; a fabricated *causal/mechanistic* claim with no number is a different shape not covered.

---

## Experiment 6 — Adversarial test of the keep-best fallback (Roadmap Phase 0.4)

**Date:** 2026-06-06
**Harness:** `stage_3/eval/test_keep_best.py`

The keep-best fallback (`respond_node` returns the least-fabricated draft when groundedness regeneration is capped) shipped in Exp 2 **untested** — the verified runs all converged before the cap, so the fallback path never fired. This experiment forces it.

### Setup

- **Part 1 — selection logic, deterministic (no LLM):** drive `respond_node` directly across its four branches (grounded→current; ungrounded→best; ungrounded+no-best→first; ungrounded+nothing→current).
- **Part 2 — keep-best tracking, real grader:** simulate a twice-failing loop. Draft **A** has 3 fabricated specifics, draft **B** has 1. Thread the state through `grade_groundedness_node` twice (gen1=A, gen2=B), then `respond_node`. Assert the fallback returns **B** (fewer fabrications), not A (the first draft).

### Hypothesis

Keep-best returns the least-fabricated draft — so the fallback yields B, not A.

### Result — hypothesis FALSIFIED, bug found, fixed, re-verified.

**First run: Part 1 passed 4/4; Part 2 FAILED.** Trace: `best_n_issues` was **1 after both A (3 fabs) and B (1 fab)** — identical — so `best` never updated past gen1 and the fallback returned **A, the worse first draft.**

**Root cause:** `n_issues = max(1, len(issues.split("|")))`. The LLM grader returns `issues` as prose, not `|`-delimited, so the count was **1 for any ungrounded answer regardless of how many fabrications it contained.** Keep-best silently degenerated to **keep-FIRST.**

**Fix:** added a structured `n_fabrications` integer to `GROUND_TOOL`; `n_issues = llm_fabrications + len(fabricated_citations)`. **Re-run: `best_n_issues` 3 → 1, B kept over A. 6/6 checks pass.** The synthesis eval (Exp 5) was re-run after the schema change — no regression (still 5/5).

### Interpretation

1. **This is the canonical case for why untested fallbacks are dangerous.** Keep-best *looked* implemented — the field existed, the wiring was there — but it silently did keep-first and returned the **worse** draft. End-to-end it was invisible: the user still got "an answer," just not the best one. Only an adversarial node-level test with *controlled, differing fabrication counts* could expose it.
2. **The bug was a metric-granularity mismatch:** the keep-best decision needed a count that distinguishes 3 from 1, but the count it used could only ever be 1. The fix moved the count from a fragile string-split to a value the grader reports directly.
3. **Test design is the lesson:** a fallback test must (a) *force* the fallback to fire and (b) make its branches *distinguishable* — here, two drafts with deliberately different fabrication counts. A test that only checked "an answer comes back" would have passed against the bug.

### Decision

- **Keep-best fixed and verified.** 0.4 complete. `n_fabrications` is now a cleaner issues signal available elsewhere if needed.
- **Bug class flagged for the roadmap:** other "looks-implemented" paths (the followup answerability, the ingest error branches) deserve the same forced-path testing before being trusted.

### Known limitations

- Part 2 relies on the grader counting fabrications *roughly* right (A=3 vs B=1); robust because the gap is large, but two near-equal drafts could still tie `best_n_issues` and fall back to keep-first. Acceptable — ties on near-equal drafts don't matter much — but worth noting.
- Single sample for Part 2 (the deterministic Part 1 is the stronger guarantee). The fabrication *count* is mildly stochastic even if the ordering is stable.

---

## Experiment 7 — Rolling history summarization (Roadmap Phase 0.5)

**Date:** 2026-06-06
**Harness:** `memory.py` unit checks + an end-to-end smoke invoke (this is a *feature*, not an eval — verified by assertions, not a labeled set).

Short-term memory previously kept only the last `MAX_TURNS` turns (`format_history` truncation); everything older was **dropped**. This replaces truncation with a **rolling summary** so long sessions keep their early context compactly.

### Setup

- **Design:** a `summarize_node` at the graph entry (`START → summarize → route_intent`) folds turns that have fallen out of the recent window into a running `summary`; `format_history(history, summary)` prepends `summary` + the last `MAX_TURNS` turns verbatim.
- **Incremental, not re-summarizing:** state carries `summary` + `n_summarized` (count of messages already folded). Each turn folds only `evictable[n_summarized:]` — newly-evicted turns — so no turn is summarized twice (summarization is itself an LLM call; re-folding would be wasteful).
- **State-persistence subtlety:** `summary` and `n_summarized` are deliberately **absent from `fresh_turn`**, so the checkpointer carries them across turns (same pattern as `history`). Putting them in `fresh_turn` would wipe the summary every turn.

### Hypothesis

(1) Old context is preserved compactly instead of dropped; (2) folding is incremental (only newly-evicted turns); (3) the first turns are free (no eviction → no LLM call); (4) the full graph flow is unbroken through the new entry node.

### Result — all four confirmed.

- **`format_history`:** summary block + last-6-turns verbatim; the newest turns occupy the window and old turns do not.
- **First fold:** with 9 turns (18 msgs) and `MAX_TURNS=6`, exactly the 6 evicted messages were summarized (`n_summarized=6`), and the oldest topics (FlashAttention, GPTQ, AWQ) appeared in the 485-char summary.
- **Idempotent:** re-folding the same history is a no-op (no re-summarization).
- **Incremental:** two more turns → exactly the 2 newly-evicted messages folded (`n_summarized=8`).
- **End-to-end:** a full invoke runs `summarize → … → respond` and returns a grounded answer; `summary` is empty after turn 1 (no eviction, no wasted call).

### Interpretation

1. **It bounds the *context* sent to the LLM while preserving old information compressed** — which is the actual goal. Truncation bounded context too, but by *losing* the old turns; summarization keeps them. The difference shows up exactly on a follow-up that references something said many turns ago.
2. **The `n_summarized` counter is the cost-correctness piece:** without it, every turn would re-summarize the whole evicted prefix (an LLM call each time). With it, summarization cost is paid once per evicted turn.
3. **Short sessions pay nothing:** `summarize_node` is a no-op until history exceeds the window — verified by the empty summary and absent `[summarize]` log on turn 1.

### Decision

- **Summarization shipped and verified.** 0.5 complete. Phase 0 (the memory layer) is now done except **0.6** (long-term episodic memory).
- The window size `MAX_TURNS=6` is unchanged; tuning it is a knob, not a correctness issue.

### Known limitations

- **History *storage* is still unbounded** — the `operator.add` reducer is append-only, so the full transcript stays in state; only the *context window* is bounded. Bounding storage (a trimming reducer / message eviction) is a separate, deferred concern.
- **Summary *fidelity* is not evaluated.** The checks confirm the right turns are folded and topics appear, but not that a downstream node can correctly answer a follow-up whose answer now lives *only* in the summary. That's the real test of summarization quality (a "followup answered from summarized-away content" case) and is not yet built.
- **Compounding compression loss** under many successive evictions (summary-of-summary drift) is untested.
- Each eviction triggers one summarization LLM call (added latency on the turn that crosses the window) — acceptable, but a per-turn cost worth noting for the Phase 2 cost work.

---

## Experiment 8 — Long-term episodic memory: routing + recall@k (Roadmap Phase 0.6)

**Date:** 2026-06-09
**Harness:** `stage_3/eval/eval_episodic.py` (routing + recall@k, self-cleaning seed)

The third memory tier: **episodic memory across sessions** — "RAG over your own conversation turns." Components: `episodic.py` (a Chroma `conversations` collection), a `memory_recall` router intent, a `recall_node`, and a write path in `respond_node` (`remember_turn`). The checkpointer is per-thread; this store outlives any thread.

### Setup

- **Part 1 — routing (3-class):** `route_intent_node` on `memory_recall` questions + corpus/followup controls. Two error types: **recall-miss** (a recall question routed elsewhere → no episodic lookup) and **hijack** (a real question pulled into `memory_recall` → answered "nothing stored").
- **Part 2 — recall@k:** seed 5 distinct past turns, query with **paraphrases that share no keywords with the stored questions**, check the right turn is retrieved. The eval seeds via `remember_turn` and self-cleans its thread.

### Hypothesis

Routing sends `memory_recall` questions to `memory_recall` with **0 hijacks**; recall retrieves the right past turn for paraphrased queries. The `followup`/`memory_recall` boundary is the likely routing soft spot (both operate on "past conversation").

### Result — feature works; the recall eval caught TWO real bugs before any clean number.

**Routing: 7/8, 0 hijacks.** The one miss is the predicted boundary — *"Remind me what we discussed about quantization"* → `followup` (a recall-miss, the safe direction; no real question was hijacked).

**Recall — two bugs surfaced, fixed, then validated:**
1. **Wrong distance metric.** The `papers` collection uses cosine (`hnsw:space`); `conversations` was created with no metadata → **L2 default**. OpenAI embeddings aren't unit-normalized, so L2 is magnitude-dominated → the same 2-3 docs ranked top for *every* query. Fix: create with `metadata={"hnsw:space": "cosine"}`. (Gotcha: Chroma ignores the metric on an existing collection — had to delete + recreate.)
2. **Silent id collision.** Ids were `f"{thread_id}-{ts:.0f}"` — integer-**second** granularity. Seeding 5 turns rapidly → they fell in 2 one-second buckets → **only 2 of 5 survived** (same-second `upsert` overwrites). Fix: `uuid4`-based ids; keep `ts` in metadata for recency.

After both fixes: **recall@3 = 5/5, recall@1 = 2/5.**

**Improvement — embed target A/B (the `recall@1` gap):**

| Embedded text | recall@1 | recall@3 |
|---|---|---|
| question only | 2/5 | 5/5 |
| **question + answer** | **5/5** | 5/5 |

The misses were all *description* queries ("the IO-aware attention algorithm") against *bare-question* embeddings ("What is FlashAttention?") — the descriptive content lives in the **answer**. Embedding `question + answer` puts it in the vector → recall@1 2/5 → 5/5.

### Interpretation

1. **The recall@k eval earned its keep — twice.** Both bugs were invisible to the end-to-end smoke test (which used few entries and writes spaced far enough apart to avoid same-second collision, and keyword-overlapping queries). Only a *multi-write, paraphrase-query* eval exposed them. The id collision in particular is **silent data loss** — the dangerous kind: the store "worked," returned answers, and quietly dropped turns.
2. **Both are the "looks-implemented" trap again** (cf. keep-best, 0.4): code that runs and returns plausible output but is broken under real load. The discipline that catches them is forcing the real conditions — concurrent writes, semantically-distant queries.
3. **The embed target is a genuine retrieval lever, now measured.** A name embeds far from its description; recall queries are descriptions. `question + answer` is the cheap, correct default — 2/5→5/5 with one line.
4. **`recall_node` uses k=3, and recall@3 was 5/5 even before the embed fix** — so the right turn was always in the LLM's context. The feature was *functionally* correct before the recall@1 improvement; the A/B sharpened ordering, not correctness.
5. **Routing boundary** (`followup` vs `memory_recall`) is the 0.1 finding one tier up: both touch "past conversation," distinguished by in-session vs cross-session. Tiebreaker deferred.

### Decision

- **0.6 complete. Phase 0 (the memory layer) is COMPLETE.** Three tiers — short-term history, in-session summary, cross-session episodic — all built and per-node validated.
- **Next: Phase 1** (durable, session-aware state) — the highest-leverage gap, and the thing that makes *all three* memory tiers usable in the deployed API.

### Known limitations

- **Tiny store (5 seeds).** Recall over a small collection is easy; a realistic store (thousands of turns, near-duplicate questions) is harder — these numbers are directional.
- **No recency/temporal filtering.** "Last week" is ignored — recall is pure similarity. `ts` is stored but unused for ranking.
- **No per-user isolation.** The store is global (no auth) — same multi-tenancy gap as Roadmap Tier 3.
- **Unbounded growth.** Every corpus turn is stored forever; no dedup/decay.
- **`followup`/`memory_recall` boundary** unfixed (1/8 routing miss) — needs the in-session-vs-cross-session tiebreaker.
- **Cosine-metric gotcha** is a footgun: changing a collection's metric requires delete + re-seed.
