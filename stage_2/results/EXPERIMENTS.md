# Stage 2 — Experiment Log

---

## Experiment 1 — Agent loop baseline (n = 1)

**Date:** 2026-05-27
**Result file:** */stage_2/results/agent_runs/run_20260527_200856.json*

### Setup

- **Retrieval method:** hybrid search (vector + BM25 + RRF), `initial_k = 50`
- **LLM model:** `claude-sonnet-4-6`
- **Loop config:** max 6 iterations, max 2048 output tokens per turn
- **System prompt:** see `agent.py` SYSTEM_PROMPT at commit `<hash>`
- **Query rewrite:** agent is allowed to rewrite the query across iterations to retrieve more precise chunks
- **Parallel tool calls:** Claude's native parallel tool-use enabled (multiple `tool_use` blocks per turn)
- **ReAct loop:** agent may call `retrieve` multiple times if early results don't give strong evidence
- **Test question (n=1):** *"How can speculative decoding be made effective at large batch sizes? What are the bottlenecks?"*

### Hypothesis

Agents need several iterations to gather strong supporting chunks for the user's query. For latency and token usage, I didn't know what to expect — first run, no prior.

### Result

- **Outcome:** synthesis truncated because `max_tokens = 2048` hit on the final iteration.
- **Total wall-clock:** **118.3 s** — LLM 116.3 s (98%), retrieve 2.0 s (2%), overhead ~0 s
- **Token usage:** 95,674 total — input 93,203, output 2,471
- **Iterations:** 4

**Per-iteration breakdown**

| Iteration | LLM time | In tokens | Out tokens |
|-----------|----------|-----------|------------|
| 1         | 3.1 s    | 975       | 138        |
| 2         | 3.1 s    | 19,305    | 142        |
| 3         | 8.7 s    | 30,802    | 143        |
| 4         | 101.5 s  | 42,121    | 2,048      |

**Cost estimate** *(Sonnet pricing: $3 / $15 per 1M tokens input / output)*

| Type      | Tokens     | Rate      | Cost     |
|-----------|------------|-----------|----------|
| Input     | 93,203     | $3 / 1M   | $0.2796  |
| Output    | 2,471      | $15 / 1M  | $0.0371  |
| **Total** | **95,674** |           | **$0.317** |

> Pricing reference: https://platform.claude.com/docs/en/about-claude/pricing

### Interpretation

Performance was worse than expected. Several distinct findings:

1. **Query refinement worked.** The agent issued different `retrieve` queries each iteration, starting broad ("speculative decoding large batch sizes throughput") and narrowing to specific mechanisms ("compute-bound memory-bound verification cost batch size"). The ReAct refinement loop is doing its job.

2. **Agent emitted ZERO reasoning text in iterations 1–3.** The model went straight to tool calls with no preceding "Thought:" text. This is a transparency problem — the system prompt should explicitly require reasoning before action.

3. **Parallel tool calls worked.** Each iteration issued two `retrieve` calls in a single LLM turn. The model didn't have to wait between deciding-call-1 and deciding-call-2 — free latency win from Claude's native parallel tool-use.

4. **`stop_reason == "max_tokens"` failed silently.** The loop's termination check (`stop_reason != "tool_use"`) treats `max_tokens` and `end_turn` identically. As a result, the agent returned a truncated answer with no warning. Bug in the loop, not just an undersized cap.

5. **Input tokens grew rapidly across iterations** (975 → 19,305 → 30,802 → 42,121). Each iteration's input includes the system prompt + all previous messages + new retrieval results. This is the dominant cost driver.

6. **Cost is input-dominated, not output-dominated.** Input accounted for ~88% of total cost ($0.2796 vs. $0.0371 output) because each iteration re-sends the full prior context plus new retrieval results. The per-turn output cap kept the output side bounded, but the real cost driver was input growth across iterations — exactly the lever prompt caching targets in Experiment 2. Separately, the output cap caused the answer to truncate (see finding #4) — a real product trade-off.

### Decision

**Actions for Experiment 2:**
- Expand `max_tokens_per_turn` from 2048 → 4096 so synthesis doesn't truncate
- Add explicit `stop_reason` branching (handle `end_turn` / `tool_use` / `max_tokens` separately) so truncation surfaces as a warning
- Add a reasoning-first instruction to the system prompt so iterations 1–3 emit visible "Thought:" text
- Implement Anthropic prompt caching for the system prompt + tool definitions (they don't change across iterations)
- Add streaming for synthesis output (UX improvement, not wall-clock)

**Hypothesis for Experiment 2:**
- Prompt caching cuts input cost by ~50% and shaves 10–15% off total wall-clock latency
- Streaming drops time-to-first-token from ~5 s to ~1 s without changing total latency
- To meaningfully cut wall-clock latency further, would need to either cap chunks fed to the final synthesis turn, or split routing/synthesis across model tiers (Haiku for tool-routing, Sonnet for synthesis)

## Experiment 2 — Latency & cost optimization (prompt caching)

**Date:** 2026-05-27
**Result file:** */stage_2/results/agent_runs/run_20260528_201638.json*

### Setup

- Expanded `max_tokens_per_turn` from 2048 → 4096
- Added explicit `stop_reason` branching (handle `end_turn` / `tool_use` / `max_tokens` separately) so truncation surfaces as a warning
- Added "Always begin each turn with one or two sentence of reasoning, stated explicitly. For example: "I need to find papers about X because Y." This reasoning must appear as text BEFORE any tool call." into `SYSTEM_PROMPT`
- Added a single rolled-forward `cache_control` breakpoint on the last content block of the last message each turn (caches tools + system + conversation-so-far)

### Hypothesis

Agent should output "Thought: I should do X to get Y..." at the beginning of every iteration. Latency should be reduced by ~50%-60%, estimated cost also should be lower.

### Result

- **Outcome:** Agent outputs complete and accurate result.
- **Total wall-clock:** **61.762 s** — LLM 59.849 s (97%), retrieve 1.912 s (3%), overhead ~0 s
- **Token usage:** input(full) = 1021 cache_write = 31679 cache_read = 20103 output = 2707
    - **Total input processed:** 52,803 (vs 1,021 billed at full rate)
- **Estimated cost:** $0.169
- **Iterations:** 3

**Per-iteration breakdown**

| Iteration | LLM time | In tokens | Out tokens | Cache write | Cache read |
|-----------|----------|-----------|------------|-------------|------------|
| 1         | 3.09 s   | 1019      | 166        | 0           | 0          |
| 2         | 4.14 s   | 1         | 180        | 20103       | 0          |
| 3         | 52.62 s  | 1         | 2361       | 11576       | 20103      |

**Cost estimate** *(Sonnet 4.6 pricing: $3 / $15 per 1M input / output; cache write 1.25×, cache read 0.1×)*

| Type         | Tokens  | Rate        | Cost       |
|--------------|---------|-------------|------------|
| Input (full) | 1,021   | $3.00 / 1M  | $0.0031    |
| Cache write  | 31,679  | $3.75 / 1M  | $0.1188    |
| Cache read   | 20,103  | $0.30 / 1M  | $0.0060    |
| Output       | 2,707   | $15.00 / 1M | $0.0406    |
| **Total**    |         |             | **$0.169** |

> Pricing reference: https://platform.claude.com/docs/en/about-claude/pricing
> Note: cache write ($0.119) is the largest line item — the savings come from the *read* on the synthesis turn, not the writes.

### Comparison

| Experiment | Total time | Total estimated cost | Latency Improvement | Cost improvement |
|------------|------------|----------------------|---------------------|------------------|
| 1          | 118.3 s    | $0.317               | `None`              | `None`           |
| 2          | **61.762 s** | **$0.169**         | **47.8%**           | **46.7%**        |

### Interpretation

Both latency and cost are reduced, several key findings:

1. **Prompt cache worked:** Latency reduced **~48%** (118.3 s → 61.8 s). Evidence the rolling breakpoint behaved exactly as designed: iteration 2 *wrote* 20,103 tokens to cache, and iteration 3 *read back* exactly 20,103 — the prior turn's prefix served at 0.1×.

2. **ReAct behavior worked:** Agent now outputs "Thought: ..." at the beginning of every iteration.

3. **Cost saving:** Estimated cost dropped **~47%** ($0.317 → $0.169). Caveat: the cache *write* (31,679 tokens at 1.25×) is the single largest line item; the savings come from the *read* on the synthesis turn, not the writes.

4. **Unnecessary content in response:** the agent's synthesis is too long / verbose.

5. **Remaining latency is output-bound.** Iteration 3 alone is 52.6 s of the 61.8 s total — now almost entirely *output generation* (2,361 tokens), which caching cannot speed up (caching only accelerates input prefill). This is why the result (~48%) landed just under the 50–60% hypothesis: the input-prefill time caching removed has been replaced by output-generation time as the dominant cost. It also means finding #4 is a latency lever — a shorter synthesis would cut the largest remaining time sink.

### Decision

**Actions for Experiment 3:**
- Prompt engineering to structure the agent's response (currently too verbose — see finding #4)

**Hypothesis for Experiment 3:**
- Response should be structured and easier to read; a shorter synthesis should also trim output-generation time (the dominant remaining latency cost — see finding #5)

## Experiment 3 — Structured synthesis via prompt engineering (A/B, n = 3)

**Date:** 2026-05-29
**Result files:**
- Structured (Option B): *run_20260529_115028_structure.json* (Q1), *run_20260529_115522_structure.json* (Q2), *run_20260529_115713_structure.json* (Q3)
- Baseline (Exp 2 prompt): *run_20260529_120142_unstructure.json* (Q1), *run_20260529_115945_unstructure.json* (Q2), *run_20260529_115908_unstructure.json* (Q3)

### Setup

- **Change under test:** added an `ANSWER FORMAT` section to `SYSTEM_PROMPT` ("Option B" — lighter guidance + soft length budget, not a rigid template). Five directives: lead with a 1–2 sentence direct answer; supporting points as bullets with headers only when parts are genuinely distinct; aim for under ~250 words unless the question demands more; suppress preamble / question-restatement / generic closing summary; brevity never overrides the groundedness rules (every inline citation kept). Also fixed long-standing typos in the prompt (`cropus`, `Paer`, `explicityly`, `differnt`).
- **Design:** A/B across **3 questions × 2 prompt versions = 6 runs**. Baseline = the exact Exp 2 prompt (no `ANSWER FORMAT` section). This matters because Exp 2 was n=1 — to claim improvement on the two *new* questions, the baseline prompt had to be re-run on them too, not compared against a single prior data point.
- **Question set (chosen for answer-shape diversity, not retrieval difficulty):**
  - **Q1** — *"How can speculative decoding be made effective at large batch sizes? What are the bottlenecks?"* (how/why + bottlenecks; carried over from Exp 1–2 for continuity)
  - **Q2** — *"What does FlashAttention-2 change about work partitioning compared to FlashAttention-1?"* (comparison)
  - **Q3** — *"How can attention complexity be reduced to linear time?"* (definitional / survey-style)
- **Held constant:** retrieval (hybrid, `initial_k = 50`), model (`claude-sonnet-4-6`), `max_tokens = 4096`, prompt caching (rolling breakpoint). Only the system prompt differs between arms.
- **Pre-committed qualitative rubric:** a "good" answer opens with a direct 1–2 sentence answer, supports with cited bullets, carries no preamble, and keeps an inline `[Paper Title (page N)]` citation on every claim.
- **Instrumentation note:** runs were labeled by filename suffix (`_structure` / `_unstructure`); the `prompt_version` JSON field was not stamped (carried forward to Exp 4).

### Hypothesis

*(Committed in Experiment 2's Decision.)* The structured prompt should produce answers that are easier to read **and** shorter; because synthesis output-generation is the dominant remaining latency cost (Exp 2 finding #5), a shorter synthesis should also cut synthesis-turn latency. Groundedness should be unaffected — the citation guard exists precisely to protect it.

### Result

- **Outcome:** structured prompt produced shorter, scannable answers across all three questions; synthesis-turn output tokens and latency dropped in every case. Groundedness held. One new failure mode surfaced (preamble leakage — see Interpretation #5).

**Per-question comparison** *(synthesis turn = final answer turn)*

| Question | Prompt | Synth out (tok) | Synth latency | Total time | Iters | Cost | Inline cites |
|----------|--------|----------------:|--------------:|-----------:|:-----:|------:|:-----:|
| Q1 spec-decoding | structured | 1,063 | 24.4 s | 38.7 s | 3 | $0.124 | 11 |
| Q1 spec-decoding | baseline   | 2,408 | 54.6 s | 68.1 s | 4 | $0.219 | 16 |
| Q2 FlashAttn-2   | structured | 662   | 14.7 s | 19.3 s | 2 | $0.073 | 5 |
| Q2 FlashAttn-2   | baseline   | 993   | 21.4 s | 25.1 s | 2 | $0.077 | 6 |
| Q3 linear attn   | structured | 857   | 19.8 s | 27.6 s | 3 | $0.141 | 6 |
| Q3 linear attn   | baseline   | 1,776 | 36.2 s | 46.9 s | 3 | $0.170 | 14 |

**Deltas (structured vs. baseline)**

| Question | Synth output | Synth latency | Total time |
|----------|:------------:|:-------------:|:----------:|
| Q1 | −55.9% | −55.4% | −43.1% \* |
| Q2 | −33.3% | −31.2% | −23.1% |
| Q3 | −51.7% | −45.2% | −41.2% |
| **mean** | **−47.0%** | **−43.9%** | — |

> \* Q1's total-time delta is confounded — the baseline ran 4 iterations vs. the structured arm's 3 (an extra retrieval round), so its total includes work unrelated to the prompt. The synthesis-turn delta (−55%) is the clean attribution.

### Interpretation

1. **Hypothesis confirmed and generalized.** Synthesis output fell 33–56% and synthesis latency 31–55% across all three answer shapes — not just the question the prompt was tuned against. Mean synthesis-turn reduction: −47% output, −44% latency.

2. **Savings scale with baseline verbosity.** Q2 (comparison) had the leanest baseline (993 output tokens) and the smallest cut (−33%); Q1 and Q3 had verbose baselines (2,408 / 1,776 tokens) and ~−50% cuts. The prompt removes fat in proportion to how much was there — so a single average understates the spread.

3. **Cost improvement < latency improvement.** Q2 cost fell only ~5% despite a 31% latency cut, because cost is dominated by cache-write + input (Exp 2 finding #6), and output is the smaller line item. Trimming output buys latency far more than dollars.

4. **Groundedness preserved — de-duplication, not gutting.** Inline citations dropped (Q1 16→11, Q3 14→6) but proportionally to length. Manual read of the Q3 structured answer shows each of the four technique families still carries its own citation; the baseline was re-citing the *same* sources across more sentences. No fabricated claims observed. One minor quirk: the Longformer/BigBird claim is sourced secondhand via the Nyströmformer paper rather than the primary sources.

5. **New failure mode — reasoning preamble leaks into the final answer (5 of 6 runs, both arms).** Answers opened with "I now have comprehensive evidence. Let me synthesize the answer." — only Q2 structured opened clean. **Root cause is a prompt collision, not a tuning miss:** `WORKFLOW` step 1 ("always begin each turn with reasoning") and `ANSWER FORMAT` ("no preamble") contradict each other on the synthesis turn, and the loop concatenates *all* text blocks into the answer (`agent.py` text-accumulation), gluing the reasoning sentence onto the front. Wording alone can't resolve it because the model emits reasoning and answer in the same turn. This is the most informative result of the experiment — the latency win went exactly as predicted; this did not.

### Decision

**Actions for Experiment 4:**
- Resolve the step-1 ↔ `ANSWER FORMAT` collision. Two candidate fixes:
  - **(a) Prompt:** scope reasoning-first to tool-use turns — "on the final answer turn, begin directly with the answer; no reasoning preamble."
  - **(b) Code:** separate the final turn's reasoning text block from the synthesis in the loop, so reasoning lands in the trace and only the answer lands in `answer`.
- Stamp `prompt_version` into the run JSON (filename-only labeling this round made programmatic A/B comparison manual).
- Treat this as the entry point to a proper **agent-level eval**: with verbosity now controlled, answer-quality / groundedness scoring becomes meaningful (distinct from Stage 1's retrieval-recall eval).

**Hypothesis for Experiment 4:**
- Fix (a) removes the preamble from the final answer with no loss of trace transparency (reasoning still visible on tool-use turns). If the model still leaks reasoning on the synthesis turn despite the instruction, fix (b) is required as the deterministic backstop.

## Experiment 4 — Eliminating reasoning-preamble leakage in the final answer

**Date:** 2026-05-29
**Result files:** *Not persisted this session.* The four fix attempts below were observed in live console output; the run JSONs were not saved to `agent_runs/` (latest saved run is the Exp 3 set at `run_20260529_120142`). **Action:** re-run the three verification questions and save, so the confirmed-fix state has artifacts (and stamp `prompt_version` while doing so).

### Setup

- **Problem (inherited from Exp 3 finding #5):** the agent's reasoning-first instruction (added in Exp 2 for transparency) leaks a transition sentence into the final answer, because the loop concatenates *all* text blocks of the terminal turn into `answer`. `ANSWER FORMAT` banned preamble, but it leaked anyway.
- **Success condition (all four must hold at once):** (1) tool turns still emit reasoning; (2) the final answer begins directly on the answer — no preamble; (3) reasoning is still preserved in the trace; (4) no stray control text in the answer.
- **Method:** iterate on the canonical question (Q1, spec-decoding), then verify the winning fix across the full Exp 3 set (Q1 / Q2 / Q3). All other config held at Exp 3 values (hybrid retrieval, `claude-sonnet-4-6`, `max_tokens = 4096`, prompt caching).

### Hypothesis

*(Committed in Experiment 3's Decision.)* (a) A prompt-only fix removes the preamble with no loss of trace transparency. (b) If the model still leaks despite the instruction, a deterministic code backstop is required.

### Result

Four successive fixes, each diagnosed from the run it produced:

| # | Change | Iter-1 reasoning | Final-answer preamble |
|---|--------|:----------------:|-----------------------|
| 1 | **Conditional prompt** — reasoning required only "on any turn where you call a tool"; final turn told not to narrate | ❌ lost | ✅ clean |
| 2 | **Strong prompt** — "before every retrieve call, including the first" + final-turn exception | ✅ restored | ❌ leaked — *"The evidence is comprehensive. Let me now synthesize the answer."* |
| 3 | **`Thought:` marker + code strip** — reasoning prefixed `Thought:`; loop strips leading `Thought:` lines from the answer | ✅ | ❌ leaked — *"Here is a comprehensive answer synthesizing all the evidence:"* (no `Thought:` prefix, so the strip missed it) |
| 4 | **Positive answer sentinel** — model emits `===ANSWER===`; `extract_answer()` keeps only text after the marker (falls back to `Thought:`-strip if absent) | — | ✅ **clean, robust by construction** |

**Verification of fix #4 across the Exp 3 question set** *(console-observed)*

| Question | Iters | Iter-1 reasons | Final answer clean | `===ANSWER===` emitted | Stray marker | Total time | Cost |
|----------|:-----:|:--------------:|:------------------:|:----------------------:|:------------:|-----------:|-----:|
| Q1 spec-decoding | 3 | ❌ | ✅ | ✅ | none | 33.0 s | $0.159 |
| Q2 FlashAttn-2   | 2 | ✅ | ✅ | ✅ | none | 18.9 s | $0.073 |
| Q3 linear attn   | 4 | ✅ | ✅ | ✅ | none | 35.2 s | $0.191 |

Three different preambles surfaced across the three runs (*"Let me synthesize a thorough answer"* / *"The retrieved chunks provide comprehensive, direct evidence…"* / *"Let me synthesize the key approaches"*) — every one landed before the marker and was discarded. The sentinel was emitted on **every** run, so the `Thought:`-strip fallback never fired.

### Interpretation

1. **Hypothesis (a) was falsified — the collision is structural.** Strengthening reasoning-first to restore iter-1 (attempt 2) revived the leak; the conditional version that killed the leak (attempt 1) dropped iter-1. Reasoning and answer share one turn and one text block, so no wording satisfies both rules at once. This see-saw is the core finding.

2. **The first code backstop also failed — for an instructive reason.** Stripping a *reasoning* marker (attempt 3) is whack-a-mole: told "no `Thought:` line in the answer," the model complied with the letter and narrated in a *new* form outside the marker. Enumerating forbidden preambles can never be complete.

3. **The working principle: mark where the answer BEGINS, not where reasoning is.** A positive sentinel (`===ANSWER===`) is robust to any preamble — known or novel — by construction, because the discard rule doesn't depend on the narration form. Confirmed across 3 questions / 3 distinct preambles, with the sentinel emitted reliably enough that the fallback was never needed. This is the belt-and-suspenders shape hypothesis (b) called for: prompt marks the boundary, code enforces the cut.

4. **Iter-1 reasoning is stochastic (2 of 3), and accepted as a known limitation.** Same prompt wording reasoned on Q2/Q3 but not Q1. It's the lowest-value reasoning in the trace (the opening query is self-documenting), and deterministic enforcement would require a re-call loop in code — complexity not worth the marginal transparency.

5. **Two incidental gains.** Separating reasoning from the answer also removed the trace-printer's double-`Thought:` labeling; and on Q3 the agent's dedicated sparse-attention retrieval round yielded *primary-source* citations for Longformer and BigBird, versus the secondhand (via Nyströmformer) citation seen in the Exp 3 structured run.

### Decision

- **Adopt the sentinel + `extract_answer()` as the final design** (`===ANSWER===`, take text after the last marker, fall back to `Thought:`-strip). The `Thought:` convention is retained for tool-turn transparency.
- **Capture artifacts:** re-run and save the three verification questions; stamp `prompt_version` into the run JSON (still outstanding from Exp 3).
- **Next milestone — agent-level eval.** With verbosity controlled (Exp 3) and the answer cleanly delimited by `===ANSWER===` (Exp 4), the `answer` field is now reliably machine-parseable — the precondition for automated answer-quality / groundedness scoring, which is distinct from Stage 1's retrieval-recall eval.

**Hypothesis for the agent-level eval:**
- A small groundedness metric (e.g. fraction of answer claims that carry a citation resolving to an actually-retrieved chunk) will surface failure modes that latency- and verbosity-focused experiments cannot — for example, the secondhand-sourcing quirk noted in Exp 3 finding #4.

## Experiment 5 — Agent answer-quality eval (LLM-as-judge, baseline)

**Date:** 2026-06-01
**Instrument:** `stage_2/eval/eval_set.json` (30 Q), `stage_2/eval/rubric.md`, `stage_2/judge.py` · **Result:** `stage_2/eval/judge.json` · **Full method + validation:** `stage_2/eval/eval_methodology.md`

### Setup

- **Eval set:** 30 questions, 3 difficulty tiers (10 simple / 10 medium / 10 hard), grounded in the 77-paper corpus. Simple = one distinctive paper; medium = within-cluster synthesis of 2-3 papers; hard = cross-cluster synthesis, version chains, or gap-finding.
- **Rubric:** 4 dimensions (factual accuracy, citation quality, completeness, coherence), 0-3 each (max 12), reported as per-dimension mean **by tier**.
- **Judge:** GPT-4.1 — deliberately **cross-family** from the Sonnet agent to avoid same-family self-preference bias. `temperature=0`, structured output (JSON-schema, scores constrained to enum 0-3). The judge verifies accuracy & citation **only against the chunks the agent actually cited** (parsed from inline `[Title (page N)]`, resolved against `chunks.json`); it uses its own expertise only for completeness & coherence.
- **Validation:** human spot-check of 5 answers (one per tier + both gap questions) against the judge's scores.

### Hypothesis

*(Committed in Exp 4's Decision.)* The eval will surface answer-quality failure modes the latency/verbosity experiments couldn't — particularly groundedness lapses — and scores will **degrade from simple → hard**.

### Result

| Tier | Factual acc | Citation | Completeness | Coherence | Total |
|------|:-----------:|:--------:|:------------:|:---------:|:-----:|
| Simple (n=10) | 3.0 | 3.0 | 3.0 | 3.0 | 12.0 |
| Medium (n=10) | 3.0 | 3.0 | 3.0 | 3.0 | 12.0 |
| Hard (n=10)   | 3.0 | 3.0 | 3.0 | 3.0 | 12.0 |

- **Every one of the 30 questions scored a perfect 12/12.** No tier degradation appeared.
- **The human spot-check disagreed with that perfection.** It caught a genuine groundedness overstatement on **h07** — the agent claimed KVQuant has a *"dedicated section titled 'Joint Weight and KV Cache Quantization' (§4.3)"* and cited KVQuant p9, which is actually a LongBench results table. The automated judge scored that same answer **10/12 in one run and 12/12 in a later run** — the signal was **not reproducible**.
- Reaching even this point required catching several harness bugs that were silently corrupting scores — see `eval_methodology.md` §Validation. The most consequential: the judge was, in an early run, grading against an **empty sources block** (a citation-page parser bug returned no pages, so 0 sources resolved with 0 flagged as unmatched) **and an empty rubric** (the system prompt was passed as the literal string `"JUDGE_SYSTEM"` instead of the variable). Both are silent failures — valid code, plausible-looking output, no error.

### Interpretation

1. **A uniform 12/12 is a ceiling/leniency finding, not a flawless agent.** A difficulty-stratified eval that fails nothing has no resolving power. Both are true: the agent's answers are genuinely strong (spot-check confirms s01/m03/h05 are excellent and well-grounded), **and** the eval cannot currently distinguish "good" from "great."
2. **The human spot-check — not the LLM judge — is the binding signal.** The committed hypothesis ("the eval will surface failure modes") *held*, but via the human: the spot-check caught h07's overstatement that the automated judge scored inconsistently.
3. **Gold-guidance contamination.** The judge is shown `key_points`/`gold_papers` (intended to anchor *completeness*). After the h07 key_point was corrected to "Atom is a genuine joint method," the judge flipped from flagging the KVQuant overstatement (10/12) to endorsing it (12/12). Guidance meant for completeness is leaking into the accuracy/citation score — a design flaw.
4. **Tier degradation did not appear**, but that is confounded by the ceiling effect — we cannot conclude the agent is tier-invariant.

### Decision

- **Report the agent as "top-of-rubric across all tiers, with the eval's ceiling effect explicitly flagged"** — do not claim perfection. The honest headline is about the *instrument*, not a perfect score.
- **Judge calibration is the key next step:** stricter rubric anchors / force the judge to name ≥1 weakness; **withhold `key_points`/`gold_papers` from the accuracy & citation pass** (show them only for completeness); consider a reasoning-model judge (o3-class); add **inter-run and inter-judge agreement** to quantify the nondeterminism seen on h07.
- **Keep the human spot-check as trusted ground truth** until the judge is calibrated.

### Note on prior runs (the harness-bug arc)

The first eval run scored every dimension oddly because of compounding silent failures, fixed in sequence: (1) citation **page parser** mis-indentation → empty `pages` → 0 sources resolved with 0 unmatched; (2) **system prompt** passed as a string literal → judge saw no rubric; (3) citation **comma-format** `[Title, page N]` not parsed (cost m06 all 14 citations → false 0/0); (4) **title normalization** for `H₂O` (unicode subscript) and `...`-truncated titles; (5) **rate limits** (uncapped `max_tokens` reserved ~16k tokens, inflating each request past the 30k TPM cap) fixed with `max_tokens`, a source budget, and 429 backoff; (6) the **source cap** itself (10) starved the judge on high-citation hard questions (h06: 40 cited pages) — raised to 50 once `max_tokens` was capped, which fit under the TPM limit without a paid tier upgrade. **The eval's first job turned out to be catching its own harness bugs — which is exactly why the human spot-check gate existed.**