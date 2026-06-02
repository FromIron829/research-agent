# Stage 2 — Agent Layer

A ReAct research agent over the Stage 1 RAG corpus (77 arXiv papers on efficient LLM inference). The agent decides *when* and *what* to retrieve, refines queries across iterations, synthesizes a cited answer, and is instrumented end-to-end for latency, cost, and answer quality.

Stage 1 built and measured **retrieval** in isolation. Stage 2 wraps it in an agent and asks two questions: *can we make the loop fast and cheap?* (Experiments 1-4) and *are the answers actually good?* (Experiment 5).

## Architecture

```
question → ReAct loop (Claude Sonnet 4.6)
             ├─ retrieve tool  → Stage 1 hybrid search (vector + BM25 + RRF)
             ├─ reasoning-first ("Thought:" before each tool call)
             ├─ prompt caching (rolling cache_control breakpoint)
             └─ ===ANSWER=== sentinel → clean, cited synthesis
```

- `agent.py` — the loop: tool-use, per-iteration timing/token trace, prompt caching, stop-reason handling, answer extraction.
- `eval/` — the agent answer-quality eval (set, rubric, methodology).
- `judge.py`, `run_eval.py` — LLM-as-judge harness.
- `results/EXPERIMENTS.md` — full experiment log (Setup → Hypothesis → Result → Interpretation → Decision).

## Experiments

| # | Focus | Headline result |
|---|-------|-----------------|
| 1 | Loop baseline (n=1) | 118 s, $0.32; surfaced silent `max_tokens` truncation + zero-reasoning bugs |
| 2 | Prompt caching + streaming | **−48% latency, −47% cost** (118→62 s, $0.32→$0.17); verified rolling cache breakpoint |
| 3 | Structured synthesis (A/B, 3 Q) | **−47% synthesis tokens**, hypothesis confirmed; surfaced reasoning-preamble leak |
| 4 | Eliminate preamble leak | Prompt-only fixes falsified (structural collision); solved with a positive `===ANSWER===` sentinel — robust by construction |
| 5 | **Answer-quality eval** | LLM-judge (GPT-4.1, cross-family) over 30 tiered questions — see honest caveat below |

## Eval result (Experiment 5) — read the caveat

The LLM-as-judge scored the agent **12/12 on all 30 questions across every tier.** This is reported honestly as a **ceiling effect in the eval, not a flawless agent**:

- A difficulty-stratified eval that fails nothing has no resolving power.
- A **human spot-check** caught a real groundedness overstatement (question h07: a fabricated/mis-cited "joint quantization" claim) that the automated judge scored **inconsistently (10/12 then 12/12 across runs).**
- **The human spot-check is the binding quality signal; calibrating the judge to discriminate is the priority next step.**

The agent's answers *are* genuinely strong (the spot-check confirms it) — but "12/12 everywhere" measures the instrument's limits, not perfection. Full design, validation, and limitations: [`eval/eval_methodology.md`](eval/eval_methodology.md).

## Key engineering challenges

- **Treating a perfect eval as a bug.** A 30/30 perfect score is implausible for any real system; recognizing the ceiling effect (rather than declaring victory) and keeping the human spot-check as ground truth is the core lesson.
- **Silent failures in the judge harness.** The judge once scored answers against an *empty* sources block (a page-parser bug yielding 0 sources with 0 flagged unmatched) *and* an *empty* rubric (the system prompt passed as a string literal `"JUDGE_SYSTEM"`). Both are valid code with no error — caught only by logging the right health metric (`n_sources`) and by the human check.
- **The preamble see-saw → sentinel (Exp 4).** Prompt-only attempts to stop reasoning leaking into the final answer kept trading one failure for another, proving the collision was *structural* (reasoning and answer share one turn). Solved deterministically with a positive answer marker the loop splits on.
- **Latent agent bug surfaced by a harder test.** A `cache_control` marker-accumulation bug (`isinstance` checking `dict` instead of `list`) only triggered on a 6-iteration question — earlier experiments never exceeded 4 iterations, so it had hidden until the eval's hard questions exercised the loop harder.

## Run

```bash
python stage_2/agent.py "your question"      # run the agent once
python stage_2/run_eval.py                   # generate answers for the 30-Q eval (~$5)
python stage_2/judge.py                      # score answers → eval/judge.json
python stage_2/spotcheck.py s01 m03 h07      # human spot-check helper
```

## Next steps

- **Calibrate the judge** so the eval discriminates: stricter rubric anchors / force ≥1 named weakness; withhold `key_points`/`gold_papers` from the accuracy pass (they currently leak into it); consider a reasoning-model judge; measure inter-run / inter-judge agreement.
- Re-run the agent quality eval once the judge can fail a strong answer, and report the (no-longer-saturated) tier breakdown.
