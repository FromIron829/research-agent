# Stage 2 — Agent Answer-Quality Eval: Methodology

## Goal

Measure the **quality of the agent's answers** — not just whether retrieval found the right chunks (that's Stage 1's recall eval). This eval asks: given the agent's full ReAct loop, are its answers factually accurate, properly grounded in citations, complete, and coherent? It is scored by an LLM-as-judge against a rubric, validated by a human spot-check.

> **Headline finding up front:** as currently calibrated the eval **saturates** — it scores the agent 12/12 on all 30 questions. That is a *ceiling effect in the instrument*, not proof of a flawless agent. A human spot-check caught a real groundedness flaw (h07) that the automated judge scored inconsistently. **The human spot-check is the binding signal; the automated judge needs calibration.** See §Validation and §Limitations.

## 1. Eval set design (`eval_set.json`)

30 questions across **3 difficulty tiers**, calibrated using the corpus's natural topic clusters (quantization, KV-cache, attention kernels, speculative decoding, serving, long-context):

| Tier | Definition | Example |
|------|-----------|---------|
| **Simple** (10) | One distinctive paper answers it | *"What does AWQ identify as salient weights?"* |
| **Medium** (10) | Within-cluster synthesis of 2-3 papers | *"Compare H2O, SnapKV, and Scissorhands eviction signals."* |
| **Hard** (10) | Cross-cluster synthesis, version chains, or gap-finding | *"Which KV/quant techniques are compatible with self-speculation, and where do they conflict?"* |

Each question carries `gold_papers`, a `difficulty_rationale`, and topic-level `key_points` (coverage hints — *not* gold facts; the judge verifies specifics against sources). The simple tier reuses/adapts questions from the Stage 1 retrieval eval (already corpus-validated). Difficulty labels are **designer intent**; the agent's scores are the empirical check.

## 2. Rubric (`rubric.md`)

Four independent dimensions, 0-3 each (max 12): **factual accuracy, citation quality, completeness, coherence**, each with concrete anchors. Reported as per-dimension mean **by tier** — the tier breakdown is the intended diagnostic (a strong-simple / weak-hard profile would be the signal).

## 3. Judge architecture (`judge.py`)

- **Model: GPT-4.1 — deliberately cross-family** from the Sonnet 4.6 agent. Same-family judging (e.g. Opus judging Sonnet) risks self-preference bias; a different family removes that confound.
- **`temperature=0`** + **structured output** (JSON schema, scores constrained to `enum [0,1,2,3]`) so the judge returns clean integers, not free text.
- **Evidence-grounded verification:** the judge scores accuracy & citation **only against the chunks the agent actually cited**, not its own memory. Pipeline: parse inline `[Title (page N)]` citations → fuzzy-match the (often abbreviated) title to the manifest → fetch those page chunks from `chunks.json` → hand them to the judge. Completeness & coherence may use the judge's own expertise.
- **Decoupled generation/judging:** `run_eval.py` runs the agent over the 30 questions once (~$5) and saves `agent_answers.json`; `judge.py` reads that and scores it. This lets the judge be re-run cheaply during calibration without re-running the agent.

## Validation — what the harness caught (and why this section matters most)

The eval's *first* job turned out to be catching its own bugs. Each was a **silent failure** — valid code, plausible output, no exception — surfaced only by inspecting intermediate signals and by the human spot-check. In order:

1. **Empty sources, silently.** A citation page-parser bug (`elif` nested one level too deep) made single-page citations parse to *zero* pages, so 0 sources resolved — yet `n_unmatched` was also 0, so nothing looked wrong. The judge was grading every answer against an empty SOURCES block. **Fix + lesson:** log `n_sources`, not just `n_unmatched`; the right health metric makes a silent failure loud.
2. **Empty rubric, silently.** The judge's system prompt was passed as the literal string `"JUDGE_SYSTEM"` instead of the variable `JUDGE_SYSTEM` — valid Python, no error. The judge had no rubric. (Combined with #1, an early run scored with *neither* sources nor rubric.)
3. **Citation-format coverage.** The agent emits both `[Title (page N)]` and `[Title, page N]`; the parser handled only the parenthesized form, silently dropping all 14 of m06's comma-style citations → a false 0/0.
4. **Title normalization.** Cited titles diverge from the manifest: `H₂O` (unicode subscript) vs `H$_2$O` (LaTeX), and `...`-truncated long titles. Normalized both sides (fold subscripts, strip `$`/`_`/`…`) before matching.
5. **Rate limits.** An uncapped `max_tokens` made OpenAI reserve ~16k output tokens per request, pushing each call past the 30k TPM cap. Fixed with `max_tokens=700`, a per-source budget, and 429 backoff — no paid tier upgrade needed.
6. **Source cap vs evidence.** The interim `MAX_SOURCES=10` starved the judge on high-citation hard questions (h06 cited 40 pages → judge saw 10 → marked the rest "unsupported"). Once `max_tokens` was capped, the request budget allowed `MAX_SOURCES=50`, which fit under the TPM limit and resolved the artifact.

**Human spot-check.** 5 answers hand-scored against the judge. The judge tracked a careful read closely on 4 — but on **h07** the human caught a groundedness overstatement (a fabricated/mis-cited "joint quantization section") that the automated judge scored **10/12 then 12/12 across runs** — non-reproducible. This is why the human check is the binding signal.

## Known limitations

- **Ceiling effect.** The judge scores all 30 answers 12/12; it cannot currently distinguish "good" from "great." A discriminating eval must be able to fail strong answers. Calibration (stricter anchors, forced weakness-finding, a reasoning-model judge) is the priority next step.
- **Gold-guidance contamination.** The judge is shown `key_points`/`gold_papers` (meant for completeness); evidence (h07's 10→12 flip after a key_point edit) suggests this leaks into the accuracy/citation score. Fix: withhold gold guidance from the accuracy/citation pass.
- **Judge nondeterminism.** Even at `temperature=0`, GPT-4.1 gave different h07 scores across runs. Inter-run agreement is not yet measured.
- **Single judge.** Cross-family reduces self-preference but one judge is still one opinion; no inter-judge agreement measured. Two LLM judges agreeing would still not equal human ground truth.
- **Judge sees only cited sources.** It can verify what was cited; it cannot fully detect a *relevant omission* the agent never retrieved. Completeness partly relies on the judge's own expertise.
- **Figure noise in chunks.** `pymupdf4llm` leaves `==> picture … omitted` placeholders and garbled "picture text" in some chunks; observed to dilute but not break scoring when clean prose co-occurs. Not remediated (would require Stage 1 re-extraction).

## Audit trail

- Eval set: `eval/eval_set.json` (30 Q with tiers, gold papers, key_points) · Rubric: `eval/rubric.md`
- Judge: `judge.py` · Runner: `run_eval.py` · Agent answers: `eval/agent_answers.json` · Scores: `eval/judge.json`
- Experiment write-up: `results/EXPERIMENTS.md` → Experiment 5
