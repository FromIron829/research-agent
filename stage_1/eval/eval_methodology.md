# Retrieval Evaluation Set — Methodology

## Goal

A retrieval evaluation set used to measure recall@5 / recall@10 for the Stage 1 retriever against a corpus of 77 efficient-LLM-inference papers. Drives both the choice of retrieval strategy (vector / hybrid / rerank) and the size-of-effect we can detect.

## Eval set evolution

This eval was built in two stages, with the second stage triggered by observed failure modes — not by a predetermined plan.

### v1 — initial n=20

1. **LLM candidate generation.** Drafted 77 candidate questions (one per paper) with Claude Haiku 4.5 against each paper's abstract. System prompt asked for natural user voice, ~50/50 specific/semantic difficulty, explicit warnings against answer-aware leakage.
2. **LLM judge pass.** Sent the 77 candidates to ChatGPT as judge-and-fixer. Judge flagged paper-name leaks and "this paper"-style phrasing; proposed rewrites.
3. **Human curation pass.** Reviewed candidates + judge fixes + source abstracts. Picked 10 keepers. Hand-wrote 10 additional specific-difficulty questions using paper-distinctive acronyms (GPTQ, FlashAttention, PagedAttention, AWQ, SmoothQuant, QLoRA, Medusa, BitNet b1.58, etc.) to rebalance the distribution.

### v2 — expansion to n=50 (failure-driven)

After running both the vector-only baseline AND a vector+rerank experiment against v1, the results showed:

- **Vector baseline:** 19/20 hit. Missed q17 (EAGLE-2) — within-cluster confusion in speculative decoding.
- **Vector + Cohere rerank:** 19/20 hit. Fixed q17, but BROKE q16 (H2O) — within-cluster confusion swapped to the KV-cache cluster.
- **Net change:** ZERO at n=20. Statistically indistinguishable. The eval hit its measurement ceiling — at n=20, the 95% CI on a 0.95 estimate is roughly [0.85, 1.00].

Diagnosis: v1 lacked resolution to distinguish retrieval techniques, and lacked enough within-cluster confusables to stress-test the systems. v2 was designed to fix both.

**v2 added 30 questions** (q21–q50), deliberately targeting observed failure modes:

| Cluster              | Added | Rationale |
|---------------------|-------|-----------|
| KV cache            | 9     | Cluster that broke under rerank. Covers Scissorhands, GEAR, Ada-KV, SnapKV, MiniCache, LoRC, ChunkKV, CacheGen, plus a specific-version H2O question paired with q16's semantic version. |
| Speculative decoding| 5     | Cluster that originally failed pre-rerank. Adds MagicDec, Hydra, EAGLE-3, Online SD, EAGLE — completing the EAGLE family (EAGLE/EAGLE-2/EAGLE-3 each get their own question for version-disambiguation testing). |
| Attention kernels   | 4     | Completes FA1/FA2/FA3 version chain; adds Ring + Striped Attention pair (within-cluster causal-vs-non-causal distinction); adds MQA as disambiguation partner for GQA. |
| Quantization        | 5     | Adds within-cluster confusables: SpQR vs. SqueezeLLM (both dense+sparse), OmniQuant vs. one-shot, QuIP# vs. QuIP version chain, AQLM as additive-codebook outlier. |
| Serving             | 4     | Adds Splitwise (heterogeneous-hardware angle vs. DistServe), AlpaServe (statistical multiplexing), DeepSpeed-FastGen (Dynamic SplitFuse), and a second Sarathi question (same-paper-different-framing consistency check). |
| Long context        | 3     | Adds Linformer / Performer / BigBird — three different sparse-attention approaches that are easy to confuse semantically. |

## Final distribution (n=50)

| Sub-topic            | Count |
|----------------------|-------|
| KV cache             | 12    |
| Quantization         | 10    |
| Attention kernel/arch| 8     |
| Serving              | 8     |
| Speculative decoding | 8     |
| Long context         | 4     |
| **Total**            | **50** |

**Difficulty:** 22 specific / 28 semantic. Semantic-skewed (v1 had specific at 100% — no measurement room there).

## What this eval can now measure

With n=50, the 95% CI on a 0.95 recall@5 estimate tightens to roughly [0.88, 0.99] — about **±3.5% resolution**. That's enough to distinguish methods that differ by 5–7 percentage points. Improvements in the 1–2 pp range still won't be statistically distinguishable, but those are also probably not meaningfully real at the engineering level.

## What I caught at each layer (cumulative)

- **Generator (Haiku):** 95/5 difficulty skew vs. 50/50 prompt target. Answer-name leakage in ~3% of candidates.
- **Judge (ChatGPT):** 0/77 rejects (over-generous). ~30% of "fixes" introduced new distinctive-term leaks while removing other issues.
- **Curation pass (human):** rejected the leaks and the too-generic candidates; hand-wrote specific-difficulty questions to rebalance.
- **v2 expansion:** went beyond curation — wrote 30 new questions deliberately targeting failure modes observed in actual baseline + rerank runs.

## Question design constraints

Every question satisfies:
- Sounds like a natural user query (not encyclopedic Q&A, no "this paper")
- Has a clear best paper (or is explicitly multi-gold, like q10)
- For **semantic** items: contains no paper-distinctive vocabulary
- For **specific** items: uses the paper's distinguishing acronym/term naturally

`notes` field on every question records the failure mode it's designed to probe — which within-cluster discrimination it's testing, what should rank where.

## Known limitations

- **Paper-level gold only.** `gold_chunk_ids` is empty across all 50 questions. Chunk-level recall is not yet measurable. Acceptable for Stage 1 baseline; revisit if needed.
- **One multi-gold question** (q10) credits a hit on either of two valid papers.
- **Topic skew matches corpus skew.** KV cache, quantization, and attention dominate by count, reflecting the corpus distribution. Sub-topic-stratified recall reporting compensates.
- **Eval scale.** n=50 has ~3.5% resolution. Improvements <5pp are not statistically distinguishable.
- **"Hard" is heuristic.** Difficulty labels reflect designer intent, not measured retrieval difficulty. Some "semantic" questions may be easier than expected; some "specific" may be harder. The retrieval results themselves are the empirical ground truth.

## Audit trail

- Drafts: `stage_1/eval_candidates.json` (77 LLM-drafted), `stage_1/rag_eval_questions_judged_fixed.jsonl` (LLM judge output)
- Final eval: `stage_1/eval.json` (50 questions, with per-question `notes` documenting design intent)
- Baseline result: `stage_1/baseline_vector.json` (against v1, n=20) — needs re-running against v2
- Rerank result: `stage_1/results/rerank_cohere.json` (against v1, n=20) — needs re-running against v2
