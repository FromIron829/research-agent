# Stage 1 — Experiment Log

Chronological log of every retrieval experiment in Stage 1. Each entry follows the same shape: **Setup → Hypothesis → Result → Interpretation → Decision.** The intent is to leave a paper trail that anyone (including future-me) can follow without re-deriving the reasoning.

Result JSON files live alongside this doc; per-question detail (top-5 retrieved, scores) is in those files.

---

## Experiment 1 — Vector baseline (n=20)

**Date:** 2026-05-26
**Result file:** `baseline_vector.json` (overwritten by Exp 3 — see eval methodology for original n=20 numbers preserved by ID)

**Setup**
- Retriever: OpenAI `text-embedding-3-small` → ChromaDB (cosine), top-k = 10
- Corpus: 1616 chunks (800-tok recursive split, 100-tok overlap, 200-char min-length filter) from 77 efficient-inference papers
- Eval: 20 hand-curated questions across 6 sub-topics, 10 specific / 10 semantic, paper-level gold

**Hypothesis**
A clean vector-only baseline over a well-curated corpus should clear the ≥0.70 recall@5 exit criterion comfortably; will probably land 0.50–0.80.

**Result**
- Overall recall@5 = **0.95** (19/20), recall@10 = 0.95
- Specific 1.00, Semantic 0.90
- Single failure: **q17 (EAGLE-2)** — semantic / speculative-decoding, top-5 all other SD papers

**Interpretation**
Stronger than expected. Three drivers: strong corpus curation (papers are mostly distinct), markdown-aware chunking via pymupdf4llm, and `text-embedding-3-small` being a genuinely strong model. The single failure is a **within-cluster confusion** problem — the question is too generic to disambiguate EAGLE-2 from its SD-cluster neighbors — not a retrieval-method problem.

**Decision**
Try a cross-encoder reranker. Hypothesis: within-cluster ambiguity is exactly where a reranker (which reads query+chunk jointly) should beat a bi-encoder (which compares vector distances).

---

## Experiment 2 — Vector + Cohere rerank-v3.5 (n=20)

**Date:** 2026-05-26
**Result file:** `rerank_cohere.json` (overwritten by Exp 3)

**Setup**
- Vector retrieval as Exp 1, but `initial_k = 20` (wider candidate set)
- Cohere `rerank-v3.5` rescores all 20 candidates, returns top 10
- Same 20-question eval, same recall@k

**Hypothesis**
Reranker will specifically fix the q17 cluster-confusion failure → recall@5 climbs to ~1.00.

**Result**
- Overall recall@5 = **0.95** (unchanged from baseline)
- **q17 fixed** ✓ — but **q16 (H2O) now fails** ✗
- Topic-level: speculative-decoding went 0.67→1.00, kv-cache went 1.00→0.67. **One-for-one swap.**

**Interpretation**
The reranker did exactly what was hypothesized for q17. But it introduced an equivalent error elsewhere (q16: KIVI promoted over H2O — surface phrase overlap on "reduce KV cache memory" outweighed the deeper "attention persistence" semantics). **Rerankers are not strictly improvements; they make different errors.** Also: at n=20 the ±5% CI cannot statistically distinguish 0.95 from 0.95 — so even genuine small improvements would be invisible.

**Decision**
Eval is at measurement ceiling. Two paths forward: (a) expand to n=50 for resolution, (b) accept and stop. Picked (a) — driven by the data, since the failures point to specific cluster weaknesses that more questions can probe.

---

## Experiment 3 — Eval expansion + re-run of Exp 1 & 2 (n=50)

**Date:** 2026-05-26
**Result files:** `baseline_vector.json`, `rerank_cohere.json` (current versions)

**Setup**
- Added 30 questions (q21–q50) targeting observed failure clusters:
  - **9 KV cache** (Scissorhands, GEAR, Ada-KV, SnapKV, MiniCache, LoRC, ChunkKV, CacheGen, H2O-specific) — the cluster that broke under rerank
  - **5 speculative decoding** (MagicDec, Hydra, EAGLE-3, Online SD, EAGLE) — completes the EAGLE-family
  - **4 attention** (FA3, Ring, Striped, MQA)
  - **5 quantization** (SpQR, OmniQuant, QuIP#, SqueezeLLM, AQLM)
  - **4 serving** (Splitwise, AlpaServe, FastGen, 2nd Sarathi)
  - **3 long-context** (Linformer, Performer, BigBird)
- Final eval: 50 questions, 22 specific / 28 semantic
- Re-ran Exp 1 and Exp 2 against the larger eval

**Hypothesis**
- Both methods will drop from 0.95 (eval is harder)
- Vector vs. rerank should now show a measurable difference if reranking actually helps

**Result**

| Method | recall@5 | recall@10 | Specific | Semantic | Failures |
|--------|----------|-----------|----------|----------|----------|
| Vector | **0.96** | 0.96 | 1.00 | 0.93 | q17 (EAGLE-2), q34 (EAGLE) |
| + Rerank | **0.96** | 0.96 | 1.00 | 0.93 | q16 (H2O), q34 (EAGLE) |

**Interpretation**
Three real findings:
1. **Methods remain statistically equivalent** (literally identical headline numbers) at n=50 with ±3.5% CI. The earlier null result was not a small-sample artifact.
2. **The q17 ↔ q16 swap reproduces.** Reranker still fixes q17 (EAGLE-2) and still breaks q16 (H2O). Two replications now. The "rerankers trade errors" pattern is reproducible for this corpus.
3. **q34 fails on BOTH methods** (new at n=50). EAGLE (2401.15077v3) doesn't appear in the top-10 vector candidates at all — the reranker cannot help when the right paper isn't a candidate. This is a **structural candidate-pool limit**, not a ranking limit. The embedding cannot map "predict feature representations" (the user's phrasing) to EAGLE's chunks' vocabulary.

**Decision**
Run BM25 + reciprocal rank fusion as the canonical third method. Hypothesis: BM25 may rescue q34 because EAGLE's title is *"Speculative Sampling Requires Rethinking Feature Uncertainty"* — "feature" matches the query directly. Result will complete the three-way comparison regardless of outcome.

---

## Experiment 4 — Vector + BM25 + Reciprocal Rank Fusion (n=50)

**Date:** 2026-05-26
**Result file:** results/hybrid_bm25_rrf.json

**Setup**
- Vector (top-50) + BM25 (top-50), fused via RRF (k=60), top-10 returned
- BM25: rank-bm25 BM250Okapi, regex tokenizer (lowercased, keeps numbers/hyphens/dots)
- Same 50-question eval

**Hypothesis**
- Overall: roughly equivalent to vector-only (~0.96) - both prior methods nulled.
- Q34 (EAGLE): MAY be resued - EAGLE's title contains "Feature Uncertainty" adn the query contains "feature"; BM25 should surface EAGLE near the top of of its ranking.
- New failures may appear if BM25 surfaces wrong papers strongly on lexical match.

**Result**

| Method | recall@5 | recall@10 | Failures @ 5 | New @ 10 |
|--------|----------|-----------|--------------|-----------|
| Vector (Exp 3a) | 0.96 | 0.96 | q17, q34 | - |
| + Rerank (Exp 3b) | 0.96 | 0.96 | q16, q34 | - |
| + BM25 + RRF | **0.96** | **0.98** | q16, q34 | q16 now in top-10 |

- Diagnostic on q34: EAGLE chunks contain "feature" 83x vs. Online SD 1x, but but Online SD contains "draft model" 129× vs. EAGLE 41×. BM25's multi-term TF-IDF accounting favored Online SD despite EAGLE's dominance on the rarer query term.

**Interpretation**
1. **Recall@5 ceiling is real.** All three retrieval methods (vector, +rerank, +BM25+RRF) tie at 0.96 with the eval's ±3.5% resolution. For this corpus at top-5, no method dominates.
2. **BM25 hybrid genuinely helps at recall@10** - small (2pp) but the first non-null result of the stage. The mechanism: RRF pulls H2O (which rerank had lost entirely) back into the candidate set at rank 6-10. Hybrid is additive at the wider-k tier even when @5 doesn't move.
3. **q34 is structurally intractable for these methods.** Multi-term BM25 weighting can be defeated by an unrelated paper having higher density on shared query terms - a failure mode that single-term reasoning hides. Would require query rewriting (expand with synonyms like "hidden states") or a domain-specific reranker to fix.

**Decision**
Close Stage 1. Three retrieval methods evaluated rigorously at n=50, a clear three-way comparison documented, both null and partial-positive results captured honestly. The recall@5 = 0.96 baseline is well past the ≥0.70 exit criterion. Move to Stage 2 (agent layer); revisit retrieval if and when downstream signal (agent quality) demands it.

---

## Running summary

| # | Method | n | recall@5 | recall@10 |
|---|--------|---|----------|-----------|
| 1 | vector-only | 20 | 0.95 | 0.95 |
| 2 | + Cohere rerank | 20 | 0.95 | 0.95 |
| 3a | vector-only | 50 | **0.96** | 0.96 |
| 3b | + Cohere rerank | 50 | **0.96** | 0.96 |
| 4 | + BM25 + RRF | 50 | 0.96 | 0.98 |
