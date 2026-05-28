# Stage 1 — Retrieval over efficient-inference papers

A RAG retrieval system over 77 arXiv papers on efficient LLM inference. Builds a vector index (and a reranker layer), measures recall@k against a 50-question hand-curated eval, and logs results per experiment.

## Directory structure

```
stage_1/
├── *.py                       Pipeline scripts (run in the order below)
├── data/                      Generated pipeline artifacts
│   ├── manifest.json          arXiv metadata for the 77 papers (from arXiv API)
│   ├── extracted.json         Per-page markdown text from the PDFs
│   ├── chunks.json            ~1616 token-aware chunks with metadata
│   └── chroma_db/             Persistent ChromaDB vector index (gitignored)
├── eval/                      Eval setup and audit trail
│   ├── eval.json              50 retrieval questions, paper-level gold
│   ├── eval_methodology.md    Eval construction methodology + limits
│   ├── eval_candidates.json   77 LLM-drafted candidates (audit trail)
│   └── eval_judged.jsonl      LLM-judge output on the candidates (audit trail)
└── results/                   Per-experiment recall@k numbers
    ├── baseline_vector.json   text-embedding-3-small, vector-only
    └── rerank_cohere.json     vector → Cohere rerank-v3.5
```

## Pipeline order

```
PDF files (../paper_pdfs/)
  ↓ build_manifest.py        → data/manifest.json
  ↓ extract.py               → data/extracted.json
  ↓ chunk.py                 → data/chunks.json
  ↓ embed_and_index.py       → data/chroma_db/

eval/eval.json               (hand-curated; see eval/eval_methodology.md)
  ↓ score.py                 → results/baseline_vector.json
  ↓ score_rerank.py          → results/rerank_cohere.json
```

`retrieve.py` and `rerank.py` are imported by the score scripts; they can also be invoked directly for ad-hoc queries:

```bash
uv run python stage_1/retrieve.py "how does AWQ protect salient weights"
uv run python stage_1/rerank.py     # demo against q17 / EAGLE-2
```

## Reproduce from scratch

Assumes `paper_pdfs/` contains the 77 PDFs and `.env` has `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `COHERE_API_KEY`.

```bash
uv sync                                       # install deps from pyproject.toml

# 1. Build the corpus
uv run python stage_1/build_manifest.py       # ~1 min — arXiv API
uv run python stage_1/extract.py              # ~5 min — pymupdf4llm
uv run python stage_1/chunk.py                # seconds — token-aware splitting
uv run python stage_1/embed_and_index.py      # ~3-5 min — OpenAI embeddings (~$0.03)

# 2. (Optional — eval.json already committed) Regenerate candidates
uv run python stage_1/generate_candidates.py  # ~3 min — Claude Haiku (~$0.05)

# 3. Run experiments
uv run python stage_1/score.py                # ~30 sec — vector-only
uv run python stage_1/score_rerank.py         # ~6 min — Cohere rerank (7s/q rate limit)
```

## Current results

Stored in `results/`. **Full experiment narrative in [`results/EXPERIMENTS.md`](results/EXPERIMENTS.md)** — each experiment documented with setup, hypothesis, result, interpretation, and decision.

| # | Method               | n  | recall@5 | recall@10 | Notes |
|---|---------------------|----|----------|-----------|-------|
| 1 | vector-only         | 20 | 0.95     | 0.95      | Exit criterion (≥0.70) cleared; one failure q17 (EAGLE-2) |
| 2 | + Cohere rerank-v3.5| 20 | 0.95     | 0.95      | Fixed q17, broke q16 (H2O) — one-for-one cluster-failure swap |
| 3a| vector-only         | 50 | **0.96** | 0.96      | Expanded eval; same pattern reproduces |
| 3b| + Cohere rerank-v3.5| 50 | **0.96** | 0.96      | Identical headline; q17/q16 swap reproduces; **q34 (EAGLE) fails on both** — structural candidate-pool limit |
| 4 | + BM25 + RRF        | 50 | 0.96      | **0.98**       | First numeric improvement — recall@10 lifted by rescuing q16 into top-10 candidate pool. q34 still fails (multi-term BM25 weighting defeated by Online SD's higher 'draft model' density). |

**Key finding:** at n=50 with ±3.5% CI, vector and vector+rerank are statistically equivalent. The remaining 2/50 failures (q34 specifically) point to embedding-vocabulary limits rather than ranking limits — see EXPERIMENTS.md Experiment 3 for the interpretation.

## Known limitations

- **Paper-level gold only.** `gold_chunk_ids` empty across all 50 questions.
- **No BM25/hybrid yet.** Pure vector + cross-encoder reranker.
- **n=50** gives ~3.5% resolution on recall numbers; smaller improvements are not statistically distinguishable.
