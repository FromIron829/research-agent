import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from retrieve import retrieve as vector_retrieve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "stage_1" / "data" / "chunks.json"

_TOKEN_RE = re.compile(r"\b\w[\w.-]*\b")

def tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())

with open(CHUNKS_PATH, encoding="utf-8") as f:
    _chunks = json.load(f)

_corpus_tokens = [tokenize(c["text"]) for c in _chunks]
_bm25 = BM25Okapi(_corpus_tokens)

def bm25_retrieve(query: str, k: int = 10):
    query_tokens = tokenize(query)
    scores = _bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [{
        "chunk_id": _chunks[i]["chunk_id"],
        "arxiv_id": _chunks[i]["arxiv_id"],
        "paper_title": _chunks[i]["paper_title"],
        "page": _chunks[i]["page"],
        "text": _chunks[i]["text"],
        "score": float(scores[i]),
    } for i in top_indices]

def retrieve_hybrid(query: str, k: int = 10, initial_k: int = 50, rrf_k: int = 60):
    vector_hits = vector_retrieve(query, initial_k)
    bm25_hits = bm25_retrieve(query, initial_k)

    rrf_scores = {}
    chunks_by_id = {}

    for rank, c in enumerate(vector_hits, start=1):
        rrf_scores[c["chunk_id"]] = rrf_scores.get(c["chunk_id"], 0.0) + 1.0 / (rrf_k + rank)
        chunks_by_id[c["chunk_id"]] = dict(c)
        chunks_by_id[c["chunk_id"]]["vector_rank"] = rank
    
    for rank, c in enumerate(bm25_hits, start=1):
        rrf_scores[c["chunk_id"]] = rrf_scores.get(c["chunk_id"], 0.0) + 1.0 / (rrf_k + rank)
        chunks_by_id.setdefault(c["chunk_id"], dict(c))
        chunks_by_id[c["chunk_id"]]["bm25_rank"] = rank
    
    top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    fused = []
    for cid in top_ids:
        chunk = chunks_by_id[cid]
        chunk["rrf_score"] = rrf_scores[cid]
        fused.append(chunk)
    return fused

if __name__ == "__main__":
    # The Stage 1 hard case — q34 (EAGLE). Run all three side-by-side.
    query = "How can speculative decoding be improved by having the draft model predict the target model's internal feature representations instead of its tokens?"
    print(f"Query: {query}\n")

    print("--- BM25 only top-5 ---")
    for i, c in enumerate(bm25_retrieve(query, k=5), 1):
        print(f"{i}. [{c['score']:6.2f}] {c['paper_title'][:60]} (p.{c['page']})")

    print("\n--- Vector only top-5 ---")
    for i, c in enumerate(vector_retrieve(query, k=5), 1):
        print(f"{i}. [{c['score']:.3f}] {c['paper_title'][:60]} (p.{c['page']})")

    print("\n--- Hybrid (RRF) top-5 ---")
    for i, c in enumerate(retrieve_hybrid(query, k=5, initial_k=50), 1):
        v = c.get('vector_rank', '-')
        b = c.get('bm25_rank', '-')
        print(f"{i}. [rrf={c['rrf_score']:.4f} | v={v} b={b}] {c['paper_title'][:50]} (p.{c['page']})")
