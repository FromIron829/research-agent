import json

from dotenv import load_dotenv
import cohere

from retrieve import retrieve as vector_retrieve

load_dotenv()

_cohere_client = cohere.ClientV2()

RERANK_MODEL = "rerank-v3.5"

def retrieve_with_rerank(query: str, k: int = 10, initial_k: int = 20):
    candidates = vector_retrieve(query, initial_k)

    response = _cohere_client.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=[c["text"] for c in candidates],
        top_n=k
    )

    reranked = []
    for r in response.results:
        chunk = dict(candidates[r.index])
        chunk["vector_score"] = chunk.pop("score")
        chunk["rerank_score"] = r.relevance_score
        reranked.append(chunk)
    return reranked

if __name__ == "__main__":
    # Demo against the exact q17 question that failed vector-only
    query = "How can the draft structure in speculative decoding adapt based on context rather than staying fixed?"
    print(f"Query: {query}\n")
    for i, c in enumerate(retrieve_with_rerank(query, k=5), 1):
        print(f"{i}. [rerank={c['rerank_score']:.3f} | vec={c['vector_score']:.3f}] "
              f"{c['paper_title'][:55]} (p.{c['page']})")
        print(f"   {c['text'][:180]}\n")