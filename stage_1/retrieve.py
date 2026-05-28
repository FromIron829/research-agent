import json
from pathlib import Path

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "stage_1" / "data" / "chroma_db"
COLLECTION_NAME = "papers"
EMBED_MODEL = "text-embedding-3-small"

_openai_client = OpenAI()
_collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(COLLECTION_NAME)

def embed_query(query: str):
    response = _openai_client.embeddings.create(model=EMBED_MODEL, input=[query])
    return response.data[0].embedding

def retrieve(query: str, k: int = 10):
    query_emb = embed_query(query)
    result = _collection.query(query_embeddings=[query_emb], n_results=k)

    chunks = []
    for chunk_id, doc, meta, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        chunks.append({
            "chunk_id": chunk_id,
            "arxiv_id": meta["arxiv_id"],
            "paper_title": meta["paper_title"],
            "page": meta["page"],
            "text": doc,
            "score": 1.0 - distance
        })
    return chunks

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "how does FlashAttention reduce memory IO"
    print(f"Query: {query}\n")
    for i, c in enumerate(retrieve(query, k=5), 1):
        print(f"{i}. [{c['score']:.3f}] {c['paper_title']} (p.{c['page']})")
        print(f"    {c['text'][:200]}\n")