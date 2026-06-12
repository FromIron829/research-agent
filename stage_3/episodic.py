import sys
import uuid
import time
from pathlib import Path
import chromadb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_1"))
from retrieve import embed_query, CHROMA_DIR

_conversations = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_or_create_collection("conversations", metadata={"hnsw:space": "cosine"})

def remember_turn(thread_id: str, question: str, answer: str, tenant: str = "public"):
    ts = time.time()
    _conversations.upsert(
        ids=[f"{thread_id}-{uuid.uuid4().hex}"],
        embeddings=[embed_query(f"{question}\n{answer}")],
        documents=[question],
        metadatas=[{"thread_id": thread_id, "ts": ts, "tenant": tenant,
                    "question": question, "answer": answer[:2000]}],
    )

def recall(query: str, k: int = 3, tenant: str = "public"):
    if _conversations.count() == 0:
        return []
    res = _conversations.query(query_embeddings=[embed_query(query)], n_results=k,
                               where={"tenant": tenant})
    return [{"question": m["question"], "answer": m["answer"], "ts": m["ts"], "score": 1.0 - d}
            for m, d in zip(res["metadatas"][0], res["distances"][0])]