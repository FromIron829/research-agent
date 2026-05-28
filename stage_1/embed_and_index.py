import json
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "stage_1" / "data" / "chunks.json"
CHROMA_DIR = PROJECT_ROOT / "stage_1" / "data" / "chroma_db"
COLLECTION_NAME = "papers"

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

def main():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    openai_client = OpenAI()
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        response = openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
        )
        embeddings = [d.embedding for d in response.data]

        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "arxiv_id": c["arxiv_id"],
                "paper_title": c["paper_title"],
                "primary_category": c["primary_category"],
                "year": c["year"],
                "page": c["page"],
            } for c in batch],
        )
        print(f"    Embedded {i + len(batch):4d}/{len(chunks)}")
    
    print(f"\nDone. Collection '{COLLECTION_NAME}' has {collection.count()} items.")
    print(f"Persisted to {CHROMA_DIR}")

if __name__ == "__main__":
    main()

