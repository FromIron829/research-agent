import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_PATH = PROJECT_ROOT / "stage_1" / "data" / "extracted.json"
OUTPUT_PATH = PROJECT_ROOT / "stage_1" / "data" / "chunks.json"

CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100
EMBEDDING_TOKENIZER = "cl100k_base"

def main():
    with open(EXTRACTED_PATH, encoding="utf-8") as f:
        extracted = json.load(f)
    
    documents = []
    for arxiv_id, paper in extracted.items():
        for page in paper["pages"]:
            documents.append(
                Document(
                    page_content=page["text"],
                    metadata={
                        "arxiv_id": arxiv_id,
                        "paper_title": paper["title"],
                        "primary_category": paper["primary_category"],
                        "year": paper["year"],
                        "page": page["page"]
                    },
                )
            )
    print(f"Built {len(documents)} page-level documents from {len(extracted)} papers")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=EMBEDDING_TOKENIZER,
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )
    spilt_docs = splitter.split_documents(documents)
    print(f"Split into {len(spilt_docs)} chunks")

    chunks = []
    per_paper_counter = {}
    for doc in spilt_docs:
        arxiv_id = doc.metadata["arxiv_id"]
        idx = per_paper_counter.get(arxiv_id, 0)
        per_paper_counter[arxiv_id] = idx + 1
        chunks.append({
            "chunk_id": f"{arxiv_id}_{idx:04d}",
            "arxiv_id": arxiv_id,
            "paper_title": doc.metadata["paper_title"],
            "primary_category": doc.metadata["primary_category"],
            "year": doc.metadata["year"],
            "page": doc.metadata["page"],
            "text": doc.page_content,
        })
    
    MIN_CHUNK_CHARS = 200

    chunks = [c for c in chunks if len(c["text"]) >= MIN_CHUNK_CHARS]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    chunks_per_paper = list(per_paper_counter.values())
    char_lens = [len(c["text"]) for c in chunks]
    print(f"\nSaved {len(chunks)} chunks to {OUTPUT_PATH}")
    print(f"Chunks per paper: min={min(chunks_per_paper)}, "
          f"avg={sum(chunks_per_paper)/len(chunks_per_paper):.1f}, "
          f"max={max(chunks_per_paper)}")
    print(f"Chunk chars:    min={min(char_lens)}, "
          f"avg={sum(char_lens)/len(char_lens):.0f}, "
          f"max={max(char_lens)}")

if __name__ == "__main__":
    main()
