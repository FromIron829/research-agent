import json
from pathlib import Path

import pymupdf4llm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "paper_pdfs"
MANIFEST_PATH = PROJECT_ROOT / "stage_1" / "data" / "manifest.json"
OUTPUT_PATH = PROJECT_ROOT / "stage_1" / "data" / "extracted.json"

REF_HEADINGS = {"references", "bibliography", "references and notes"}

def strip_references(pages: list[dict]) -> list[dict]:
    cleaned = []
    for page in pages:
        lines = page["text"].split("\n")
        cut_at = None
        for i, line in enumerate(lines):
            normalized = line.strip().strip("#* ").strip().lower()
            if normalized in REF_HEADINGS:
                cut_at = i
                break
        if cut_at is not None:
            kept = "\n".join(lines[:cut_at]).strip()
            if kept:
                cleaned.append({"page": page["page"], "text": kept})
            break
        cleaned.append(page)
    return cleaned

def main():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_paths)} PDFs, {len(manifest)} manifest entries")

    extracted = {}
    failures = []
    missing_metadata = []
    suspicious = []

    for pdf_path in pdf_paths:
        arxiv_id = pdf_path.stem

        if arxiv_id not in manifest:
            missing_metadata.append(arxiv_id)
            continue

        try:
            page_chunks = pymupdf4llm.to_markdown(
                str(pdf_path),
                page_chunks=True,
                show_progress=False,
            )
        except Exception as e:
            failures.append((arxiv_id, str(e)))
            continue

        pages = [
            {"page": i + 1, "text": chunk["text"]}
            for i, chunk in enumerate(page_chunks)
        ]

        pages = strip_references(pages)

        paper_chars = sum(len(pg["text"]) for pg in pages)
        if paper_chars < 1000:
            suspicious.append((arxiv_id, paper_chars))
        
        meta = manifest[arxiv_id]
        extracted[arxiv_id] = {
            "title": meta["title"],
            "primary_category": meta["primary_category"],
            "year": meta["year"],
            "pages": pages,
        }
        print(f" {arxiv_id}: {len(pages)} pages, {paper_chars:,} chars")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2, ensure_ascii=False)

    total_pages = sum(len(p["pages"]) for p in extracted.values())
    total_chars = sum(len(pg["text"]) for p in extracted.values() for pg in p["pages"])
    print(f"\nExtracted {len(extracted)} papers | {total_pages} pages | {total_chars:,} chars")
    print(f"Saved to {OUTPUT_PATH}")

    if missing_metadata:
        print(f"\n[skipped] {len(missing_metadata)} PDFs with no manifest entry: {missing_metadata}")
    if failures:
        print(f"\n[failed] {len(failures)} PDFs could not be extracted:")
        for arxiv_id, err in failures:
            print(f"    {arxiv_id}: {err}")
    if suspicious:
        print(f"\n[inspect] {len(suspicious)} papers extracted suspicious little text (likely scanned):")
        for arxiv_id, chars in suspicious:
            print(f"    {arxiv_id}: {chars} chars")

if __name__ == "__main__":
    main()