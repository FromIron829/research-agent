import json
from collections import Counter
from pathlib import Path

import arxiv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "paper_pdfs"
OUTPUT_PATH = PROJECT_ROOT / "stage_1" / "data" / "manifest.json"

pdf_ids = [p.stem for p in PDF_DIR.glob("*.pdf")]
print(f"Found {len(pdf_ids)} PDFs")

client = arxiv.Client()
client.query_url_format = "https://export.arxiv.org/api/query?{}"
search = arxiv.Search(id_list=pdf_ids)
results = list(client.results(search))
print(f"Fetched {len(results)} from arXiv")

manifest = {}
for result in results:
    arxiv_id = result.get_short_id()
    manifest[arxiv_id] = {
        "title": result.title,
        "primary_category": result.primary_category,
        "abstract": result.summary,
        "authors": [author.name for author in result.authors],
        "year": result.published.year
    }

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"Saved {len(manifest)} entries to {OUTPUT_PATH}")

missing = set(pdf_ids) - set(manifest.keys())
if missing:
    print(f"WARNING: {len(missing)} IDs did not come back: {missing}")
else:
    print(f"All {len(manifest)} PDFs matched to metadata.") 

categories = Counter(entry["primary_category"] for entry in manifest.values())
print("Category spread: ", categories)