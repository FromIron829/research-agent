import json
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "stage_1" / "data" / "manifest.json"
OUTPUT_PATH = PROJECT_ROOT / "stage_1" / "eval" / "eval_candidates.json"

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are helping build a retrieval evaluation set for a RAG system over ML research papers on efficient LLM inference.

Given a paper's title and abstract, generate ONE realistic question that:
- A researcher familiar with the area might actually ask when searching for this paper
- Is answerable from the paper itself (not external knowledge)
- Sounds like natural user language - NOT an encyclopedic exam question
- Is specific enough that one paper is clearly the best answer

CRITICAL - avoid "answer-aware" bias:
- Do NOT lift distinctive phrases verbatim from the abstract
- For "semantic" difficulty: paraphrase the underlying idea so the question contains NONE of the paper's distinctive terms/acronyms. A researcher who knew the concept but not this paper's name for it should be able to write the same question.
- For "specific" difficulty: a rare technical term or acronym is OK, but phrase it as a user query, not a textbook prompt.

Aim for roughly 50% specific / 50% semantic across the corpus — decide for THIS paper which type makes more sense."""

TOOL = {
    "name": "submit_candidate",
    "description": "Submit a single candidate evaluation question for the retrieval eval set.",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question in natural user-like voice.",
            },
            "difficulty": {
                "type": "string",
                "enum": ["specific", "semantic"],
                "description": "specific = uses rare term/acronym (BM25-favorable). semantic = paraphrased, no exact term overlap (embedding-favorable).",
            },
            "topic": {
                "type": "string",
                "description": "Short topic tag, 1-3 words.",
            },
            "rationale": {
                "type": "string",
                "description": "One sentence: why this question is a good retrieval test for this paper."
            },
        },
        "required": ["question", "difficulty", "topic", "rationale"],
    }
}

def generate_candidate(client: Anthropic, title: str, abstract: str):
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "submit_candidate"},
        messages=[{
            "role": "user",
            "content": f"Paper title: {title}\n\nAbstract:\n{abstract}",
        }],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_candidate":
            return block.input
    return None

def main():
    client = Anthropic()

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    
    candidates = []
    failures = []

    for i, (arxiv_id, meta) in enumerate(manifest.items(), start=1):
        try:
            result = generate_candidate(client, meta["title"], meta["abstract"])
        except Exception as e:
            failures.append((arxiv_id, str(e)))
            continue

        if result is None:
            failures.append((arxiv_id, "no tool_use block in response"))
            continue

        candidates.append({
            "candidate_id": f"c{i:03d}",
            "question": result["question"],
            "gold_paper_ids": [arxiv_id],
            "topic": result["topic"],
            "difficulty": result["difficulty"],
            "rationale": result["rationale"],
            "source_title": meta["title"],
            "source_abstrat": meta["abstract"],
            "keep": False,
        })
        print(f"    [{i:3d}/{len(manifest)}] {arxiv_id} ({result['difficulty']}): {result['question'][:90]}")
    

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated {len(candidates)} candidates, saved to {OUTPUT_PATH}")

    if failures:
        print(f"\n{len(failures)} failed:")
        for arxiv_id, err in failures:
            print(f"    {arxiv_id}: {err}")
    
    dist = Counter(c["difficulty"] for c in candidates)
    print(f"\nDifficulty distribution: {dict(dist)}")

if __name__ == "__main__":
    main()