import time

import json
from collections import defaultdict
from pathlib import Path

from rerank import retrieve_with_rerank

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = PROJECT_ROOT / "stage_1" / "eval" / "eval.json"
RESULTS_DIR = PROJECT_ROOT / "stage_1" / "results"
RESULTS_PATH = RESULTS_DIR / "rerank_cohere.json"


def recall_at_k(question: dict, retrieved: list[dict], k: int) -> float:
    gold = set(question["gold_paper_ids"])
    top_k_papers = {c["arxiv_id"] for c in retrieved[:k]}
    return 1.0 if gold & top_k_papers else 0.0


def main():
    with open(EVAL_PATH) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} eval questions\n")

    per_question = []
    
    RERANK_DELAY = 7

    for q in questions:
        retrieved = retrieve_with_rerank(q["question"], k=10, initial_k=20)
        r5 = recall_at_k(q, retrieved, 5)
        r10 = recall_at_k(q, retrieved, 10)
        per_question.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "difficulty": q["difficulty"],
            "topic": q["topic"],
            "gold_paper_ids": q["gold_paper_ids"],
            "recall_at_5": r5,
            "recall_at_10": r10,
            "top_5_papers": [
                {
                    "arxiv_id": c["arxiv_id"],
                    "title": c["paper_title"],
                    "rerank_score": round(c["rerank_score"], 3),
                    "vector_score": round(c["vector_score"], 3),
                }
                for c in retrieved[:5]
            ],
        })
        print(f"  {q['question_id']} [{q['difficulty']:<8}] r@5={r5:.0f} r@10={r10:.0f}")
        time.sleep(RERANK_DELAY)
        
    n = len(per_question)
    overall_r5 = sum(p["recall_at_5"] for p in per_question) / n
    overall_r10 = sum(p["recall_at_10"] for p in per_question) / n

    by_difficulty = defaultdict(list)
    by_topic = defaultdict(list)
    for p in per_question:
        by_difficulty[p["difficulty"]].append(p["recall_at_5"])
        by_topic[p["topic"]].append(p["recall_at_5"])

    print(f"\n=== EXPERIMENT: vector+rerank ({n} questions, text-embedding-3-small + Cohere rerank-v3.5) ===")
    print(f"Overall:  recall@5 = {overall_r5:.2f}   recall@10 = {overall_r10:.2f}")

    print("\nBy difficulty (recall@5):")
    for diff, vals in sorted(by_difficulty.items()):
        print(f"  {diff:<10}  {sum(vals)/len(vals):.2f}  (n={len(vals)})")

    print("\nBy topic (recall@5):")
    for topic, vals in sorted(by_topic.items()):
        print(f"  {topic:<22}  {sum(vals)/len(vals):.2f}  (n={len(vals)})")

    failures = [p for p in per_question if p["recall_at_5"] == 0.0]
    if failures:
        print(f"\nFailures at recall@5 ({len(failures)}/{n}):")
        for p in failures:
            print(f"  {p['question_id']} [{p['difficulty']} | {p['topic']}]")
            print(f"    Q:    {p['question'][:90]}")
            print(f"    Gold: {p['gold_paper_ids']}")
            print(f"    Top:  {[t['arxiv_id'] for t in p['top_5_papers']]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "vector+rerank",
        "embed_model": "text-embedding-3-small",
        "rerank_model": "rerank-v3.5",
        "initial_k": 20,
        "final_k": 10,
        "n_questions": n,
        "overall": {"recall_at_5": overall_r5, "recall_at_10": overall_r10},
        "by_difficulty": {d: sum(v)/len(v) for d, v in by_difficulty.items()},
        "by_topic": {t: sum(v)/len(v) for t, v in by_topic.items()},
        "per_question": per_question,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
