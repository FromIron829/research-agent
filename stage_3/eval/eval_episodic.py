import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph import route_intent_node
import episodic
from eval_router import FA_HIST   # reuse a history fixture for the control cases

EVAL_THREAD = "eval-episodic"

# ======================= Part 1: memory_recall routing =======================
# (history, message, expected_intent)
ROUTING = [
    # memory_recall: about the user's OWN past conversation record
    ([], "What did I ask you about earlier?", "memory_recall"),
    ([], "What was that attention paper you told me about?", "memory_recall"),
    ([], "Which papers have I asked about so far?", "memory_recall"),
    ([], "What did we talk about last week?", "memory_recall"),
    ([], "Remind me what we discussed about quantization.", "memory_recall"),   # boundary vs followup
    # controls: must NOT be hijacked to memory_recall
    ([], "What is FlashAttention?", "corpus"),
    (FA_HIST, "How does it compare to GPTQ?", "corpus"),
    (FA_HIST, "Summarize that in one sentence.", "followup"),
]

def run_routing():
    print("=== Part 1: memory_recall routing ===")
    rows = []
    for hist, msg, exp in ROUTING:
        got = route_intent_node({"history": hist, "question": msg})["intent"]
        rows.append((msg, exp, got))
    correct = sum(1 for _, e, g in rows if e == g)
    # recall_miss: a recall question NOT routed to memory_recall (won't hit the episodic store)
    recall_miss = [(m, g) for m, e, g in rows if e == "memory_recall" and g != "memory_recall"]
    # hijack: a real corpus/followup question wrongly pulled INTO memory_recall (won't get answered)
    hijack = [(m, e, g) for m, e, g in rows if e != "memory_recall" and g == "memory_recall"]
    print(f"Accuracy: {correct}/{len(rows)}")
    print(f"\nRECALL MISSES ({len(recall_miss)}) - recall question routed elsewhere (no episodic lookup):")
    for m, g in recall_miss:
        print(f"    got {g!r}: {m}")
    print(f"\nHIJACKS ({len(hijack)}) - real question pulled into memory_recall (answered 'nothing stored'):")
    for m, e, g in hijack:
        print(f"    expected {e!r}: {m}")
    return recall_miss, hijack

# ======================= Part 2: recall@k over the episodic store =======================
SEED = [
    ("What is FlashAttention?",        "FlashAttention is an IO-aware exact attention algorithm using tiling."),
    ("How does GPTQ quantize weights?", "GPTQ quantizes weights to 3-4 bits post-training, one-shot."),
    ("Explain PagedAttention.",        "PagedAttention manages the KV cache like OS virtual-memory paging."),
    ("What is speculative decoding?",  "Speculative decoding drafts tokens with a small model, then verifies."),
    ("How does AWQ work?",             "AWQ is activation-aware weight quantization protecting salient weights."),
]
# (paraphrased recall query, expected target question)
QUERIES = [
    ("the IO-aware attention algorithm I asked about", "What is FlashAttention?"),
    ("that method that quantizes weights to 3-4 bits",  "How does GPTQ quantize weights?"),
    ("the KV cache paging technique",                   "Explain PagedAttention."),
    ("drafting tokens with a small model then verifying", "What is speculative decoding?"),
    ("activation-aware quantization of salient weights", "How does AWQ work?"),
]

def run_recall():
    print("\n=== Part 2: recall@k over the episodic store ===")
    for q, a in SEED:
        episodic.remember_turn(EVAL_THREAD, q, a)
    try:
        hit1 = hit3 = 0
        for query, target in QUERIES:
            hits = episodic.recall(query, k=3)
            got_qs = [h["question"] for h in hits]
            r1 = got_qs[:1] == [target]
            r3 = target in got_qs
            hit1 += r1; hit3 += r3
            mark = "OK  " if r1 else ("@3  " if r3 else "MISS")
            print(f"  [{mark}] {query!r}\n        top: {got_qs}")
        n = len(QUERIES)
        print(f"\nrecall@1: {hit1}/{n}    recall@3: {hit3}/{n}")
    finally:
        episodic._conversations.delete(where={"thread_id": EVAL_THREAD})   # self-clean
        print("(cleaned up seeded eval turns)")

if __name__ == "__main__":
    run_routing()
    run_recall()
