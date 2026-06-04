import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph import grade_groundedness_node

CHUNKS = [
    {"paper_title": "FlashAttention: Fast and Memory-Efficient Exact Attention", "page": 2,
     "text": "FlashAttention uses tiling to load blocks of Q, K, V into on-chip SRAM and avoids "
             "materializing the large N×N attention matrices S and P in HBM."},
    {"paper_title": "FlashAttention: Fast and Memory-Efficient Exact Attention", "page": 5,
     "text": "Instead of storing the attention matrix for the backward pass, FlashAttention stores "
             "the softmax normalization statistics and recomputes attention on-chip."},
]

CASES = [
    ("FlashAttention uses tiling to keep blocks in SRAM and avoid writing the attention matrices to HBM "
     "[FlashAttention: Fast and Memory-Efficient Exact Attention (page 2)].",
     True,  "clean / grounded"),
    ("FlashAttention recomputes attention in the backward pass instead of storing it "
     "[FlashAttention: Fast and Memory-Efficient Exact Attention (page 5)].",
     True,  "clean / grounded #2"),
    ("FlashAttention uses tiling [FlashAttention: Fast and Memory-Efficient Exact Attention (page 2)]. "
     "It also applies 8-bit weight quantization [GPTQ (page 3)].",
     False, "FABRICATED CITATION (GPTQ not retrieved) -> deterministic floor"),
    ("FlashAttention achieves a 100x speedup on every GPU "
     "[FlashAttention: Fast and Memory-Efficient Exact Attention (page 2)].",
     False, "UNSUPPORTED CLAIM, valid citation -> LLM grader"),
    ("FlashAttention was invented by OpenAI in 2025 [MadeUpPaper (page 1)].",
     False, "fabricated paper + claim"),
]

def run():
    caught = total_ungrounded = false_pos = total_grounded = 0
    print("=== grade_groundedness ==")
    for ans, expected, label in CASES:
        grounded = grade_groundedness_node({"answer": ans, "chunks": CHUNKS})["grounded"]
        ok = (grounded == expected)
        if expected:
            total_grounded += 1
            false_pos += (grounded is False)
        else:
            total_ungrounded += 1
            caught += (grounded is False)
        print(f"  [{'OK  ' if ok else 'MISS'}] expected={expected} got={grounded}  ({label})")
    print(f"\nUngrounded caught: {caught}/{total_ungrounded}  (detection rate)")
    print(f"False positives:   {false_pos}/{total_grounded}  (grounded answers wrongly flagged)")

if __name__ == "__main__":
    run()