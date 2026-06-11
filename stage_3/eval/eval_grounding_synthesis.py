import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph import grade_groundedness_node

# Two papers, BOTH "retrieved" (so verify_citations does not fire) — this isolates the
# LLM grader's synthesis-vs-fabrication discrimination, the distinction added in Exp 2.
CHUNKS = [
    {"paper_title": "FlashAttention: Fast and Memory-Efficient Exact Attention", "page": 2,
     "text": "FlashAttention is an IO-aware exact attention algorithm that uses tiling and "
             "recomputation to reduce reads/writes between GPU HBM and on-chip SRAM. It is up to "
             "3x faster than standard attention on GPT-2."},
    {"paper_title": "GPTQ: Accurate Post-Training Quantization", "page": 1,
     "text": "GPTQ is a one-shot post-training weight quantization method that compresses model "
             "weights to 3-4 bits with negligible accuracy loss."},
    {"paper_title": "GPTQ: Accurate Post-Training Quantization", "page": 9,
     "text": "GPTQ's inference speedup comes from reduced memory movement, not from fewer "
             "arithmetic operations."},
]

# (answer, expected_grounded, category, label)
CASES = [
    # --- SYNTHESIS that must be SPARED (expected grounded=True) ---
    ("FlashAttention speeds up attention computation by cutting HBM traffic "
     "[FlashAttention: Fast and Memory-Efficient Exact Attention (page 2)], while GPTQ compresses "
     "weights to 3-4 bits [GPTQ: Accurate Post-Training Quantization (page 1)]. They target "
     "different bottlenecks and are complementary — both could be applied to the same model.",
     True, "SYNTHESIS", "pure cross-source synthesis + reasonable inference ('complementary')"),

    ("Both methods reduce memory pressure rather than arithmetic: GPTQ's speedup comes from less "
     "memory movement [GPTQ: Accurate Post-Training Quantization (page 9)], and FlashAttention "
     "reduces HBM reads/writes [FlashAttention: Fast and Memory-Efficient Exact Attention (page 2)]. "
     "In that sense they share a memory-centric philosophy.",
     True, "SYNTHESIS", "high-level characterization not verbatim in any single source"),

    # --- GROUNDED SPECIFICS control (real numbers ARE in sources -> grounded=True) ---
    ("FlashAttention is up to 3x faster than standard attention on GPT-2 "
     "[FlashAttention: Fast and Memory-Efficient Exact Attention (page 2)]; GPTQ quantizes weights "
     "to 3-4 bits [GPTQ: Accurate Post-Training Quantization (page 1)].",
     True, "GROUNDED-SPECIFICS", "specific numbers that are actually in the sources"),

    # --- FABRICATED SPECIFICS that must be CAUGHT (expected grounded=False) ---
    ("FlashAttention cuts HBM traffic [FlashAttention: Fast and Memory-Efficient Exact Attention "
     "(page 2)] and GPTQ delivers a 4-8x model size reduction at 2-bit precision "
     "[GPTQ: Accurate Post-Training Quantization (page 1)].",
     False, "FABRICATION", "fabricated number ('4-8x at 2-bit') not in sources, amid valid synthesis"),

    ("FlashAttention achieves up to 3x speedup on GPT-2 [FlashAttention: Fast and Memory-Efficient "
     "Exact Attention (page 2)] and simultaneously lowers model perplexity by 50% "
     "[FlashAttention: Fast and Memory-Efficient Exact Attention (page 2)].",
     False, "FABRICATION", "fabricated benchmark ('50% lower perplexity') beside a real number"),
]

N = 3   # the synthesis boundary is stochastic; sample each case

def run():
    false_pos = false_neg = 0          # false_pos: synthesis punished; false_neg: fabrication passed (DANGEROUS)
    n_syn = n_fab = 0
    print("=== grade_groundedness: synthesis vs fabrication (N=%d/case) ===\n" % N)
    for ans, expected, cat, label in CASES:
        verdicts = []
        for _ in range(N):
            g = grade_groundedness_node({"answer": ans, "chunks": CHUNKS, "history": [],
                                         "question": "How do FlashAttention and GPTQ compare?"})["grounded"]
            verdicts.append(g)
        majority = sum(verdicts) >= (N / 2)        # majority-grounded
        stable = "stable" if len(set(verdicts)) == 1 else "FLIPPED"
        ok = (majority == expected)
        if expected:
            n_syn += 1
            false_pos += (not majority)            # expected grounded, got ungrounded
        else:
            n_fab += 1
            false_neg += majority                  # expected ungrounded, got grounded
        print(f"  [{'OK  ' if ok else 'MISS'}] {cat:18} exp={expected!s:5} got={verdicts} ({stable})")
        print(f"        {label}")

    print(f"\nFALSE POSITIVES (synthesis wrongly flagged): {false_pos}/{n_syn}  <- Exp 2 regression guard")
    print(f"FALSE NEGATIVES (fabrication passed = DANGEROUS): {false_neg}/{n_fab}")

if __name__ == "__main__":
    run()
