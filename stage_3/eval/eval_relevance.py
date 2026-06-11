import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph import grade_relevance_node, retrieve_hybrid

LABELED = [
    # in-corpus  -> grader should say SUFFICIENT (relevant=True)
    ("How does FlashAttention reduce memory I/O?", True),
    ("What does AWQ identify as salient weights?", True),
    ("How does PagedAttention prevent KV cache fragmentation in vLLM?", True),
    ("How does GPTQ achieve 3-4 bit post-training quantization?", True),
    ("How does H2O decide which KV cache entries to evict?", True),
    ("What did FlashAttention-2 change about work partitioning?", True),
    ("How does Medusa accelerate decoding with multiple heads?", True),
    # RELABELED 2026-06-11: the Transformer paper was ingested by CRAG during Stage 3.3 testing —
    # corpus-mutating agents invalidate static eval labels (Exp 18 finding).
    ("What is the original Transformer architecture from 'Attention is All You Need'?", True),
    # out-of-corpus -> grader should say INSUFFICIENT (relevant=False)
    ("What is the Mamba state-space model architecture?", False),
    ("Explain how convolutional neural networks work.", False),
    ("How does BERT's masked language modeling pretraining work?", False),
    ("How do diffusion models generate images?", False),
    ("What is batch normalization and why does it help training?", False),
]

def run():
    results = []
    for q, in_corpus in LABELED:
        chunks = retrieve_hybrid(q, k=10)
        relevant = grade_relevance_node({"question": q, "chunks": chunks})["relevant"]
        results.append((q, in_corpus, relevant))
    
    correct = sum(1 for _, exp, got in results if exp == got)
    over_trigger = [q for q, exp, got in results if exp and not got]
    missed_gap = [q for q, exp, got in results if not exp and got]

    print(f"\n=== grade_relevance ===")
    print(f"Accuracy: {correct}/{len(results)} = {correct/len(results):.0%}")
    print(f"\nOVER-TRIGGERED ({len(over_trigger)}) - in-corpus graded insufficient (wasteful refine/ingest):")
    for q in over_trigger: print(f"    -{q}")
    print(f"\nMISSED GAPS ({len(missed_gap)}) - out-of-corpus graded sufficient (answers from junk = dangerous:)")
    for q in missed_gap: print(f"    -{q}")

if __name__ == "__main__":
    run()

