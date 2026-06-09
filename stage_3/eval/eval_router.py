import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph import route_intent_node
# stage_3/eval/eval_router.py  —  labeled set for route_intent_node
# Each case: (history, message, expected_intent)
# route_intent_node reads state["history"] + state["question"] -> {"intent": ...}

# --- Reusable conversation histories (assistant content kept deliberately scoped) ---
FA_HIST = [
    {"role": "user", "content": "What is FlashAttention?"},
    {"role": "assistant", "content": "FlashAttention is an IO-aware exact attention algorithm "
     "that uses tiling and recomputation to cut reads/writes between GPU HBM and SRAM, making "
     "attention faster and more memory-efficient."},
]   # NOTE: covers WHAT FA is. Does NOT cover GPTQ, PagedAttention, or specific speedup numbers.

AWQ_HIST = [
    {"role": "user", "content": "How does AWQ identify salient weights?"},
    {"role": "assistant", "content": "AWQ flags salient weights by activation magnitude and "
     "scales them before quantization to reduce error."},
]   # Does NOT cover SmoothQuant.

KV_HIST = [
    {"role": "user", "content": "Explain KV cache compression techniques."},
    {"role": "assistant", "content": "KV cache compression cuts the memory of stored key/value "
     "states via token eviction (H2O, Scissorhands), quantization (KIVI), and layer-wise "
     "allocation (PyramidKV)."},
]   # Lists techniques. Does NOT cover Mamba, nor latency mechanisms.

QAT_HIST = [
    {"role": "user", "content": "What is quantization-aware training?"},
    {"role": "assistant", "content": "QAT trains the model with quantization in the loop, "
     "generally yielding better accuracy than post-training quantization but at high training cost."},
]   # DOES mention accuracy -> a later accuracy question is answerable from history.

GAP_CASES = [
    ([], "What can you do?"),
    (FA_HIST, "Thanks, that's super clear!"),
]

MULTI_HIST = [
    {"role": "user", "content": "What is FlashAttention?"},
    {"role": "assistant", "content": "FlashAttention: IO-aware exact attention via tiling."},
    {"role": "user", "content": "How does GPTQ quantize weights?"},
    {"role": "assistant", "content": "GPTQ is one-shot post-training quantization to 3-4 bits."},
]

LABELED = [
    # --- A. Clear corpus: first-turn standalone questions (no history) ---
    ([], "What is FlashAttention?", "corpus"),
    ([], "How does AWQ identify salient weights?", "corpus"),
    ([], "Explain KV cache compression techniques.", "corpus"),

    # --- B. Clear followup: pure transform/recall of the prior answer, no new retrieval ---
    (FA_HIST, "Summarize that in one sentence.", "followup"),
    (AWQ_HIST, "Can you rephrase that for a beginner?", "followup"),
    (KV_HIST, "Put your previous answer as bullet points.", "followup"),
    (KV_HIST, "Which technique did you mention first?", "followup"),

    # --- C. New-entity follow-up: SAFETY-CRITICAL (the Exp 2 bug). New entity => corpus ---
    (FA_HIST, "How does it compare to GPTQ?", "corpus"),          # SAFETY-CRITICAL
    (FA_HIST, "Is it better than PagedAttention?", "corpus"),      # SAFETY-CRITICAL
    (AWQ_HIST, "How does that differ from SmoothQuant?", "corpus"),# SAFETY-CRITICAL
    (FA_HIST, "What about Medusa — how does that speed things up?", "corpus"),  # SAFETY-CRITICAL
    (KV_HIST, "Does Mamba have the same problem?", "corpus"),      # SAFETY-CRITICAL; Mamba is out-of-corpus, but routing is still corpus (ingestion path decides later)

    # --- D. Same-entity DEEPENING: needs info not in the prior answer => corpus ---
    (FA_HIST, "What speedup numbers does it achieve?", "corpus"),  # FA_HIST has no numbers
    (KV_HIST, "How do these techniques reduce latency?", "corpus"),# KV_HIST has no latency detail
    (FA_HIST, "How does the backward pass recomputation work in detail?", "corpus"),

    # --- E. Same-entity, answerable from history => followup  (paired contrast w/ row D) ---
    (QAT_HIST, "How does it affect model accuracy?", "followup"),  # QAT_HIST already states accuracy
    (MULTI_HIST, "What was the quantization one I asked about?", "followup"),  # recall from recent history
    # NOTE: TAXONOMY-GAP cases ("What can you do?", "Thanks") live in GAP_CASES above —
    # reported but NOT scored, since neither class truly fits (motivates a future meta route).
]

def run():
    rows = []
    for hist, msg, expected in LABELED:
        got = route_intent_node({"history": hist, "question": msg})["intent"]
        rows.append((msg, expected, got))
    
    correct = sum(1 for _, e, g in rows if e == g)
    dangerous = [(m, g) for m, e, g in rows if e == "corpus" and g != "corpus"]
    safe = [(m, g) for m, e, g in rows if e == "followup" and g != "followup"]

    print("\n=== route_intent ===")
    print(f"Accuracy: {correct}/{len(rows)} = {correct/len(rows):.0%}")

    print(f"\nDANGEROUS ({len(dangerous)}) - corpus question routed to followup "
          "(answered from histroy, no retrieval -> ungrounded):")
    for m, g in dangerous:
        print(f"    got {g!r}: {m}")

    print(f"\nSAFE ({len(safe)}) - followup routed to corpus (wasteful retrieval, still grounded):")
    for m, g in safe:
        print(f"    got {g!r}: {m}")
    
    print(f"\nTAXONOMY-GAP (observed rounting, NOT scored - neither class truly fits:)")
    for hist, msg in GAP_CASES:
        got = route_intent_node({"history": hist, "question": msg})["intent"]
        print(f"    got {got!r}: {msg}")

if __name__ == "__main__":
    run()