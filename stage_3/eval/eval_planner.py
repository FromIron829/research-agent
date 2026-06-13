# Each case: (history, message, should_fetch, should_omit)
#   should_fetch = entities that MUST appear in sub_queries (miss = DANGEROUS)
#   should_omit  = history-covered entities that should NOT appear (present = WASTEFUL)

from eval_router import FA_HIST, AWQ_HIST, KV_HIST, MULTI_HIST   # reuse the 0.1 histories

PLAN_CASES = [
    # --- A. Passthrough: empty history -> sub_queries == [question] ---
    ([], "What is FlashAttention?", ["FlashAttention"], []),

    # --- B. Canonical: comparison, ONE entity known -> fetch the new one, omit the known one ---
    (FA_HIST, "How does it compare to GPTQ?", ["GPTQ"], ["FlashAttention"]),

    # --- C. New-entity follow-up: fetch the new entity, omit the discussed one ---
    (FA_HIST, "What about Medusa?", ["Medusa"], ["FlashAttention"]),
    (AWQ_HIST, "How does that differ from SmoothQuant?", ["SmoothQuant"], ["AWQ"]),

    # --- D. Multi-decomposition: two NEW entities -> expect 2 sub_queries covering both ---
    (FA_HIST, "How do GPTQ and AWQ compare?", ["GPTQ", "AWQ"], ["FlashAttention"]),
    (KV_HIST, "How do GPTQ and SmoothQuant quantize weights?", ["GPTQ", "SmoothQuant"], []),

    # --- E. OVER-OMISSION TRAPS: entity named in history but the needed info is NOT -> must REfetch ---
    (FA_HIST, "What speedup numbers does it achieve?", ["FlashAttention"], []),   # numbers not in history
    (KV_HIST, "How does H2O decide what to evict?", ["H2O"], []),                 # H2O named, never explained
    (KV_HIST, "How do these techniques reduce latency?", ["KV cache"], []),       # latency detail not in history
]

# Reported, NOT scored: both entities already in history -> ideal planner emits ~0 sub_queries
# (arguably should have been a followup at the router). Observe behavior, don't force a label.
GAP_CASES = [
    (MULTI_HIST, "Which is more memory-efficient?", ["FlashAttention", "GPTQ"]),
]

# Tolerant entity matching: "FlashAttention" / "Flash Attention" / "FA" must all count as a hit.
ALIASES = {
    "FlashAttention": ["flashattention", "flash attention"],
    "GPTQ":           ["gptq"],
    "AWQ":            ["awq"],
    "SmoothQuant":    ["smoothquant", "smooth quant"],
    "Medusa":         ["medusa"],
    "H2O":            ["h2o", "h₂o", "h2 o"],
    "KV cache":       ["kv cache", "kv-cache", "key/value", "key value", "kvcache"],
}

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from graph import plan_query_node

def _hit(entity: str, sub_queries: list[str]) -> bool:
    """True if entity (via aliases) appears in any sub_query — space/case tolerant."""
    needles = [a.replace(" ", "") for a in ALIASES.get(entity, [entity.lower()])]
    blob = "".join(sub_queries).lower().replace(" ", "")
    return any(n in blob for n in needles)

def run():
    rows = []
    for hist, msg, fetch, omit in PLAN_CASES:
        sq = plan_query_node({"history": hist, "question": msg})["sub_queries"]
        missing   = [e for e in fetch if not _hit(e, sq)]   # DANGEROUS
        redundant = [e for e in omit  if _hit(e, sq)]       # WASTEFUL
        rows.append((msg, sq, missing, redundant))

    dangerous = [(m, miss, sq) for m, sq, miss, red in rows if miss]
    wasteful  = [(m, red,  sq) for m, sq, miss, red in rows if red]
    clean     = sum(1 for _, _, miss, red in rows if not miss and not red)

    print("\n=== plan_query ===")
    print(f"Clean (all fetched, none redundant): {clean}/{len(rows)}")

    print(f"\nDANGEROUS ({len(dangerous)}) - should-fetch entity MISSING from sub_queries "
          "(never retrieved -> ungrounded/incomplete):")
    for m, miss, sq in dangerous:
        print(f"    missing {miss}: {m}\n        sub_queries={sq}")

    print(f"\nWASTEFUL ({len(wasteful)}) - should-omit entity PRESENT (redundant re-retrieval):")
    for m, red, sq in wasteful:
        print(f"    redundant {red}: {m}\n        sub_queries={sq}")

    print(f"\nGAP (observed, NOT scored - both entities already in history, ideal ~0 sub_queries):")
    for hist, msg, known in GAP_CASES:
        sq = plan_query_node({"history": hist, "question": msg})["sub_queries"]
        print(f"    {msg}\n        sub_queries={sq}  (known: {known})")

    return {"name": "planner", "n": len(rows), "dangerous": len(dangerous),
            "metrics": {"clean": clean, "wasteful": len(wasteful)}}

if __name__ == "__main__":
    run()
