import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import graph
from graph import grade_groundedness_node, respond_node

# respond_node writes to the episodic store (0.6); stub it out for unit isolation
graph.episodic.remember_turn = lambda *a, **k: None
CFG = {"configurable": {"thread_id": "keep-best-test"}}

_results = []
def check(name, cond):
    _results.append(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond

# ---- Part 1: respond_node selection logic (deterministic, no LLM) ----
def test_respond_selection():
    print("Part 1 - respond_node selection logic (deterministic):")
    base = {"question": "q", "history": []}
    cur = respond_node({**base, "grounded": True, "answer": "CUR", "first_answer": "FIRST", "best_answer": "BEST"}, CFG)["answer"]
    check("grounded -> returns current answer", cur == "CUR")
    ung = respond_node({**base, "grounded": False, "answer": "CUR", "first_answer": "FIRST", "best_answer": "BEST"}, CFG)["answer"]
    check("ungrounded -> returns best_answer (not current/first)", ung == "BEST")
    nobest = respond_node({**base, "grounded": False, "answer": "CUR", "first_answer": "FIRST", "best_answer": ""}, CFG)["answer"]
    check("ungrounded, no best -> falls back to first_answer", nobest == "FIRST")
    nothing = respond_node({**base, "grounded": False, "answer": "CUR", "first_answer": "", "best_answer": ""}, CFG)["answer"]
    check("ungrounded, no best/first -> falls back to current", nothing == "CUR")

# ---- Part 2: keep-best tracking across a twice-failing loop (real grader) ----
CHUNKS = [
    {"paper_title": "FlashAttention: Fast and Memory-Efficient Exact Attention", "page": 2,
     "text": "FlashAttention uses tiling and recomputation to reduce reads/writes between GPU HBM and SRAM."},
]
# A = 3 fabricated specifics; B = 1 fabricated specific (plus a grounded claim). Both ungrounded; B is 'better'.
A = ("FlashAttention gives a 100x speedup [FlashAttention: Fast and Memory-Efficient Exact Attention (page 2)], "
     "uses 8-bit quantization, and was trained on 10 trillion tokens.")
B = ("FlashAttention reduces HBM reads/writes via tiling [FlashAttention: Fast and Memory-Efficient Exact "
     "Attention (page 2)] and reaches a 100x speedup.")

def test_keep_best_tracking():
    print("\nPart 2 - keep-best tracking across a twice-failing loop (real grader):")
    s = {"question": "q", "history": [], "chunks": CHUNKS, "answer": A,
         "first_answer": A, "best_answer": "", "best_n_issues": None, "gen_attempts": 0, "issues": ""}
    s.update(grade_groundedness_node(s))                    # gen1 (3 fabrications)
    print(f"    after gen1 (A, 3 fabs): grounded={s['grounded']} best_n_issues={s['best_n_issues']}")
    s["answer"] = B
    s.update(grade_groundedness_node(s))                    # gen2 (1 fabrication)
    print(f"    after gen2 (B, 1 fab):  grounded={s['grounded']} best_n_issues={s['best_n_issues']}")
    final = respond_node(s, CFG)["answer"]
    check("both attempts ungrounded (cap scenario reached)", not s["grounded"])
    check("keep-best distinguishes drafts: B (fewer fabs) kept over A", final == B and final != A)

if __name__ == "__main__":
    test_respond_selection()
    test_keep_best_tracking()
    print(f"\n{sum(_results)}/{len(_results)} checks passed")
