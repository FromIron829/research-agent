#!/usr/bin/env python
"""Run every per-node eval, emit a scorecard, and gate on regression vs baseline.json.

The gate metric is each node's DANGEROUS-error count — the project's "never let this regress"
column (an out-of-corpus question answered from junk, a fabrication passed, a should-fetch entity
dropped). A run FAILS (exit 1) if any eval's dangerous count exceeds its baseline. Safe/wasteful
metrics are reported but not gated.

Usage:
  python run_all.py                  # local: full suite (incl. index-dependent evals)
  python run_all.py --ci             # CI: index-free evals only (no Chroma corpus needed)
  python run_all.py --update-baseline   # write current scorecard as the new baseline (no gate)

CI note: the index-free evals (router, planner, grounding-synthesis, keep-best) need only an
ANTHROPIC_API_KEY — they use LLM-only nodes or hardcoded fixture chunks, never the vector store.
relevance + episodic need a populated index, so they run locally only.
"""
import os
import sys
import json
import argparse
import importlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                 # eval modules
sys.path.insert(0, str(HERE.parent))          # graph
BASELINE = HERE / "baseline.json"

CI_EVALS = ["eval_router", "eval_planner", "eval_grounding_synthesis", "test_keep_best"]
LOCAL_ONLY = ["eval_relevance", "eval_episodic"]   # need the populated vector store


def run_suite(modules):
    scorecard = {}
    for mod_name in modules:
        print(f"\n{'=' * 72}\n  RUNNING  {mod_name}\n{'=' * 72}")
        result = importlib.import_module(mod_name).run()
        scorecard[result["name"]] = {
            "dangerous": result["dangerous"], "n": result["n"],
            "metrics": result.get("metrics", {}),
        }
    return scorecard


def gate(scorecard, baseline):
    """Return the list of (name, baseline_dangerous, current_dangerous) regressions."""
    regressions = []
    for name, cur in sorted(scorecard.items()):
        base = baseline.get(name)
        if base is None:
            print(f"  [new ] {name}: dangerous={cur['dangerous']} (no baseline — not gated)")
        elif cur["dangerous"] > base["dangerous"]:
            regressions.append((name, base["dangerous"], cur["dangerous"]))
            print(f"  [FAIL] {name}: dangerous {base['dangerous']} -> {cur['dangerous']}  REGRESSION")
        else:
            print(f"  [ok  ] {name}: dangerous={cur['dangerous']} (baseline {base['dangerous']})")
    return regressions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true", help="index-free evals only")
    ap.add_argument("--update-baseline", action="store_true", help="write baseline, skip gate")
    ap.add_argument("--prompt-version", help="eval a specific prompt variant (e.g. stage3-prompts-v2) "
                                             "to prove it's non-regressive before rollout")
    args = ap.parse_args()

    if args.prompt_version:
        os.environ["RA_PROMPT_VERSION"] = args.prompt_version   # _cache_context honors this when state lacks one
        print(f"  [prompt variant] evaluating CORE_SYSTEM = {args.prompt_version}")

    modules = CI_EVALS if args.ci else CI_EVALS + LOCAL_ONLY
    scorecard = run_suite(modules)

    print(f"\n{'=' * 72}\n  SCORECARD\n{'=' * 72}")
    print(json.dumps(scorecard, indent=2))

    if args.update_baseline:
        # merge so we don't drop baselines for evals not run this time (e.g. --ci keeps local ones)
        existing = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
        existing.update(scorecard)
        BASELINE.write_text(json.dumps(existing, indent=2) + "\n")
        print(f"\nWrote baseline -> {BASELINE.relative_to(HERE.parent.parent)}")
        return 0

    if not BASELINE.exists():
        print("\nNo baseline.json yet — establish one with:  run_all.py --update-baseline")
        return 1

    baseline = json.loads(BASELINE.read_text())
    print(f"\n{'=' * 72}\n  REGRESSION GATE  (dangerous-error counts vs baseline)\n{'=' * 72}")
    regressions = gate(scorecard, baseline)
    if regressions:
        print(f"\n❌ {len(regressions)} dangerous-error regression(s) — failing the build.")
        return 1
    print("\n✅ No dangerous-error regressions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
