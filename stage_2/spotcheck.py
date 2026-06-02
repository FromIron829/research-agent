import sys, json
from pathlib import Path
from judge import load_source_index, parse_citations, gather_sources, STAGE2, EVAL_PATH, ANSWER_PATH, JUDGED_PATH

ids = sys.argv[1:] or ["s01", "m03", "h05", "h07"]   # default: a spread across tiers + both gap questions

eval_q  = {q["id"]: q for q in json.loads(EVAL_PATH.read_text())["questions"]}
answers = {a["id"]: a for a in json.loads(ANSWER_PATH.read_text())}
judged  = {j["id"]: j for j in json.loads(JUDGED_PATH.read_text())["judged"]}
index, titles = load_source_index()

for qid in ids:
    q, a, j = eval_q[qid], answers[qid], judged[qid]
    cites = parse_citations(a["answer"])
    sources, unmatched = gather_sources(cites, index, titles)
    print("=" * 90)
    print(f"{qid} [{q['tier']}]  {q['question']}")
    print(f"key_points: {q['key_points']}")
    print("-" * 90, "\nANSWER:\n", a["answer"])
    print("-" * 90, f"\nCITED SOURCES ({len(sources)} resolved, {len(unmatched)} unmatched: {unmatched}):")
    for s in sources:
        print(f"  • {s['paper_title']} p{s['page']} — {s['text'][:200].strip()}...")
    print("-" * 90, "\nJUDGE SAID:")
    for dim in ["factual_accuracy", "citation_quality", "completeness", "coherence"]:
        print(f"  {dim:18} {j['scores'][dim]}  — {j['scores'][f'{dim}_reason']}")
    print(f"  TOTAL {j['scores']['total']}/12")
    print("  >>> your scores (acc/cite/comp/coh): ____ / ____ / ____ / ____\n")
