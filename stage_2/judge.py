import sys, json, re, difflib, time
from pathlib import Path
from collections import defaultdict

from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()
STAGE2 = Path(__file__).resolve().parent
ROOT = STAGE2.parent
CHUNKS_PATH = ROOT / "stage_1" / "data" / "chunks.json"
EVAL_PATH = STAGE2 / "eval" / "eval_set.json"
ANSWER_PATH = STAGE2 / "eval" / "agent_answers.json"
JUDGED_PATH = STAGE2 / "eval" / "judge.json"

JUDGE_MODEL = "gpt-4.1"

# ---------- source index: title -> page -> [chunk texts] ----------
def load_source_index(chunks_path: Path = CHUNKS_PATH):
    chunks = json.loads(chunks_path.read_text())
    index = defaultdict(lambda: defaultdict(list))
    for c in chunks:
        index[c["paper_title"]][c["page"]].append(c["text"])
    return index, list(index.keys())

# ---------- parse [Title (page N)] citations out of an answer ----------
CITE_BLOCK = re.compile(r"\[([^\]]+)\]")
ENTRY = re.compile(r"\s*(.*?)\s*[(,]\s*pages?\s*([\d,\s\u2013-]+)\)?", re.IGNORECASE)

def _expand_pages(spec: str):
    pages = []
    for tok in spec.split(","):
        tok = tok.strip().replace("\u2013", "-")
        if "-" in tok:
            a, b = tok.split("-", 1)
            if a.strip().isdigit() and b.strip().isdigit():
                pages.extend(range(int(a), int(b) + 1))
        elif tok.isdigit():
            pages.append(int(tok))
    return sorted(set(pages))

def parse_citations(answer: str):
    cites = []
    for block in CITE_BLOCK.findall(answer):
        for entry in block.split(";"):
            m = ENTRY.search(entry)
            if m:
                cites.append({"title": m.group(1).strip(), "pages": _expand_pages(m.group(2))})
    return cites

# ---------- fuzzy-match cited title -> canonical title, then fetch text ----------
_SUBS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def _norm(s: str) -> str:
    """Normalize a title for matching: fold unicode subscripts, strip LaTeX $/_ , drop the
    agent's '...' truncation, lowercase. So 'H₂O: Heavy-Hitter Oracle...' (agent, truncated)
    matches 'H$_2$O: Heavy-Hitter Oracle for ...' (manifest)."""
    s = s.translate(_SUBS).replace("$", "").replace("_", "").replace("…", "").replace("...", "")
    return re.sub(r"\s+", " ", s.lower()).strip()

def match_title(cited: str, titles: list[str]):
    cited_n = _norm(cited)
    norm_to_title = {_norm(t): t for t in titles}
    if cited_n in norm_to_title:                                   # 1. exact (normalized)
        return norm_to_title[cited_n]
    cands = [t for t in titles if cited_n in _norm(t) or _norm(t).startswith(cited_n)]
    if len(cands) == 1:                                            # 2. unique substring/prefix
        return cands[0]
    if cands:
        best = difflib.get_close_matches(cited_n, [_norm(t) for t in cands], n=1, cutoff=0)
        if best:
            return norm_to_title[best[0]]
    best = difflib.get_close_matches(cited_n, list(norm_to_title), n=1, cutoff=0.6)  # 3. fuzzy
    return norm_to_title[best[0]] if best else None

def gather_sources(citations, index, titles):
    sources, unmatched, seen = [], [], set()
    for c in citations:
        t = match_title(c["title"], titles)
        if t is None:
            unmatched.append(c["title"]); continue
        for p in c["pages"]:
            texts = index.get(t, {}).get(p, [])
            if not texts:
                unmatched.append(f"{t} (page {p} not found)"); continue
            if (t, p) not in seen:
                seen.add((t, p))
                sources.append({"paper_title": t, "page": p, "text": "\n".join(texts)})
    return sources, unmatched

# ---------- the judge ----------
JUDGE_SYSTEM = """You are an expert evaluator of a research assistant that answers questions about a corpus of efficient-LLM-inference papers.
Score the answer on four 0-3 dimensions. Be strict and calibrated.

Verify FACTUAL ACCURACY and CITATION QUALITY ONLY against the SOURCES provided (the chunks the assistant cited). If a claim's support is not in the provided text, treat it as unsupported. Use your own expertise only for COMPLETENESS and COHERENCE.

RUBRIC (0-3 each):
- factual_accuracy: 0 fabricated/contradicted; 1 a serious error; 2 minor imprecision; 3 fully accurate vs sources. Correctly flagging a gap is accurate; inventing a paper is 0-1.
- citation_quality: 0 none/fabricated; 1 many uncited or mis-pointed; 2 mostly cited & supported; 3 every claim cited, resolves to a source, and the source supports it.
- completeness: 0 misses central point; 1 omits major aspects; 2 minor omissions; 3 fully addresses the question at its tier (medium=genuine contrast; hard=cross-paper synthesis / gap reasoning).
- coherence: 0 disorganized; 1 understandable but padded; 2 clear; 3 direct-answer-first, structured, concise.

Score the dimensions INDEPENDENTLY — a fluent, well-cited answer can still be factually wrong."""

SCORE_SCHEMA = {
    "name": "rubric_scores",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "factual_accuracy":        {"type": "integer", "enum": [0, 1, 2, 3]},
            "factual_accuracy_reason": {"type": "string"},
            "citation_quality":        {"type": "integer", "enum": [0, 1, 2, 3]},
            "citation_quality_reason": {"type": "string"},
            "completeness":            {"type": "integer", "enum": [0, 1, 2, 3]},
            "completeness_reason":     {"type": "string"},
            "coherence":               {"type": "integer", "enum": [0, 1, 2, 3]},
            "coherence_reason":        {"type": "string"},
        },
        "required": ["factual_accuracy", "factual_accuracy_reason", "citation_quality", "citation_quality_reason",
                     "completeness", "completeness_reason", "coherence", "coherence_reason"],
    },
}

MAX_SOURCES = 50
MAX_SOURCE_CHARS = 2000

def format_sources(sources):
    if not sources:
        return "(No SOURCES - nothing the answer cited resolved to a retrieved chunk.)"
    shown = sources[:MAX_SOURCES]
    parts = [f"[{s['paper_title']} - page {s['page']}]\n{s['text'][:MAX_SOURCE_CHARS]}" for s in shown]
    if len(sources) > MAX_SOURCES:
        parts.append(f"(+{len(sources) - MAX_SOURCES} more cited pages omitted to fit context budget)")
    return "\n\n".join(parts)

def _create_with_retry(client, **kwargs):
    for attempt in range(6):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            wait = 2 ** attempt
            print(f"    rate-limited; sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError("rate-limit retries exhausted")

def score_answer(client, q: dict, answer: str, sources: list[dict]):
    user = f"""QUESTION (tier={q['tier']}): {q['question']}

KEY POINTS a strong answer covers (guidance, not a checklist): {q['key_points']}
GOLD PAPERS: {q['gold_papers']}

----- ASSISTANT ANSWER -----
{answer}

----- SOURCES (the chunks the assistant cited) -----
{format_sources(sources)}

Score the answer against the rubric."""
    resp = _create_with_retry(
        client=client,
        model=JUDGE_MODEL,
        temperature=0,
        max_tokens=700,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_schema", "json_schema": SCORE_SCHEMA},
    )
    msg = resp.choices[0].message
    if msg.refusal:
        raise RuntimeError(f"judge refused: {msg.refusal}")
    scores = json.loads(msg.content)
    scores["total"] = (scores["factual_accuracy"] + scores["citation_quality"] + scores["completeness"] + scores["coherence"])
    return scores

# ---------- aggregate by tier ----------
def aggregate(judged: list[dict]):
    dims = ["factual_accuracy", "citation_quality", "completeness", "coherence", "total"]
    by_tier = defaultdict(list)
    for j in judged:
        by_tier[j["tier"]].append(j["scores"])
    out = {}
    for tier, rows in by_tier.items():
        out[tier] = {d: round(sum(r[d] for r in rows) / len(rows), 2) for d in dims}
        out[tier]["n"] = len(rows)
    return out

if __name__ == "__main__":
    eval_q = {q["id"]: q for q in json.loads(EVAL_PATH.read_text())["questions"]}
    answers = json.loads(ANSWER_PATH.read_text())
    index, titles = load_source_index()
    client = OpenAI()

    judged = []
    for a in answers:
        q = eval_q[a["id"]]
        cites = parse_citations(a["answer"])
        sources, unmatched = gather_sources(cites, index, titles)
        scores = score_answer(client, q, a["answer"], sources)
        judged.append({"id": q["id"], "tier": q["tier"], "scores": scores,
                       "n_citations": len(cites), "n_unmatched": len(unmatched),
                       "n_sources": len(sources), "sources_capped": len(sources) > MAX_SOURCES,
                       "unmatched": unmatched})
        s = scores
        print(f"{q['id']:4} [{q['tier']:6}] total={s['total']:2}/12 "
              f"acc={s['factual_accuracy']} cite={s['citation_quality']} "
              f"comp={s['completeness']} coh={s['coherence']} "
              f"| cites={len(cites)} unmatched={len(unmatched)}")

    summary = aggregate(judged)
    JUDGED_PATH.write_text(json.dumps({"summary_by_tier": summary, "judged": judged},
                                          indent=2, ensure_ascii=False))
    print("\nBy tier:\n", json.dumps(summary, indent=2))