import re
import sys
import time
import json
import operator
import email.utils
from datetime import datetime, timezone
import hashlib
import requests
from pathlib import Path
from typing import TypedDict
import xml.etree.ElementTree as ET
from typing import TypedDict, Annotated
from memory import format_history, summarize_history
import episodic

from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_1"))
from hybrid import retrieve_hybrid, add_chunks
from retrieve import _collection, _openai_client, EMBED_MODEL, embed_query
from extract import strip_references

load_dotenv()
client = Anthropic()
MODEL = "claude-sonnet-4-6"
MAX_ATTEMPTS = 3
MAX_GEN = 2

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_HEADERS = {"User-Agent": "research-agent/0.1 (https://github.com/FromIron829/research-agent)"}
_ARXIV_CACHE = Path(__file__).resolve().parent / ".arxiv_cache"
_ARXIV_CACHE.mkdir(exist_ok=True)
_last_arxiv_call = [0.0]
_ARXIV_THROTTLE_FILE = _ARXIV_CACHE / "last_call.txt"

CITE_BLOCK = re.compile(r"\[([^\]]+)\]")
ENTRY = re.compile(r"\s*(.*?)\s*[(,]\s*pages?\s*[\d,\s\u2013-]+\)?", re.IGNORECASE)
_SUBS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

def _norm(s: str) -> str:
    s = s.translate(_SUBS).replace("$", "").replace("_", "").replace("…", "").replace("...", "")
    return re.sub(r"\s+", " ", s.lower()).strip()

def _title_matches(proposed: str, page_text: str, threshold: float = 0.6):
    head = _norm(page_text[:800])
    words = [w for w in _norm(proposed).split() if len(w) > 3]
    if not words:
        return True
    hits = sum(1 for w in words if w in head)
    return hits / len(words) >= threshold

def verify_citations(answer: str, chunks: list[dict], history: list[dict] | None = None) -> list[str]:
    """Deterministic: return cited papers that were NOT in the retrieved chunks (likely fabricated)."""
    retrieved = {_norm(c["paper_title"]) for c in chunks}
    if history:
        for msg in history:
            if msg["role"] == "assistant":
                for block in CITE_BLOCK.findall(msg["content"]):
                    for entry in block.split(";"):
                        m = ENTRY.search(entry)
                        if m:
                            retrieved.add(_norm(m.group(1).strip()))
    bad = []
    for block in CITE_BLOCK.findall(answer):
        for entry in block.split(";"):           # a bracket can hold multiple papers
            m = ENTRY.search(entry)
            if not m:
                continue
            t = _norm(m.group(1))
            if t and not any(t in r or r.startswith(t) for r in retrieved):
                bad.append(m.group(1).strip())
    return bad

def _parse_retry_after(value, default):
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        try:
            dt = email.utils.parsedate_to_datetime(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
        except Exception:
            return default
    
def _throttle(min_gap: float = 3.5):
    last = 0.0
    if _ARXIV_THROTTLE_FILE.exists():
        try:
            last = float(_ARXIV_THROTTLE_FILE.read_text().strip())
        except ValueError:
            pass
    gap = time.time() - last
    if gap < min_gap:
        time.sleep(min_gap - gap)
    _ARXIV_THROTTLE_FILE.write_text(str(time.time()))      # PERSISTS across script restarts

class GraphState(TypedDict):
    question: str
    query: str
    rewritten_query: str
    sub_queries: list[str]
    chunks: list[dict]
    relevant: bool
    attempts: int
    first_answer: str
    answer: str
    grounded: bool
    issues: str
    gen_attempts: int
    candidate: dict
    approved: bool
    ingested: bool
    ingested_aid: str
    history: Annotated[list[dict], operator.add]
    intent: str
    best_answer: str
    best_n_issues: int
    summary: str          # running summary of turns older than the recent window (persists across turns)
    n_summarized: int     # how many history messages have already been folded into `summary`

# ---------- Paper injection ----------
def search_arxiv(query: str, max_results: int = 1, retries: int = 2):
    # cache: never re-hit arXiv for a query we've already searched
    key = hashlib.md5(f"{query}|{max_results}".encode()).hexdigest()
    cache_file = _ARXIV_CACHE / f"{key}.json"
    if cache_file.exists():
        print(f"[arxiv] cache hit: {query!r}")
        return json.loads(cache_file.read_text())

    for attempt in range(retries):
        _throttle()
        try:
            resp = requests.get(ARXIV_API, params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
            }, headers=ARXIV_HEADERS, timeout=(5, 20))     # short read timeout for the request path
        except requests.exceptions.RequestException as err:
            print(f"[arxiv] request failed ({type(err).__name__}); brief retry")
            time.sleep(2)
            continue

        if resp.status_code == 429:
            print("[arxiv] 429 — degrading gracefully (no long wait in request path)")
            return []                                       # don't block the request; degrade to "not in corpus"

        resp.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        out = []
        for e in root.findall("a:entry", ns):
            pdf = next((l.get("href") for l in e.findall("a:link", ns) if l.get("title") == "pdf"), None)
            out.append({
                "title": " ".join(e.findtext("a:title", default="", namespaces=ns).split()),
                "arxiv_id": e.findtext("a:id", default="", namespaces=ns).rsplit("/", 1)[-1],
                "pdf_url": pdf,
            })
        cache_file.write_text(json.dumps(out))
        return out

    return []      # fail soft after retries — never raise/hang the request

def propose_ingestion_node(state: GraphState):
    # PRIMARY: the LLM names the foundational paper. For the well-known papers the corpus is missing
    # (Transformer, Adam, BERT...), the LLM knows the exact arXiv ID more reliably than keyword search
    # surfaces it. A wrong/hallucinated ID is caught downstream by the title check in ingest_node.
    question = state.get("rewritten_query") or state["question"]
    cand = llm_propose_paper(question)
    if cand:
        print(f"[propose] LLM proposed: {cand['title']} ({cand['arxiv_id']})")

    # SECONDARY: only if the LLM won't commit (obscure paper) -> arXiv search + STRICT re-rank.
    if cand is None:
        q = client.messages.create(model=MODEL, max_tokens=40, messages=[{"role": "user", "content":
                "Output ONLY a concise arXiv search query (key terms or likely title) for the foundational "
                f"paper answering this. Plain text only — no markdown or quotes.\n{question}"}])
        search_query = re.sub(r'[*`"]', "", "".join(b.text for b in q.content if b.type == "text")).strip()
        print(f"[propose] arXiv query: {search_query!r}")
        try:
            results = search_arxiv(search_query, max_results=5)
        except Exception as err:
            print(f"[propose] arXiv lookup failed: {err}")
            results = []
        if results:
            listing = "\n".join(f"{i}: {r['title']}" for i, r in enumerate(results))
            pick = client.messages.create(model=MODEL, max_tokens=10, messages=[{"role": "user", "content":
                    f"Question: {state['question']}\n\nCandidates:\n{listing}\n\n"
                    "Output the index of the candidate that IS the specific paper the question is about. "
                    "A merely related or similar paper is NOT a match — output -1 if none is that exact paper. "
                    "Output ONLY the integer."}])
            raw = "".join(b.text for b in pick.content if b.type == "text").strip()
            m = re.search(r"-?\d+", raw)
            idx = int(m.group()) if m else -1
            if 0 <= idx < len(results):
                cand = results[idx]
                print(f"[propose] search picked {idx}: {cand['title']} ({cand['arxiv_id']})")

    if cand is None:
        return {"answer": "Not covered by the corpus, and I couldn't confidently identify an arXiv paper to add."}

    return {"candidate": cand,
            "answer": f"Not in corpus. Proposed: **{cand['title']}** ({cand['arxiv_id']})."}

# ---------- RAG retrieve ----------
def _retrieve_from_paper(query: str, aid: str, k: int = 6):
    res = _collection.query(query_embeddings=[embed_query(query)], n_results=k, where={"arxiv_id": aid})
    return [{
        "chunk_id": cid, "arxiv_id": m["arxiv_id"], "paper_title": m["paper_title"],
        "page": m["page"], "text": doc, "score": 1.0 - dist,
    } for cid, doc, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]

def retrieve_node(state: GraphState):
    sub_queries = [s for s in state.get("sub_queries", []) if s]
    if not sub_queries:
        sub_queries = [state.get("query") or state["question"]]
    
    seen = set()
    merged = []
    for q in sub_queries:
        for chunk in retrieve_hybrid(q, k=10):
            if chunk["chunk_id"] not in seen:
                seen.add(chunk["chunk_id"])
                merged.append(chunk)

    aid = state.get("ingested_aid")
    if aid and not any(c["arxiv_id"] == aid for c in merged):
        boost = _retrieve_from_paper(sub_queries[0], aid, k=6)
        merged = boost + merged

    print(f"[retrieve] sub_queries={sub_queries} -> {len(merged)} chunks")
    return {"chunks": merged, "query": sub_queries[0]}

# ---------- Tools ----------
GRADE_TOOL = {
    "name": "grade",
    "description": "Judge whether the retrieved sources are sufficient to answer the question well.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sufficient": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["sufficient", "reason"],
    },
}

GROUND_TOOL = {
    "name": "groundedness",
    "description": "Check whether the answer fabricates specific facts not supported by the sources.",
    "input_schema": {
        "type": "object",
        "properties": {
            "grounded": {"type": "boolean"},
            "issues": {
                "type": "string",
                "description": (
                    "Flag a claim ONLY if it asserts a specific fact, number, benchmark, or result "
                    "that is absent from or contradicted by the sources (a fabrication). "
                    "Do NOT flag high-level synthesis, characterizations, or reasonable inferences "
                    "that follow from combining the sources — comparison answers are expected to synthesize. "
                    "Quote each fabricated sentence verbatim, or 'none'."
                ),
            },
            "n_fabrications": {
                "type": "integer",
                "description": "Count of DISTINCT fabricated claims you flagged above (0 if grounded). "
                               "Used to pick the least-fabricated draft when regeneration is capped.",
            },
        },
        "required": ["grounded", "issues", "n_fabrications"],
    },
}

PROPOSE_TOOL = {
    "name": "propose_paper",
    "description": "Name the single arXiv paper that best answers the question - only if confident of its real arXiv ID.",
    "input_schema": {
        "type": "object",
        "properties": {
            "known": {
                "type": "boolean",
                "description": "True ONLY if you are confident this is a real paper with this exact arXiv ID."
            },
            "title": {
                "type": "string",
            },
            "arxiv_id": {
                "type": "string",
                "description": "e.g. 1706.03762 - the real identifier, empty if unsure." 
            },
        },
        "required": ["known", "title", "arxiv_id"],
    },
}

ROUTE_TOOL = {
    "name": "route",
    "description": "Classify the user's new message to dispatch it correctly.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": ["corpus", "followup"],
                       "description": "corpus = any question that needs retrieval from the research-paper knowledge base, "
                       "including comparisions involving a new entity not yet discussed. "
                       "followup = the question can be fully answered from the conversation above. "
                       "It summarizes, rephrases, or asks about content ALREADY in the prior assistant answer. "
                       "If the question introduces ANY entity, paper, or technique not covered in the conversation "
                       "(even if phrased conversationally), classify as corpus. "
                       "Even for a topic ALREADY discussed, if the question asks about an aspect, detail, number, "
                       "or mechanism that is NOT actually stated in the prior answer, classify as corpus — "
                       "followup applies ONLY when the answer is already contained in the conversation. "
                       "When unsure whether the conversation fully covers the answer, choose corpus."
                       "memory_recall = the user is asking about THEIR OWN past conversations, possibly across "
                       "sessions (e.g. 'what did I ask about last week', 'what was that paper you mentioned earlier'). "
                       "Answer by retrieving from the stored conversation history, NOT the papers corpus."
                       },
        },
        "required": ["intent"],
    },
}

PLAN_TOOL = {
    "name": "plan_queries",
    "description": "Rewrite the user's question and split it into retrieval sub-queries.",
    "input_schema": {
        "type": "object",
        "properties": {
            "rewritten_question": {
                "type": "string",
                "description": "The question rewritten as a fully standalone sentence - resolve all pronous and references using the conversation history.",
            },
            "sub_queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "One focused retrieval query per distinct topic. "
                    "For a simple question: one element. "
                    "For a comparison or contrast between N distinct things: N elements, one per thing. "
                    "Keep each sub-query short and factual - it goes directly into a vector search."
                ),
            },
        },
        "required": ["rewritten_question", "sub_queries"],
    },
}

def summarize_node(state: GraphState):
    """Entry node: fold any turns that fell out of the recent window into the running summary,
    so long sessions keep their early context compactly instead of truncating it away."""
    before = state.get("n_summarized", 0)
    summary, n = summarize_history(
        state.get("history", []), state.get("summary", ""), before, client, MODEL)
    if n != before:
        print(f"[summarize] folded {n - before} message(s) -> summary now {len(summary)} chars (n_summarized={n})")
    return {"summary": summary, "n_summarized": n}

def route_intent_node(state: GraphState):
    hist = format_history(state.get("history", []), state.get("summary", ""))
    msg = client.messages.create(
        model=MODEL, max_tokens=100,
        tools=[ROUTE_TOOL], tool_choice={"type": "tool", "name": "route"},
        messages=[{"role": "user", "content": f"{hist}\nUser's new message: {state['question']}\n\nClassify the intent."}],
    )
    intent = next((b.input for b in msg.content if b.type == "tool_use"), {}).get("intent", "corpus")
    print(f"[route] intent={intent}")
    return {"intent": intent}

def recall_node(state: GraphState):
    hists = episodic.recall(state["question"], k=3)
    if not hists:
        answer = "I don't have any earlier conversations stored to recall from yet."
        return {"answer": answer,
                "history": [{"role": "user", "content": state["question"]},
                            {"role": "assistant", "content": answer}]}
    ctx = "\n\n".join(
        f"[{datetime.fromtimestamp(h['ts'])}] You asked: {h['question']}\n"
        f"I answered: {h['answer']}"
        for h in hists
    )
    msg = client.messages.create(
        model=MODEL, max_tokens=400,
        messages=[{"role": "user", "content":
                   f"Past conversation turns (most relevant first):\n{ctx}\n\n"
                   f"The user now asks: {state['question']}\n\n"
                   "Answer using ONLY these past turns - remind them what was discussed and when."}],
    )
    answer = "".join(b.text for b in msg.content if b.type == "text")
    return {"answer": answer,
            "history": [{"role": "user", "content": state["question"]},
                        {"role": "assistant", "content": answer}]}

def plan_query_node(state: GraphState):
    hist = format_history(state.get("history", []), state.get("summary", ""))
    if not hist:
        q = state["question"]
        print(f"[plan] no histroy - passthrough: {q!r}")
        return {"query": q, "rewritten_query": q, "sub_queries": [q]}
    
    msg = client.messages.create(
        model=MODEL, max_tokens=200,
        tools=[PLAN_TOOL], tool_choice={"type": "tool", "name": "plan_queries"},
        messages=[{"role": "user", "content":
                   f"{hist}\nNew message: {state['question']}\n\n"
                   "Step 1: rewrite the message as a fully standalone question - "
                   "resolve all pronous and references using the conversation above.\n\n"
                   "Step 2: Decide what new information needs to be fetched from the knowledge base. "
                   "Any topic already explained in detail in the conversation above does NOT need a sub-query - "
                   "that information is avaliable from history and will be used at generation time. "
                   "Only generate sub-queries for topics NOT yet covered in the conversation."}],
    )
    p = next((b.input for b in msg.content if b.type == "tool_use"), {})
    rewritten = (p.get("rewritten_question") or state["question"]).strip()
    sub_queries = [s.strip() for s in (p.get("sub_queries") or []) if s.strip()]
    if not sub_queries:
        sub_queries = [rewritten]
    
    print(f"[plan] rewritten={rewritten!r}")
    print(f"[plan] sub_queries={sub_queries}")
    return {"query": sub_queries[0], "rewritten_query": rewritten, "sub_queries": sub_queries}

def answer_from_history_node(state: GraphState):
    hist = format_history(state.get("history", []), state.get("summary", ""))
    msg = client.messages.create(
        model=MODEL, max_tokens=512,
        messages=[{"role": "user", "content":
                   f"{hist}\nUser's new message {state['question']}\n\n"
                   "Respond using ONLY the conversation above - do not retrieve or use outside knowledge."}],
    )
    answer = "".join(b.text for b in msg.content if b.type == "text")
    return {"answer": answer,
            "history": [{"role": "user", "content": state["question"]},
                        {"role": "assistant", "content": answer}]}

def rewrite_query_node(state: GraphState):
    hist = format_history(state.get("history", []), state.get("summary", ""))
    if not hist:
        return {"query": state["question"]}
    msg = client.messages.create(
        model=MODEL, max_tokens=80,
        messages=[{"role": "user", "content":
                   f"{hist}\nNew message: {state['question']}\n\n"
                   "Rewrite the new message as a standalone search query that resolves all pronous "
                   "and references using the conversation above. Output only the query string."}],
    )
    new_query = next((b.text for b in msg.content if b.type == "text"), state["question"]).strip()
    print(f"[rewrite_query] {state['question']!r} -> {new_query!r}")
    return {"query": new_query}

def grade_relevance_node(state: GraphState):
    context = "\n\n".join(f"[{c['paper_title']} p{c['page']}] {c['text']}" for c in state["chunks"])
    question = state.get("rewritten_query") or state["question"]
    hist = format_history(state.get("history", []), state.get("summary", ""))
    msg = client.messages.create(
        model=MODEL, max_tokens=300,
        tools=[GRADE_TOOL], tool_choice={"type": "tool", "name": "grade"},
        messages=[{"role": "user", "content":
                   f"{hist}"
                   f"Question: {question}\n\n"
                   f"Newly retrieved sources:\n{context}\n\n"
                   "The conversation history above is also available at generation time. "
                   "Grade whether the retrieved sources COMBINED WITH the conversation history "
                   "are sufficient to answer the question. "
                   "For comparision questions, sources convering each entity separately are sufficient - "
                   "no single source needs to directly compare them"
                   }],
    )
    grade = next(b.input for b in msg.content if b.type == "tool_use")
    sufficient = grade.get("sufficient", True)
    reason = grade.get("reason", "")
    attempts = state.get("attempts", 0) + 1
    print(f"[grade] sufficient={sufficient} (attempt {attempts}) — {reason}")
    return {"relevant": sufficient, "attempts": attempts}

def grade_groundedness_node(state: GraphState):
    context = "\n\n".join(f"[{c['paper_title']} p{c['page']}] {c['text']}" for c in state["chunks"])
    hist = format_history(state.get("history", []), state.get("summary", ""))
    msg = client.messages.create(
        model=MODEL, max_tokens=400,
        tools=[GROUND_TOOL], tool_choice={"type": "tool", "name": "groundedness"},
        messages=[{"role": "user", "content":
            f"{hist}"
            f"Retrieved sources:\n{context}\n\n"
            f"Answer:\n{state['answer']}\n\n"
            "Flag ONLY claims asserting a specific fact, number, or result that is absent from or "
            "contradicted by BOTH the retrieved sources AND the conversation history (a fabrication). "
            "Do NOT flag reasonable synthesis or high-level characterizations — comparisons synthesize by nature. "
            "If you cannot quote a specific fabricated sentence, set grounded=true, issues='none'."}],
    )
    g = next(b.input for b in msg.content if b.type == "tool_use")
    grounded = g.get("grounded", True)
    issues = (g.get("issues") or "none").strip()
    raw_fabs = g.get("n_fabrications", 0)
    llm_fabs = raw_fabs if isinstance(raw_fabs, int) else 0

    # Guard: "not grounded" with no named issue is a false positive -> treat as grounded
    if not grounded and issues.lower() in ("", "none"):
        grounded = True
        llm_fabs = 0

    # Deterministic floor: a citation to a paper that wasn't retrieved is fabricated, full stop.
    bad = verify_citations(state["answer"], state["chunks"], state.get("history", []))
    if bad:
        grounded = False
        note = "Citations to papers NOT in the retrieved sources (fabricated): " + "; ".join(bad)
        issues = note if issues.lower() in ("", "none") else f"{issues} | {note}"
        print(f"[citation-check] {len(bad)} unresolved citation(s): {bad}")

    gen_attempts = state.get("gen_attempts", 0) + 1
    # Count = LLM-flagged fabrications + deterministically-caught fabricated citations. This is the
    # keep-best signal: it must distinguish a 3-fabrication draft from a 1-fabrication one.
    n_issues = 0 if grounded else max(1, llm_fabs + len(bad))
    print(f"[groundedness] grounded={grounded} (gen attempt {gen_attempts}) — {issues[:80]}")

    out = {"grounded": grounded, "issues": issues, "gen_attempts": gen_attempts}
    best_n = state.get("best_n_issues")
    if grounded or best_n is None or n_issues < best_n:
        out["best_answer"] = state["answer"]
        out["best_n_issues"] = 0 if grounded else n_issues
    return out

# ---------- Refine query to retrieve again ----------
def refine_query_node(state: GraphState):
    msg = client.messages.create(
        model=MODEL, max_tokens=80,
        messages=[{"role": "user", "content":
                   f"This search query returned insufficient results: {state['query']!r}\n"
                   f"For the question: {state['question']!r}\n"
                   "Write ONE improved search query using different terms or a sharper angle. Output only the query."}],
    )
    new_q = "".join(b.text for b in msg.content if b.type == "text").strip()
    print(f"[refine] {state['query']!r} -> {new_q!r}")
    return {"query": new_q}

# ---------- Result generation ----------
def generate_node(state: GraphState):
    context = "\n\n".join(
        f"[{c['paper_title']} (page {c['page']})]\n{c['text']}" for c in state["chunks"]
    )
    fix = ""
    if state.get("issues") and state["issues"].lower() != "none":
        fix = (f"\n\nA reviewer flagged these claims as possibly unsupported: {state['issues']}\n"
                        "For EACH flagged claim, do exactly one of: (a) keep it and add a citation that is "
                        "actually present in the sources above, or (b) remove just that one claim. "
                        "Do NOT alter any other claim. Do NOT invent citations or page numbers.")

    hist = format_history(state.get("history", []), state.get("summary", "")) 
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="Answer the question using ONLY the provided sources. Cite as [Paper Title (page N)].",
        messages=[{"role": "user", "content": f"{hist}\nQuestion: {state['question']}\n\nSources:\n{context}{fix}"}],
    )
    answer = "".join(b.text for b in msg.content if b.type == "text")
    out = {"answer": answer}
    if not state.get("first_answer"):
        out["first_answer"] = answer
    return out

def respond_node(state: GraphState, config):
    if state.get("grounded"):
        answer = state["answer"]
    else:
        answer = state.get("best_answer") or state.get("first_answer") or state["answer"]
        print(f"[respond] ungrounded after cap -> returning best attempt "
              f"({state.get('best_n_issues')} issue(s))")
    episodic.remember_turn(config["configurable"]["thread_id"], state["question"], answer)
    return {"answer": answer,
            "history": [{"role": "user", "content": state["question"]},
                        {"role": "assistant", "content": answer}]}

# ---------- Human approval ----------
def approval_node(state: GraphState):
    cand = state["candidate"]
    decision = interrupt({
        "candidate": cand,
        "prompt": f"Add '{cand['title']}' ({cand['arxiv_id']}) to the knowledge base? (yes/no)",
    })
    approved = str(decision).strip().lower() in ("yes", "y", "approve")
    print(f"[approval] decision={decision!r} -> approved={approved}")
    if not approved:
        return {"approved": False, "answer": f"Declined - '{cand['title']}' was not added."}
    return {"approved": True}

def ingest_node(state: GraphState):
    cand = state["candidate"]
    aid = cand["arxiv_id"]
    print(f"[ingest] downloading {aid} ...")
    try:
        r = requests.get(cand["pdf_url"], headers=ARXIV_HEADERS, timeout=60)
        r.raise_for_status()
        pdf = r.content
    except requests.exceptions.RequestException as err:
        print(f"[ingest] download failed: {err}")
        return {"ingested": True, "answer": f"Couldn't download {cand['title']} from arXiv. Try again later."}
    tmp = Path(f"/tmp/{aid}.pdf"); tmp.write_bytes(pdf)

    raw = pymupdf4llm.to_markdown(str(tmp), page_chunks=True, show_progress=False)
    first_page = raw[0]["text"] if raw else ""
    if not _title_matches(cand["title"], first_page):
        print(f"[ingest] title mismatch - '{aid}' is not '{cand['title']}'; aborting to protect the corpus")
        return {"ingested": True,
                "answer": f"Couldn't verify arXiv:{aid} matches '{cand['title']}' - not adding it to avoid corrupting the corpus."}
    
    pages = strip_references([{"page": i + 1, "text": p["text"]} for i, p in enumerate(raw)])
    docs = [Document(page_content=pg["text"],
                                     metadata={"arxiv_id": aid, "paper_title": cand["title"], "page": pg["page"]})
                            for pg in pages]
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=800, chunk_overlap=100)
    splits = [d for d in splitter.split_documents(docs) if len(d.page_content) >= 200]
    print(f"[ingest] {len(pages)} pages -> {len(splits)} chunks; embedding + upserting ...")

    texts = [d.page_content for d in splits]
    embs = _openai_client.embeddings.create(model=EMBED_MODEL, input=texts).data
    _collection.upsert(
        ids=[f"{aid}_{i:04d}" for i in range(len(splits))],
        embeddings=[e.embedding for e in embs],
        documents=texts,
        metadatas=[{"arxiv_id": aid, "paper_title": cand["title"], "page": d.metadata["page"]} for d in splits],
    )
    new_chunks = [{
        "chunk_id": f"{aid}_{i:04d}",
        "arxiv_id": aid,
        "paper_title": cand["title"],
        "page": d.metadata["page"],
        "text": d.page_content,
    } for i, d in enumerate(splits)]
    add_chunks(new_chunks)

    print(f"[ingest] upserted into '{_collection.name}' -> now includes {cand['title']}")
    return {"ingested": True,
            "ingested_aid": aid,
            "attempts": 0, 
            "query": state.get("rewritten_query") or state["question"],
            "sub_queries": state.get("sub_queries") or [state.get("rewritten_query") or state["question"]]}

# ---------- Routers ----------
def route_after_grade(state: GraphState):
    if state["relevant"]:
        return "generate"
    if state["attempts"] >= MAX_ATTEMPTS:
        if state.get("ingested"):
            print("[route] still insufficient after ingestion -> answer with what we have")
            return "generate"
        print("[route] retrieval insufficient after refinement -> propose ingestion")
        return "propose_ingestion"
    return "refine_query"

def route_after_groundedness(state: GraphState):
    if state["grounded"]:
        return "respond"
    if state["gen_attempts"] >= MAX_GEN:
        print("[route] groundedness cap reached -> responding as-is")
        return "respond"
    return "regenerate"

def route_after_propose(state: GraphState):
    return "approval" if state.get("candidate") else "end"

def route_after_approval(state: GraphState):
    return "ingest" if state["approved"] else "deny"

def llm_propose_paper(question: str):
    """Fallback when arXiv search is unavailable: let the LLM name the paper from parametric knowledge."""
    msg = client.messages.create(
        model=MODEL, max_tokens=200,
        tools=[PROPOSE_TOOL], tool_choice={"type": "tool", "name": "propose_paper"},
        messages=[{"role": "user", "content":
                   f"The local corpus cannot answer this question:\n{question}\n\n"
                   "Name the single foundational arXiv paper that does - but ONLY if you are confident of its real "
                   "arXiv ID. If you are not sure of the exact ID, set known=false. Do NOT guess an ID."}],
    )
    p = next((b.input for b in msg.content if b.type == "tool_use"), {})
    aid = (p.get("arxiv_id") or "").strip()
    if not p.get("known") or not aid:
        return None
    return {"title": (p.get("title") or "").strip(), "arxiv_id": aid,
            "pdf_url": f"https://arxiv.org/pdf/{aid}"}

builder = StateGraph(GraphState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade_relevance", grade_relevance_node)
builder.add_node("refine_query", refine_query_node)
builder.add_node("generate", generate_node)
builder.add_node("respond", respond_node)
builder.add_node("grade_groundedness", grade_groundedness_node)
builder.add_node("propose_ingestion", propose_ingestion_node)
builder.add_node("approval", approval_node)
builder.add_node("ingest", ingest_node)
builder.add_node("route_intent", route_intent_node)
builder.add_node("answer_from_history", answer_from_history_node)
builder.add_node("plan_query", plan_query_node)
builder.add_node("summarize", summarize_node)
builder.add_node("recall", recall_node)

builder.add_edge(START, "summarize")
builder.add_edge("summarize", "route_intent")
builder.add_conditional_edges("route_intent", lambda s: s["intent"], {
    "corpus": "plan_query",
    "followup": "answer_from_history",
    "memory_recall": "recall",
})
builder.add_edge("recall", END)
builder.add_edge("plan_query", "retrieve")
builder.add_edge("answer_from_history", END)
builder.add_edge("retrieve", "grade_relevance")
builder.add_conditional_edges("grade_relevance", route_after_grade, {
    "generate": "generate",
    "refine_query": "refine_query",
    "propose_ingestion": "propose_ingestion"
})
builder.add_edge("refine_query", "retrieve")
builder.add_edge("generate", "grade_groundedness")
builder.add_conditional_edges("grade_groundedness", route_after_groundedness, {
    "respond": "respond",
    "regenerate": "generate"
})
builder.add_conditional_edges("propose_ingestion", route_after_propose, {
    "approval": "approval",
    "end": END
})
builder.add_conditional_edges("approval", route_after_approval, {
    "ingest": "ingest",
    "deny": END,
})
builder.add_edge("respond", END)
builder.add_edge("ingest", "retrieve")
_conn = sqlite3.connect(str(Path(__file__).resolve().parent / "checkpoints.db"),
                        check_same_thread=False)
graph = builder.compile(checkpointer=SqliteSaver(_conn))

def fresh_turn(question: str):
    return {
        "question": question, "query": "", "rewritten_query": "", "sub_queries": [], "chunks": [], "relevant": False, "attempts": 0,
        "first_answer": "", "answer": "", "grounded": False, "issues": "", "gen_attempts": 0,
        "candidate": {}, "approved": False, "ingested": False, "ingested_aid": "", "intent": "", "best_answer": "", "best_n_issues": None,
    }

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "session-1"}}
    while True:
        q = input("\nyou: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        result = graph.invoke(fresh_turn(q), config=config)
        while "__interrupt__" in result:
            intr = result["__interrupt__"][0]
            print(f"\n>>> APPROVAL NEEDED:", intr.value["prompt"])
            result = graph.invoke(Command(resume=input("yes/no: ")), config=config)
        print("\nagent:", result["answer"])