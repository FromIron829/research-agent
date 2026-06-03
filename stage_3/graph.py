import sys
from pathlib import Path
from typing import TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_1"))
from hybrid import retrieve_hybrid

load_dotenv()
client = Anthropic()
MODEL = "claude-sonnet-4-6"
MAX_ATTEMPTS = 3
MAX_GEN = 2

class GraphState(TypedDict):
    question: str
    query: str
    chunks: list[dict]
    relevant: bool
    attempts: int
    answer: str
    grounded: bool
    issues: str
    gen_attempts: int

def retrieve_node(state: GraphState):
    query = state.get("query") or state["question"]
    chunks = retrieve_hybrid(query, k=10)
    print(f"[retrieve] query={query!r} -> {len(chunks)} chunks")
    return {"chunks": chunks, "query": query}

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
    "description": "Check whether every claim in the answer is supported by the sources.",
    "input_schema": {
        "type": "object",
        "properties": {
            "grounded": {"type": "boolean"},
            "issues": {"type": "string", "description": "Unsupported claims, or 'none'."},
        },
        "required": ["grounded", "issues"],
    },
}

def grade_relevance_node(state: GraphState):
    context = "\n\n".join(f"[{c['paper_title']} p{c['page']}] {c['text'][:400]}" for c in state["chunks"])
    msg = client.messages.create(
        model=MODEL, max_tokens=300,
        tools=[GRADE_TOOL], tool_choice={"type": "tool", "name": "grade"},
        messages=[{"role": "user", "content":
                   f"Question: {state['question']}\n\nRetrieved sources:\n{context}\n\n"
                   "Are these sufficient to answer the question well?"}],
    )
    grade = next(b.input for b in msg.content if b.type == "tool_use")
    sufficient = grade.get("sufficient", True)
    reason = grade.get("reason", "")
    attempts = state.get("attempts", 0) + 1
    print(f"[grade] sufficient={sufficient} (attempt {attempts}) — {reason[:80]}")
    return {"relevant": sufficient, "attempts": attempts}

def grade_groundedness_node(state: GraphState):
    context = "\n\n".join(f"[{c['paper_title']} p{c['page']}] {c['text'][:400]}" for c in state["chunks"])
    msg = client.messages.create(
        model=MODEL, max_tokens=400,
        tools=[GROUND_TOOL], tool_choice={"type": "tool", "name": "groundedness"},
        messages=[{"role": "user", "content":
                   f"Sources:\n{context}\n\nAnswer:\n{state['answer']}\n\n"
                   "Is every claim in the answer supported by the sources?"}],
    )
    g = next(b.input for b in msg.content if b.type == "tool_use")
    grounded = g.get("grounded", True)
    issues = g.get("issues", "none")
    gen_attempts = state.get("gen_attempts", 0) + 1
    print(f"[groundedness] grounded={grounded} (gen attempt {gen_attempts}) — {issues[:80]}")
    return {"grounded": grounded, "issues": issues, "gen_attempts": gen_attempts}

def refine_query_node(state: GraphState):
    msg = client.messages.create(
        model=MODEL, max_tokens=80,
        messages=[{"role": "user", "content":
                   f"This search query returned insufficient results: {state['query']!r}\n"
                   f"For the question: {state['question']!r}\n"
                   "Write ONE improved search query suing different terms or a sharper angle. Output only the query."}],
    )
    new_q = "".join(b.text for b in msg.content if b.type == "text").strip()
    print(f"[refine] {state['query']!r} -> {new_q!r}")
    return {"query": new_q}

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

    msg = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="Answer the question using ONLY the provided sources. Cite as [Paper Title (page N)].",
        messages=[{"role": "user", "content": f"Question: {state['question']}\n\nSources:\n{context}{fix}"}],
    )
    answer = "".join(b.text for b in msg.content if b.type == "text")
    return {"answer": answer}

def route_after_grade(state: GraphState):
    if state["relevant"]:
        return "generate"
    if state["attempts"] >= MAX_ATTEMPTS:
        print("[route] max attempts reached -> answering with what we have")
        return "generate"
    return "refine_query"

def route_after_groundedness(state: GraphState):
    if state["grounded"]:
        return "respond"
    if state["gen_attempts"] >= MAX_GEN:
        print("[route] groundedness cap reached -> responding as-is")
        return "respond"
    return "regenerate"

builder = StateGraph(GraphState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade_relevance", grade_relevance_node)
builder.add_node("refine_query", refine_query_node)
builder.add_node("generate", generate_node)
builder.add_node("grade_groundedness", grade_groundedness_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_relevance")
builder.add_conditional_edges("grade_relevance", route_after_grade, {
    "generate": "generate",
    "refine_query": "refine_query",
})
builder.add_edge("refine_query", "retrieve")
builder.add_edge("generate", "grade_groundedness")
builder.add_conditional_edges("grade_groundedness", route_after_groundedness, {
    "respond": END,
    "regenerate": "generate"
})
graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({"question": "How does FlashAttention reduce memory I/O?"})
    print(result["answer"])
