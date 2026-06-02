import sys
import json
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage_1"))

from hybrid import retrieve_hybrid

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "stage_2" / "results" / "agent_runs"

MODEL = "claude-sonnet-4-6"
MAX_ITERATIONS = 6
MAX_TOKENS_PER_TURN = 4096

# SYSTEM_PROMPT = """You are a research assistant grounded in a cropus of 77 ML research papers on efficient LLM inference.

# YOUR JOB
# - Answer questions about the paper in the corpus.
# - Use the `retrieve` tool to find supporting evidence before answering
# - You MAY can retrieve multiple times with different queries to gather evidence from different angles.
# - Cite paper titles and page numbers for every factual claim.
# - If the corpus does not cover the question, say so explicityly - do not speculate.

# GROUNDEDNESS RULES
# - Every factual claim in your answer MUST be supported by a retrieved chunk.
# - Cite inline as: [Paer Title (page N)]
# - If retrieved chunks disagree, present both views with their sources.
# - Do not invent paper titles or claims not present in the chunks.

# WORKFLOW
# 1. Always begin each turn with one or two sentence of reasoning, stated explicitly. For example: "I need to find papers about X because Y." This reasoning must appear as text BEFORE any tool call.
# 2. Think briefly about what the question is really asking.
# 3. Call retrieve with a precise query.
# 4. Read the results. If they are sufficient, synthesize the answer.
# 5. If results miss the mark, REFINE your query and retrieve again - try different terminology, more specific terms, or a differnt angle.
# 6. Final answer in markdown. Concise - quality over verbosity."""


SYSTEM_PROMPT = """You are a research assistant grounded in a corpus of 77 ML research papers on efficient LLM inference.

YOUR JOB
- Answer questions about the papers in the corpus.
- Use the `retrieve` tool to find supporting evidence before answering
- You may retrieve multiple times with different queries to gather evidence from different angles.
- Cite paper titles and page numbers for every factual claim.
- If the corpus does not cover the question, say so explicitly - do not speculate.

GROUNDEDNESS RULES
- Every factual claim in your answer MUST be supported by a retrieved chunk.
- Cite inline as: [Paper Title (page N)]
- If retrieved chunks disagree, present both views with their sources.
- Do not invent paper titles or claims not present in the chunks.

WORKFLOW
1. Before every retrieve call — including the very first — write your reasoning on a line beginning with `Thought:` (one or two sentences stating why you are searching, e.g. "Thought: I need to find papers about X because Y."), then make the tool call.
2. Think briefly about what the question is really asking.
3. Call retrieve with a precise query.
4. Read the results. If they are sufficient, synthesize the answer.
5. If results miss the mark, REFINE your query and retrieve again - try different terminology, more specific terms, or a different angle.
6. Once the evidence is sufficient, write the final answer following ANSWER FORMAT below.

ANSWER FORMAT
- Emit a line containing exactly `===ANSWER===` to mark the start of your final answer. Everything after this marker is the answer shown to the user; put no reasoning, narration, or transition after it.
- Open the answer with a 1-2 sentence direct answer to the question, before any supporting detail. Do not restate the question.
- Follow with supporting points as short bullets, grouped under a brief header only when the answer has genuinely distinct parts (e.g. mechanisms vs. bottlenecks).
- Aim for under ~250 words unless the question genuinely requires more. End at the last substantive point; no generic closing summary.
- Brevity never overrides the GROUNDEDNESS RULES: keep every inline citation and still present disagreeing sources, even when compressing."""

TOOL_RETRIEVE = {
    "name": "retrieve",
    "description": (
        "Search the corpus of efficient-LLM-inference papers and return the top-k most relevant chunks. "
        "Each chunk includes its paper title, page number, and text. "
        "Use this whenever you need evidence to ground a claim. "
        "You can call it multiple times with different queries to find supporting evidence from multiple angles."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific and use terminology likely to appear in the papers (acronyms, technique names, technical phrases).",
            },
            "k": {
                "type": "integer",
                "description": "Number of chunks to return. Default 10, max 20.",
                "default": 10,
            },
        },
        "required": ["query"]
    },
}

def format_chunks_for_llm(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[Result {i}] Paper: {c['paper_title']} (page {c['page']})")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines)

ANSWER_SENTINEL = "===ANSWER==="

def strip_thought_prefix(text: str) -> str:
    """Drop leading 'Thought:' reasoning lines (fallback when the sentinel is absent)."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("Thought:")):
        i += 1
    return "\n".join(lines[i:]).strip()

def extract_answer(text: str) -> str:
    """Return the answer the model marked with ANSWER_SENTINEL, discarding any preamble before it.
    Robust to novel preamble phrasings because it keys on where the answer STARTS, not on the
    narration form. Falls back to stripping Thought: lines if the model omitted the sentinel.
    Reasoning is still preserved in the trace (turn['text']); only the answer field is cleaned."""
    if ANSWER_SENTINEL in text:
        return text.rsplit(ANSWER_SENTINEL, 1)[1].strip()
    return strip_thought_prefix(text)

def roll_cache_breakpoint(messages: list[dict]) -> None:
    """keep ONE cache_control breakpoint on the last content block of the conversation."""
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)
    last_content = messages[-1]["content"]
    if isinstance(last_content, list) and last_content and isinstance(last_content[-1], dict):
        last_content[-1]["cache_control"] = {"type": "ephemeral"}

def answer(question: str, max_iteration: int = MAX_ITERATIONS) -> dict:
    client = Anthropic()
    messages = [{"role": "user", "content": question}]
    trace = []

    overall_start = time.perf_counter()
    llm_total_s = 0.0
    retrieve_total_s = 0.0

    def finalize(answer_text: str, iteration: int, truncated: bool = False) -> dict:
        """Build the result dict. Closure captures question/trace/timers from the enclosing scope."""
        total_s = time.perf_counter() - overall_start

        total_input = sum(t["input_tokens"] for t in trace)
        total_output = sum(t["output_tokens"] for t in trace)
        total_write = sum(t.get("cache_creation_tokens", 0) for t in trace)
        total_read = sum(t.get("cache_read_tokens", 0) for t in trace)

        cost = (
            total_input / 1e6 * 3.00 +
            total_write / 1e6 * 3.75 +
            total_read / 1e6 * 0.30 +
            total_output / 1e6 * 15.00
        )

        return {
            "question": question,
            "answer": answer_text,
            "iterations": iteration,
            "truncated": truncated,
            "trace": trace,
            "timing": {
                "total_seconds": round(total_s, 3),
                "llm_seconds": round(llm_total_s, 3),
                "retrieve_seconds": round(retrieve_total_s, 3),
                "overhead_seconds": round(total_s - llm_total_s - retrieve_total_s, 3),
            },
            "usage": {
                "input_tokens": total_input,
                "cache_write_tokens": total_write,
                "cache_read_tokens": total_read,
                "output_tokens": total_output,
                "total_input_processed": total_input + total_write + total_read,
                "estimated_cost_usd": round(cost, 4)
            },
        }
    
    for iteration in range(max_iteration):
        print(f"\n--- Iteration {iteration + 1} ---", flush=True)

        roll_cache_breakpoint(messages)

        llm_start = time.perf_counter()
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS_PER_TURN,
            system=SYSTEM_PROMPT,
            tools=[TOOL_RETRIEVE],
            messages=messages,
        )
        llm_elapsed_s = time.perf_counter() - llm_start
        llm_total_s += llm_elapsed_s

        turn = {
            "iteration": iteration + 1,
            "stop_reason": response.stop_reason,
            "llm_seconds": round(llm_elapsed_s, 3),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
            "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
            "text": "",
            "tool_calls": [],
        }
        print(
            f"LLM call: {llm_elapsed_s:.2f}s | "
            f"in={turn['input_tokens']} "
            f"cache_write={turn['cache_creation_tokens']} "
            f"cache_read={turn['cache_read_tokens']} "
            f"out={turn['output_tokens']}",
            flush=True,
        )

        for block in response.content:
            if block.type == "text":
                turn["text"] += block.text
            elif block.type == "tool_use":
                turn["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        trace.append(turn)

        if turn["text"]:
            print(turn["text"][:300], flush=True)
        for tc in turn["tool_calls"]:
            print(f"Tool call: retrieve(query={tc['input'].get('query', '')!r}, k={tc['input'].get('k', 10)})", flush=True)
        
        # --- Terminal cases: return early ---
        if response.stop_reason == "end_turn":
            return finalize(extract_answer(turn["text"]), iteration + 1)

        if response.stop_reason == "max_tokens":
            print(f"\n WARNING: output truncated at max_tokens ({MAX_TOKENS_PER_TURN}). Answer is incomplete.", flush=True)
            return finalize(extract_answer(turn["text"]), iteration + 1, truncated=True)

        if response.stop_reason != "tool_use":
            print(f"\n Unexpected stop_reason: {response.stop_reason}", flush=True)
            return finalize(extract_answer(turn["text"]), iteration + 1)
        
        # --- Stop reason: "tool_use": excute tools, then the loop continue ---
        messages.append({"role": "assistant", "content": response.content})

        tool_result_block = []
        for tc in turn["tool_calls"]:
            if tc["name"] == "retrieve":
                q = tc["input"]["query"]
                k = min(tc["input"].get("k", 10), 20)

                retrieve_start = time.perf_counter()
                chunks = retrieve_hybrid(q, k=k)
                retrieve_elapsed_s = time.perf_counter() - retrieve_start
                retrieve_total_s += retrieve_elapsed_s
                tc["retrieve_seconds"] = round(retrieve_elapsed_s, 3)

                tc["result_summary"] = [
                    {"arxiv_id": c["arxiv_id"], "title": c["paper_title"], "page": c["page"]}
                    for c in chunks
                ]
                print(f"Tool retrieve: {retrieve_elapsed_s:.2f}s", flush=True)
                print(f" Top 3 retrieved:", flush=True)
                for r in tc["result_summary"][:3]:
                    print(f"    -> {r['title']} (p.{r['page']})", flush=True)
                
                tool_result_block.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": format_chunks_for_llm(chunks),
                })
        messages.append({"role": "user", "content": tool_result_block})
    
    print(f"\n Max iterations ({MAX_ITERATIONS}) reached without a final answer.", flush=True)
    return finalize("[max iterations reached without final answer]", max_iteration)


def save_run(result: dict, runs_dir: Path = RUNS_DIR) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = runs_dir / f"run_{timestamp}_unstructure.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return path

if __name__ == "__main__":
    import sys

    default = "How can attention complexity be reduced to linear time?"
    question = " ".join(sys.argv[1:]) or default

    print(f"Question: {question}\n")
    result = answer(question)
    
    print(f"\n{'=' * 60}")
    print(f"FINAL ANSWER ({result['iterations']} iterations)")
    print('=' * 60)
    print(result["answer"])

    print(f"\n{'=' * 60}")
    print("REASONING TRACE")
    print('=' * 60)
    for turn in result["trace"]:
        print(f"\n--- Iteration {turn['iteration']} (stop_reason: {turn['stop_reason']}) ---")
        if turn["text"]:
            print(turn["text"][:400])
        for tc in turn["tool_calls"]:
            q = tc["input"].get("query", "")
            k = tc["input"].get("k", 10)
            print(f"\nTool call: retrieve(query={q!r}, k={k})")
            for r in tc.get("result_summary", [])[:3]:
                print(f" -> {r['title'][:60]} (p.{r['page']})")

    t = result["timing"]
    print(f"\nTiming: total={t['total_seconds']}s "
          f"| llm={t['llm_seconds']}s ({100*t['llm_seconds']/t['total_seconds']:.0f}%) "
          f"| retrieve={t['retrieve_seconds']}s "
          f"| overhead={t['overhead_seconds']}s")

    u = result["usage"]
    print(
        f"\nTokens: input(full)={u['input_tokens']} "
        f"cache_write={u['cache_write_tokens']} "
        f"cache_read={u['cache_read_tokens']} "
        f"output={u['output_tokens']}"
    )
    print(f"Total input processed: {u['total_input_processed']:,} "
          f"(vs {u['input_tokens']:,} billed at full rate)")
    print(f"Estimated cost: ${u['estimated_cost_usd']}")

    #saved = save_run(result)
    #print(f"\nRun saved to {saved}")

