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
MAX_TOKENS_PER_TURN = 2048

SYSTEM_PROMPT = """You are a research assistant grounded in a cropus of 77 ML research papers on efficient LLM inference.

YOUR JOB
- Answer questions about the paper in the corpus.
- Use the `retrieve` tool to find supporting evidence before answering
- You MAY can retrieve multiple times with different queries to gather evidence from different angles.
- Cite paper titles and page numbers for every factual claim.
- If the corpus does not cover the question, say so explicityly - do not speculate.

GROUNDEDNESS RULES
- Every factual claim in your answer MUST be supported by a retrieved chunk.
- Cite inline as: [Paer Title (page N)]
- If retrieved chunks disagree, present both views with their sources.
- Do not invent paper titles or claims not present in the chunks.

WORKFLOW
1. Think briefly about what the question is really asking.
2. Call retrieve with a precise query.
3. Read the results. If they are sufficient, synthesize the answer.
4. If results miss the mark, REFINE your query and retrieve again - try different terminology, more specific terms, or a differnt angle.
5. Final answer in markdown. Concise - quality over verbosity."""

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

def answer(question: str, max_iterations: int = MAX_ITERATIONS) -> dict:
    client = Anthropic()
    messages = [{"role": "user", "content": question}]
    trace = []

    overall_start = time.perf_counter()
    llm_total_s = 0.0
    retrieve_total_s = 0.0

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---", flush=True)
        
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
            "text": "",
            "tool_calls": [],
        }

        print(
            f"LLM call: {llm_elapsed_s:.2f}s | "
            f"in={response.usage.input_tokens} out={response.usage.output_tokens} tokens",
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
            print(f"Thought: {turn['text'][:300]}", flush=True)
        for tc in turn["tool_calls"]:
            print(f"Tool call: retrieve(query={tc['input'].get('query', '')!r}, k={tc['input'].get('k', 10)})", flush=True)

        if response.stop_reason != "tool_use":
            total_s = time.perf_counter() - overall_start
            return {
                "question": question,
                "answer": turn["text"],
                "iteration": iteration + 1,
                "trace": trace,
                "timing": {
                    "total_seconds": round(total_s, 3),
                    "llm_seconds": round(llm_total_s, 3),
                    "retrieve_seconds": round(retrieve_total_s, 3),
                    "overhead_seconds": round(total_s - llm_total_s - retrieve_total_s, 3),
                },
            }

        messages.append({"role": "assistant", "content": response.content})

        tool_result_blocks = []
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

                print(f"  Top 3 retrieved:", flush=True)
                for r in tc["result_summary"][:3]:
                    print(f"    → {r['title'][:60]} (p.{r['page']})", flush=True)

                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": format_chunks_for_llm(chunks),
                })
        messages.append({"role": "user", "content": tool_result_blocks})
    
    return {
        "question": question,
        "answer": "[max iterations reached without final answer]",
        "iterations": max_iterations,
        "trace": trace,
    }

def save_run(result: dict, runs_dir: Path = RUNS_DIR) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = runs_dir / f"run_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return path

if __name__ == "__main__":
    import sys

    default = "How can speculative decoding be made effective at large batch sizes? What are the bottlenecks?"
    question = " ".join(sys.argv[1:]) or default

    print(f"Question: {question}\n")
    result = answer(question)
    
    print(f"\n{'=' * 60}")
    print(f"FINAL ANSWER ({result['iteration']} iterations)")
    print('=' * 60)
    print(result["answer"])

    print(f"\n{'=' * 60}")
    print("REASONING TRACE")
    print('=' * 60)
    for turn in result["trace"]:
        print(f"\n--- Iteration {turn['iteration']} (stop_reason: {turn['stop_reason']}) ---")
        if turn["text"]:
            print(f"Thought: {turn["text"][:400]}")
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

    saved = save_run(result)
    print(f"\nRun saved to {saved}")

