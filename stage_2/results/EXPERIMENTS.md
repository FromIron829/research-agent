# Stage 2 — Experiment Log

---

## Experiment 1 — Agent loop baseline (n = 1)

**Date:** 2026-05-27
**Result file:** */stage_2/results/agent_runs/run_20260527_200856.json*

### Setup

- **Retrieval method:** hybrid search (vector + BM25 + RRF), `initial_k = 50`
- **LLM model:** `claude-sonnet-4-6`
- **Loop config:** max 6 iterations, max 2048 output tokens per turn
- **System prompt:** see `agent.py` SYSTEM_PROMPT at commit `<hash>`
- **Query rewrite:** agent is allowed to rewrite the query across iterations to retrieve more precise chunks
- **Parallel tool calls:** Claude's native parallel tool-use enabled (multiple `tool_use` blocks per turn)
- **ReAct loop:** agent may call `retrieve` multiple times if early results don't give strong evidence
- **Test question (n=1):** *"How can speculative decoding be made effective at large batch sizes? What are the bottlenecks?"*

### Hypothesis

Agents need several iterations to gather strong supporting chunks for the user's query. For latency and token usage, I didn't know what to expect — first run, no prior.

### Result

- **Outcome:** synthesis truncated because `max_tokens = 2048` hit on the final iteration.
- **Total wall-clock:** **118.3 s** — LLM 116.3 s (98%), retrieve 2.0 s (2%), overhead ~0 s
- **Token usage:** 95,674 total — input 93,203, output 2,471
- **Iterations:** 4

**Per-iteration breakdown**

| Iteration | LLM time | In tokens | Out tokens |
|-----------|----------|-----------|------------|
| 1         | 3.1 s    | 975       | 138        |
| 2         | 3.1 s    | 19,305    | 142        |
| 3         | 8.7 s    | 30,802    | 143        |
| 4         | 101.5 s  | 42,121    | 2,048      |

**Cost estimate** *(Sonnet pricing: $3 / $15 per 1M tokens input / output)*

| Type      | Tokens     | Rate      | Cost     |
|-----------|------------|-----------|----------|
| Input     | 93,203     | $3 / 1M   | $0.2796  |
| Output    | 2,471      | $15 / 1M  | $0.0371  |
| **Total** | **95,674** |           | **$0.317** |

> Pricing reference: https://platform.claude.com/docs/en/about-claude/pricing

### Interpretation

Performance was worse than expected. Several distinct findings:

1. **Query refinement worked.** The agent issued different `retrieve` queries each iteration, starting broad ("speculative decoding large batch sizes throughput") and narrowing to specific mechanisms ("compute-bound memory-bound verification cost batch size"). The ReAct refinement loop is doing its job.

2. **Agent emitted ZERO reasoning text in iterations 1–3.** The model went straight to tool calls with no preceding "Thought:" text. This is a transparency problem — the system prompt should explicitly require reasoning before action.

3. **Parallel tool calls worked.** Each iteration issued two `retrieve` calls in a single LLM turn. The model didn't have to wait between deciding-call-1 and deciding-call-2 — free latency win from Claude's native parallel tool-use.

4. **`stop_reason == "max_tokens"` failed silently.** The loop's termination check (`stop_reason != "tool_use"`) treats `max_tokens` and `end_turn` identically. As a result, the agent returned a truncated answer with no warning. Bug in the loop, not just an undersized cap.

5. **Input tokens grew rapidly across iterations** (975 → 19,305 → 30,802 → 42,121). Each iteration's input includes the system prompt + all previous messages + new retrieval results. This is the dominant cost driver.

6. **Cost is input-dominated, not output-dominated.** Input accounted for ~88% of total cost ($0.2796 vs. $0.0371 output) because each iteration re-sends the full prior context plus new retrieval results. The per-turn output cap kept the output side bounded, but the real cost driver was input growth across iterations — exactly the lever prompt caching targets in Experiment 2. Separately, the output cap caused the answer to truncate (see finding #4) — a real product trade-off.

### Decision

**Actions for Experiment 2:**
- Expand `max_tokens_per_turn` from 2048 → 4096 so synthesis doesn't truncate
- Add explicit `stop_reason` branching (handle `end_turn` / `tool_use` / `max_tokens` separately) so truncation surfaces as a warning
- Add a reasoning-first instruction to the system prompt so iterations 1–3 emit visible "Thought:" text
- Implement Anthropic prompt caching for the system prompt + tool definitions (they don't change across iterations)
- Add streaming for synthesis output (UX improvement, not wall-clock)

**Hypothesis for Experiment 2:**
- Prompt caching cuts input cost by ~50% and shaves 10–15% off total wall-clock latency
- Streaming drops time-to-first-token from ~5 s to ~1 s without changing total latency
- To meaningfully cut wall-clock latency further, would need to either cap chunks fed to the final synthesis turn, or split routing/synthesis across model tiers (Haiku for tool-routing, Sonnet for synthesis)

## Experiment 2 — Latency & cost optimization (prompt caching)

**Date:** 2026-05-27
**Result file:** */stage_2/results/agent_runs/run_20260528_201638.json*

### Setup

- Expanded `max_tokens_per_turn` from 2048 → 4096
- Added explicit `stop_reason` branching (handle `end_turn` / `tool_use` / `max_tokens` separately) so truncation surfaces as a warning
- Added "Always begin each turn with one or two sentence of reasoning, stated explicitly. For example: "I need to find papers about X because Y." This reasoning must appear as text BEFORE any tool call." into `SYSTEM_PROMPT`
- Added a single rolled-forward `cache_control` breakpoint on the last content block of the last message each turn (caches tools + system + conversation-so-far)

### Hypothesis

Agent should output "Thought: I should do X to get Y..." at the beginning of every iteration. Latency should be reduced by ~50%-60%, estimated cost also should be lower.

### Result

- **Outcome:** Agent outputs complete and accurate result.
- **Total wall-clock:** **61.762 s** — LLM 59.849 s (97%), retrieve 1.912 s (3%), overhead ~0 s
- **Token usage:** input(full) = 1021 cache_write = 31679 cache_read = 20103 output = 2707
    - **Total input processed:** 52,803 (vs 1,021 billed at full rate)
- **Estimated cost:** $0.169
- **Iterations:** 3

**Per-iteration breakdown**

| Iteration | LLM time | In tokens | Out tokens | Cache write | Cache read |
|-----------|----------|-----------|------------|-------------|------------|
| 1         | 3.09 s   | 1019      | 166        | 0           | 0          |
| 2         | 4.14 s   | 1         | 180        | 20103       | 0          |
| 3         | 52.62 s  | 1         | 2361       | 11576       | 20103      |

**Cost estimate** *(Sonnet 4.6 pricing: $3 / $15 per 1M input / output; cache write 1.25×, cache read 0.1×)*

| Type         | Tokens  | Rate        | Cost       |
|--------------|---------|-------------|------------|
| Input (full) | 1,021   | $3.00 / 1M  | $0.0031    |
| Cache write  | 31,679  | $3.75 / 1M  | $0.1188    |
| Cache read   | 20,103  | $0.30 / 1M  | $0.0060    |
| Output       | 2,707   | $15.00 / 1M | $0.0406    |
| **Total**    |         |             | **$0.169** |

> Pricing reference: https://platform.claude.com/docs/en/about-claude/pricing
> Note: cache write ($0.119) is the largest line item — the savings come from the *read* on the synthesis turn, not the writes.

### Comparison

| Experiment | Total time | Total estimated cost | Latency Improvement | Cost improvement |
|------------|------------|----------------------|---------------------|------------------|
| 1          | 118.3 s    | $0.317               | `None`              | `None`           |
| 2          | **61.762 s** | **$0.169**         | **47.8%**           | **46.7%**        |

### Interpretation

Both latency and cost are reduced, several key findings:

1. **Prompt cache worked:** Latency reduced **~48%** (118.3 s → 61.8 s). Evidence the rolling breakpoint behaved exactly as designed: iteration 2 *wrote* 20,103 tokens to cache, and iteration 3 *read back* exactly 20,103 — the prior turn's prefix served at 0.1×.

2. **ReAct behavior worked:** Agent now outputs "Thought: ..." at the beginning of every iteration.

3. **Cost saving:** Estimated cost dropped **~47%** ($0.317 → $0.169). Caveat: the cache *write* (31,679 tokens at 1.25×) is the single largest line item; the savings come from the *read* on the synthesis turn, not the writes.

4. **Unnecessary content in response:** the agent's synthesis is too long / verbose.

5. **Remaining latency is output-bound.** Iteration 3 alone is 52.6 s of the 61.8 s total — now almost entirely *output generation* (2,361 tokens), which caching cannot speed up (caching only accelerates input prefill). This is why the result (~48%) landed just under the 50–60% hypothesis: the input-prefill time caching removed has been replaced by output-generation time as the dominant cost. It also means finding #4 is a latency lever — a shorter synthesis would cut the largest remaining time sink.

### Decision

**Actions for Experiment 3:**
- Prompt engineering to structure the agent's response (currently too verbose — see finding #4)

**Hypothesis for Experiment 3:**
- Response should be structured and easier to read; a shorter synthesis should also trim output-generation time (the dominant remaining latency cost — see finding #5)