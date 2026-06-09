# Short-term (conversation) memory helpers.
MAX_TURNS = 6

def format_history(history: list[dict], summary: str = "", max_turns: int = MAX_TURNS):
    """Context = running summary of older turns (if any) + the last `max_turns` turns verbatim."""
    recent = history[-max_turns * 2:]
    parts = []
    if summary:
        parts.append("Summary of earlier conversation:\n" + summary)
    if recent:
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in recent]
        parts.append("Conversation so far:\n" + "\n".join(lines))
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n"

def summarize_history(history, prior_summary, n_summarized, client, model, max_turns: int = MAX_TURNS):
    """Fold turns that have fallen OUT of the recent window into a running summary.

    Incremental: only messages not yet summarized (index >= n_summarized) are folded, so no
    turn is summarized twice. Returns (summary, n_summarized). Keeps context bounded while
    preserving older context in compressed form — unlike raw truncation, which drops it.
    """
    evictable = history[:-max_turns * 2] if max_turns > 0 else list(history)
    new_msgs = evictable[n_summarized:]
    if not new_msgs:
        return prior_summary, n_summarized

    convo = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in new_msgs)
    prior = f"Existing summary:\n{prior_summary}\n\n" if prior_summary else ""
    msg = client.messages.create(
        model=model, max_tokens=300,
        messages=[{"role": "user", "content":
                   f"{prior}New conversation turns to fold in:\n{convo}\n\n"
                   "Update the running summary so it preserves the key topics, entities (papers / "
                   "techniques), and facts discussed, in at most 5-8 sentences. Output only the summary."}],
    )
    new_summary = "".join(b.text for b in msg.content if b.type == "text").strip()
    return new_summary, len(evictable)
