# Short-term (conversation) memory helpers.
MAX_TURNS = 6

def format_history(history: list[dict], max_turns: int = MAX_TURNS):
    recent = history[-max_turns * 2:]
    if not recent:
        return ""
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in recent]
    return "Conversation so far:\n" + "\n".join(lines) + "\n"
