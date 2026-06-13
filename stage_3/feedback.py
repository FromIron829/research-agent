# Online feedback loop: response log + user 👍/👎 (Roadmap 4.1a).
# Every terminal answer is logged with the attribution needed to explain a later
# thumbs-down (prompt_version, model, grounded, tokens). Feedback rows reference a
# response_id and are ownership-checked, mirroring auth.py's thread model.
# DATABASE_URL -> Postgres (durable, same RDS as auth + checkpointer), absent -> SQLite (dev).
import os
import secrets
import sqlite3
from pathlib import Path
from datetime import datetime, timezone


def _conn():
    url = os.environ.get("DATABASE_URL")
    if url:
        from psycopg import connect
        return connect(url, autocommit=True), "%s"
    db = sqlite3.connect(str(Path(__file__).resolve().parent / "feedback.db"),
                         check_same_thread=False, isolation_level=None)
    return db, "?"


_db, _P = _conn()


def _exec(sql, params=()):
    cur = _db.cursor()
    cur.execute(sql.replace("?", _P), params)
    return cur


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init():
    """Create the response log + feedback tables."""
    _exec("""CREATE TABLE IF NOT EXISTS responses (
               response_id    TEXT PRIMARY KEY,
               thread_id      TEXT,
               key_id         TEXT,
               tenant         TEXT,
               question       TEXT,
               answer         TEXT,
               prompt_version TEXT,
               model          TEXT,
               grounded       INTEGER,
               tokens_used    INTEGER,
               created_at     TEXT NOT NULL)""")
    _exec("""CREATE TABLE IF NOT EXISTS feedback (
               feedback_id TEXT PRIMARY KEY,
               response_id TEXT NOT NULL,
               key_id      TEXT,
               rating      INTEGER NOT NULL,   -- +1 thumbs up / -1 thumbs down
               comment     TEXT,
               created_at  TEXT NOT NULL)""")


def log_response(thread_id, key_id, tenant, question, answer,
                 prompt_version, model, grounded=None, tokens_used=None) -> str:
    """Record a terminal answer; return its response_id (the feedback join key)."""
    rid = "r_" + secrets.token_hex(8)
    _exec("INSERT INTO responses VALUES (?,?,?,?,?,?,?,?,?,?,?)",
          (rid, thread_id, key_id, tenant, question, answer,
           prompt_version, model,
           None if grounded is None else int(bool(grounded)),
           tokens_used, _now()))
    return rid


def response_owner(response_id):
    """key_id that produced this response, or None if the response is unknown."""
    row = _exec("SELECT key_id FROM responses WHERE response_id = ?", (response_id,)).fetchone()
    return row[0] if row else None


def record_feedback(response_id, key_id, rating, comment=None) -> str:
    """Persist one 👍/👎. Caller must validate ownership first (see api.py)."""
    fid = "f_" + secrets.token_hex(8)
    _exec("INSERT INTO feedback VALUES (?,?,?,?,?,?)",
          (fid, response_id, key_id, 1 if rating > 0 else -1, comment, _now()))
    return fid


def quality_summary():
    """Online monitor + A/B comparison: overall 👍/👎 and a per-prompt_version breakdown
    (responses, avg tokens, satisfaction) — the head-to-head a prompt rollout needs."""
    total = _exec("SELECT COUNT(*) FROM responses").fetchone()[0]
    up = _exec("SELECT COUNT(*) FROM feedback WHERE rating > 0").fetchone()[0]
    down = _exec("SELECT COUNT(*) FROM feedback WHERE rating < 0").fetchone()[0]

    by_version = {}
    for pv, cnt, avg_tok in _exec(
            "SELECT prompt_version, COUNT(*), AVG(tokens_used) FROM responses "
            "GROUP BY prompt_version").fetchall():
        by_version[pv or "unknown"] = {
            "responses": cnt, "avg_tokens": round(avg_tok, 1) if avg_tok else None,
            "up": 0, "down": 0, "satisfaction": None}
    for pv, u, d in _exec(
            "SELECT r.prompt_version, "
            "SUM(CASE WHEN f.rating > 0 THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN f.rating < 0 THEN 1 ELSE 0 END) "
            "FROM responses r JOIN feedback f ON f.response_id = r.response_id "
            "GROUP BY r.prompt_version").fetchall():
        v = by_version.setdefault(pv or "unknown",
                                  {"responses": 0, "avg_tokens": None, "up": 0, "down": 0})
        v["up"], v["down"] = u or 0, d or 0
        v["satisfaction"] = round(v["up"] / (v["up"] + v["down"]), 3) if (v["up"] + v["down"]) else None

    rated = up + down
    return {
        "responses_logged": total,
        "feedback_total": rated,
        "thumbs_up": up,
        "thumbs_down": down,
        "satisfaction": round(up / rated, 3) if rated else None,
        "by_prompt_version": by_version,
    }
