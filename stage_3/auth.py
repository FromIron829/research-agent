# API-key auth + thread ownership
# DATABASE_URL -> Postgres, absent -> local SQLite file (dev).
import os
import secrets
import hashlib
import sqlite3
from pathlib import Path

def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def _conn():
    url = os.environ.get("DATABASE_URL")
    if url:
        from psycopg import connect
        return connect(url, autocommit=True), "%s"
    db = sqlite3.connect(str(Path(__file__).resolve().parent / "auth.db"),
                         check_same_thread=False, isolation_level=None)
    return db, "?"

_db, _P = _conn()

def _exec(sql, params=()):
    cur = _db.cursor()
    cur.execute(sql.replace("?", _P), params)
    return cur

def init():
    """Create tables; bootstrap the admin key from env if it isn't registered yet."""
    _exec("""CREATE TABLE IF NOT EXISTS api_keys (
               key_id TEXT PRIMARY KEY, key_hash TEXT UNIQUE NOT NULL,
               name TEXT NOT NULL, is_admin INTEGER NOT NULL DEFAULT 0)""")
    _exec("""CREATE TABLE IF NOT EXISTS thread_owners (
               thread_id TEXT PRIMARY KEY, key_id TEXT NOT NULL)""")
    admin = os.environ.get("ADMIN_KEY")
    if admin and not lookup(admin):
        _exec("INSERT INTO api_keys VALUES (?, ?, ?, 1)", ("admin", _hash(admin), "admin"))
        print("[auth] bootstrapped admin key from env")

def lookup(key: str):
    """Presented key -> (key_id, is_admin) or None."""
    row = _exec("SELECT key_id, is_admin FROM api_keys WHERE key_hash = ?", (_hash(key),)).fetchone()
    return (row[0], bool(row[1])) if row else None

def mint(name: str) -> str:
    """Create a key; return the plaintext ONCE (only the hash is stored)."""
    key = "ra_" + secrets.token_urlsafe(24)
    key_id = "k_" + secrets.token_hex(6)
    _exec("INSERT INTO api_keys VALUES (?, ?, ?, 0)", (key_id, _hash(key), name))
    return key

def claim_thread(thread_id: str, key_id: str):
    _exec("INSERT INTO thread_owners VALUES (?, ?)", (thread_id, key_id))

def thread_owner(thread_id: str):
    row = _exec("SELECT key_id FROM thread_owners WHERE thread_id = ?", (thread_id,)).fetchone()
    return row[0] if row else None
