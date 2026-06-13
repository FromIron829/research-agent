# Durable vector store for the WRITABLE corpora (4.3): per-tenant overlays + episodic memory.
#   DATABASE_URL set  -> pgvector on the same RDS as the checkpointer/auth/feedback (durable).
#   absent            -> Chroma on local disk (dev), unchanged behavior.
# The frozen, reproducible BASE corpus is intentionally NOT here — it stays on Chroma (retrieve.py),
# because durability only matters for data you can't rebuild from committed artifacts.
#
# PgCollection mimics the exact subset of the Chroma Collection API the callers use
# (count / upsert / add / query / get / delete), with identical return shapes, so graph.py and
# episodic.py are backend-agnostic. Cosine distance matches: Chroma cosine and pgvector `<=>`
# (vector_cosine_ops) both return 1 - cosine_similarity, so callers' `1.0 - dist` is unchanged.
import os

DIM = 1536   # text-embedding-3-small

_pg_conn = None
_chroma_cache = {}


def _pg():
    global _pg_conn
    if _pg_conn is None:
        import psycopg
        from pgvector.psycopg import register_vector
        _pg_conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
        _pg_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(_pg_conn)               # adapt python list <-> pgvector
        _pg_conn.execute(
            f"""CREATE TABLE IF NOT EXISTS vec_store (
                  collection TEXT NOT NULL,
                  id         TEXT NOT NULL,
                  document   TEXT,
                  metadata   JSONB,
                  embedding  vector({DIM}),
                  PRIMARY KEY (collection, id))""")
        _pg_conn.execute("CREATE INDEX IF NOT EXISTS vec_store_coll ON vec_store (collection)")
        _pg_conn.execute("CREATE INDEX IF NOT EXISTS vec_store_hnsw ON vec_store "
                         "USING hnsw (embedding vector_cosine_ops)")
    return _pg_conn


class PgCollection:
    """A Chroma-Collection-shaped view over rows of vec_store namespaced by `collection`."""
    def __init__(self, conn, name):
        self.conn, self.name = conn, name

    @staticmethod
    def _where(where):
        if not where:
            return "", []
        clause, params = "", []
        for k, v in where.items():
            clause += " AND metadata->>%s = %s"     # callers only use equality filters
            params += [k, str(v)]
        return clause, params

    def count(self):
        return self.conn.execute("SELECT count(*) FROM vec_store WHERE collection=%s",
                                 (self.name,)).fetchone()[0]

    def upsert(self, ids, embeddings, documents=None, metadatas=None):
        from psycopg.types.json import Jsonb
        from pgvector import Vector
        documents = documents if documents is not None else [None] * len(ids)
        metadatas = metadatas if metadatas is not None else [{}] * len(ids)
        with self.conn.cursor() as cur:
            for i, cid in enumerate(ids):
                cur.execute(
                    "INSERT INTO vec_store (collection, id, document, metadata, embedding) "
                    "VALUES (%s,%s,%s,%s,%s) "
                    "ON CONFLICT (collection, id) DO UPDATE SET "
                    "document=EXCLUDED.document, metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding",
                    (self.name, cid, documents[i], Jsonb(metadatas[i]), Vector(embeddings[i])))

    add = upsert   # episodic uses uuids -> upsert-as-add is safe

    def query(self, query_embeddings, n_results=10, where=None):
        from pgvector import Vector
        clause, params = self._where(where)
        rows = self.conn.execute(
            "SELECT id, document, metadata, embedding <=> %s AS dist FROM vec_store "
            "WHERE collection=%s" + clause + " ORDER BY dist LIMIT %s",
            [Vector(query_embeddings[0]), self.name] + params + [n_results]).fetchall()
        return {"ids": [[r[0] for r in rows]], "documents": [[r[1] for r in rows]],
                "metadatas": [[r[2] for r in rows]], "distances": [[float(r[3]) for r in rows]]}

    def get(self, ids=None, where=None, include=None):
        clause, params = self._where(where)
        if ids:
            clause += " AND id = ANY(%s)"
            params.append(list(ids))
        rows = self.conn.execute(
            "SELECT id, document, metadata FROM vec_store WHERE collection=%s" + clause,
            [self.name] + params).fetchall()
        return {"ids": [r[0] for r in rows], "documents": [r[1] for r in rows],
                "metadatas": [r[2] for r in rows]}

    def delete(self, ids=None, where=None):
        clause, params = self._where(where)
        if ids:
            clause += " AND id = ANY(%s)"
            params.append(list(ids))
        self.conn.execute("DELETE FROM vec_store WHERE collection=%s" + clause, [self.name] + params)


def get_collection(name, space="cosine"):
    """Durable pgvector collection in prod; Chroma collection in local dev."""
    if os.environ.get("DATABASE_URL"):
        return PgCollection(_pg(), name)
    import chromadb
    from retrieve import CHROMA_DIR
    if name not in _chroma_cache:
        _chroma_cache[name] = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_or_create_collection(
            name, metadata={"hnsw:space": space})
    return _chroma_cache[name]
