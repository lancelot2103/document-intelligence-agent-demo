"""Document store with hybrid retrieval.

One interface, two backends:
  SQLiteStore    -- default, zero external services, good up to ~50k chunks
  PgVectorStore  -- production path, real ANN index, enforced inside Postgres

Hybrid retrieval = vector rank + BM25 rank, fused with Reciprocal Rank Fusion (RRF).
RRF is used instead of weighted score addition because cosine scores and BM25 scores
live on different scales; normalising them is where most hybrid pipelines quietly break.
"""

import json
import math
import re
import sqlite3
from dataclasses import dataclass

import numpy as np

from config import DB_PATH

TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\-\.]*")
STOP = set(
    """a an and are as at be by for from has have in is it its of on or that the
    this to was were will with""".split()
)


def tokenize(text: str):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


@dataclass
class Doc:
    doc_id: str
    title: str
    source: str
    text: str
    embedding: np.ndarray


# ---------------------------------------------------------------------------
# SQLite backend
# -----------------------------------------------------------------------------
class SQLiteStore:
    def __init__(self, path: str = DB_PATH):
        self.path = path
        self._init()

    def _init(self):
        con = sqlite3.connect(self.path)
        con.execute(
            """CREATE TABLE IF NOT EXISTS docs (
                doc_id TEXT PRIMARY KEY,
                title TEXT,
                source TEXT,
                text TEXT,
                embedding BLOB
            )"""
        )
        con.commit()
        con.close()

    def reset(self):
        con = sqlite3.connect(self.path)
        con.execute("DELETE FROM docs")
        con.commit()
        con.close()

    def upsert(self, doc: Doc):
        con = sqlite3.connect(self.path)
        con.execute(
            "INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?)",
            (doc.doc_id, doc.title, doc.source, doc.text, doc.embedding.tobytes()),
        )
        con.commit()
        con.close()

    def all_docs(self):
        con = sqlite3.connect(self.path)
        rows = con.execute("SELECT doc_id,title,source,text,embedding FROM docs").fetchall()
        con.close()
        out = []
        for doc_id, title, source, text, blob in rows:
            out.append(Doc(doc_id, title, source, text, np.frombuffer(blob, dtype=np.float32)))
        return out

    # -- retrieval ---------------------------------------------------------
    def search(self, query: str, k: int = 4, rrf_k: int = 60):
        from embed import embed_text

        docs = self.all_docs()
        if not docs:
            return []

        q_vec = embed_text(query)
        vec_scores = np.array([float(np.dot(q_vec, d.embedding)) for d in docs])
        bm25 = _bm25_scores(query, [d.text for d in docs])

        vec_rank = _ranks(vec_scores)
        kw_rank = _ranks(bm25)

        fused = 1.0 / (rrf_k + vec_rank) + 1.0 / (rrf_k + kw_rank)

        order = np.argsort(-fused)[:k]
        results = []
        for idx in order:
            d = docs[int(idx)]
            results.append(
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "source": d.source,
                    "text": d.text,
                    "score": round(float(fused[int(idx)]), 4),
                    "vector_score": round(float(vec_scores[int(idx)]), 4),
                    "bm25_score": round(float(bm25[int(idx)]), 4),
                }
            )
        return results


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank, higher score -> smaller rank."""
    order = np.argsort(-scores)
    ranks = np.empty_like(scores, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _bm25_scores(query: str, texts, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    n = len(texts)
    if n == 0:
        return np.zeros(0)
    doc_tokens = [tokenize(t) for t in texts]
    avgdl = sum(len(t) for t in doc_tokens) / n
    df = {}
    for toks in doc_tokens:
        for w in set(toks):
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    q_terms = tokenize(query)
    scores = np.zeros(n, dtype=np.float32)
    for i, toks in enumerate(doc_tokens):
        counts = {}
        for w in toks:
            counts[w] = counts.get(w, 0) + 1
        dl = len(toks)
        s = 0.0
        for w in q_terms:
            f = counts.get(w, 0)
            if f == 0:
                continue
            denom = f + k1 * (1 - b + b * dl / max(avgdl, 1.0))
            s += idf.get(w, 0.0) * f * (k1 + 1) / denom
        scores[i] = s
    return scores


# ---------------------------------------------------------------------------
# pgvector backend (production path)
# -----------------------------------------------------------------------------
PG_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    doc_id      TEXT PRIMARY KEY,
    title       TEXT,
    source      TEXT,
    text        TEXT,
    -- tenant / permission columns live here, not in the prompt
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    visibility  TEXT NOT NULL DEFAULT 'public',
    embedding   vector(1536)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""

PG_QUERY = """
SELECT doc_id, title, source, text,
       1 - (embedding <=> %(qvec)s) AS vector_score
FROM chunks
WHERE tenant_id = %(tenant_id)s
  AND (%(visibility)s = 'internal' OR visibility = 'public')
ORDER BY embedding <=> %(qvec)s
LIMIT %(k)s;
"""


class PgVectorStore:
    """Same interface as SQLiteStore.

    Note what the WHERE clause does: permission filtering is part of the query,
    not a post-filter on the LLM output. A post-filter still leaks the chunk into
    the context window, which is the bug most RAG systems ship.
    """

    def __init__(self, dsn: str):
        import psycopg

        self.dsn = dsn
        self._psycopg = psycopg

    def _con(self):
        return self._psycopg.connect(self.dsn)

    def create_schema(self):
        with self._con() as con, con.cursor() as cur:
            cur.execute(PG_DDL)

    def upsert(self, doc: Doc):
        with self._con() as con, con.cursor() as cur:
            cur.execute(
                """INSERT INTO chunks (doc_id,title,source,text,embedding)
                   VALUES (%s,%s,%s,%s,%s)
                   ON CONFLICT (doc_id) DO UPDATE
                   SET text = EXCLUDED.text, embedding = EXCLUDED.embedding""",
                (doc.doc_id, doc.title, doc.source, doc.text, doc.embedding.tolist()),
            )

    def search(self, query: str, k: int = 4, tenant_id: str = "default", visibility: str = "public"):
        from embed import embed_text

        qvec = embed_text(query).tolist()
        with self._con() as con, con.cursor() as cur:
            cur.execute(
                PG_QUERY,
                {"qvec": qvec, "k": k, "tenant_id": tenant_id, "visibility": visibility},
            )
            rows = cur.fetchall()
        return [
            {"doc_id": r[0], "title": r[1], "source": r[2], "text": r[3], "score": r[4]}
            for r in rows
        ]


def get_store():
    import os

    dsn = os.getenv("DATABASE_URL", "")
    if dsn:
        return PgVectorStore(dsn)
    return SQLiteStore()
