"""Embedding backends.

Default is a deterministic hashing embedder so the demo runs offline with no API key
and no model download. It is lexical, not semantic -- on purpose:
demonstrating the pipeline, not pretending a hash is a real embedding model.

Swap to a real model by setting EMBED_BACKEND=openai (text-embedding-3-small)
or dropping in bge-m3 / multilingual-e5 inside embed_text().
"""

import hashlib
import os
import re

import numpy as np

DIM = 256
TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\-\.]*")


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def _hash(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)


def _hashing_embed(text: str, dim: int = DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = tokenize(text)
    for i, tok in enumerate(tokens):
        vec[_hash(tok) % dim] += 1.0
        # bigrams keep a little word-order signal
        if i + 1 < len(tokens):
            vec[_hash(tok + "_" + tokens[i + 1]) % dim] += 0.5
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def _openai_embed(texts):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
    if os.getenv("EMBED_BACKEND", "hashing") == "openai":
        return _openai_embed([text])[0]
    return _hashing_embed(text)


def embed_many(texts):
    if os.getenv("EMBED_BACKEND", "hashing") == "openai":
        return _openai_embed(texts)
    return np.stack([_hashing_embed(t) for t in texts])
