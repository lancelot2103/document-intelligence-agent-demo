import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
DB_PATH = os.path.join(ROOT, "index.db")

TOP_K = int(os.getenv("TOP_K", "4"))

# Embedding backend: "hashing" (zero-cost, offline, deterministic) or "openai"
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "hashing")

# LLM backend: "mock" (offline template) or "openai"
LLM_BACKEND = os.getenv("LLM_BACKEND", "mock")

# Optional pgvector backend. Leave empty to use SQLite.
DATABASE_URL = os.getenv("DATABASE_URL", "")
