"""Index the sample corpus. Incremental: doc_id is derived from the file name."""

import os
import re

from config import DATA_DIR
from embed import embed_text
from store import Doc, get_store


def load_documents(data_dir: str = DATA_DIR):
    docs = []
    for name in sorted(os.listdir(data_dir)):
        if not name.lower().endswith((".txt", ".md")):
            continue
        path = os.path.join(data_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        title = text.splitlines()[0].lstrip("# ").strip() if text else name
        doc_id = re.sub(r"[^a-zA-Z0-9]+", "-", os.path.splitext(name)[0]).lower()
        docs.append(
            Doc(doc_id=doc_id, title=title, source=name, text=text, embedding=embed_text(text))
        )
    return docs


def ingest(reset: bool = True):
    store = get_store()
    if reset and hasattr(store, "reset"):
        store.reset()
    docs = load_documents()
    for d in docs:
        store.upsert(d)
    return len(docs)


if __name__ == "__main__":
    n = ingest()
    print(f"indexed {n} documents")
