# Document Intelligence Agent — a working reference implementation

A small, runnable reference for one specific claim:

> An agent that touches money should not be allowed to produce numbers, and should not be
> allowed to write anything without a human saying yes.

This repo is that claim, in code, in about 600 lines. It runs offline with no API key,
and every figure it prints can be verified by hand.

▶ **[Watch the 2-minute narrated walkthrough](https://github.com/lancelot2103/document-intelligence-agent-demo/releases/download/v1.0/agent-demo-walkthrough.mp4)**

---

## What it actually does

Take two invoices and a question:

```
Compare INV-2026-0413 and INV-2026-0418. Which is higher and by how much?
```

Output:

```
INV-2026-0413 is USD 18,450.00; INV-2026-0418 is USD 19,120.00 (tax 7%).
INV-2026-0418 is the higher one, by USD 670.00 (3.63%).

Figures:
- INV-2026-0413: USD 18,450.00
- INV-2026-0418: USD 19,120.00, tax 7%

Sources:
- invoice_0413.txt (invoice-0413)
- invoice_0418.txt (invoice-0418)
```

Check the arithmetic yourself: `19,120 − 18,450 = 670`, and `670 / 18,450 = 3.63%`.
The model never computed either number. A regex extracted them, a subtraction compared them.

---

## Architecture

```
                    ┌──────────────┐
   question ───────▶│   retrieve   │  hybrid: vector rank + BM25 rank, fused with RRF
                    └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │    route     │  rule-based. decides whether arithmetic is needed.
                    └──────┬───────┘  the model does not get this vote.
                 ┌─────────┴─────────┐
                 ▼                   ▼
          ┌─────────────┐      ┌───────────┐
          │  run_tool   │      │  answer   │
          │ deterministic│     └───────────┘
          └──────┬──────┘
                 ▼
          ┌─────────────┐
          │   approve   │  interrupt() — the graph suspends. nothing is written.
          └──────┬──────┘  a human resume command is required to continue.
                 ▼
             answer
```

Four decisions worth reading the code for:

| Decision | Where | Why it matters |
|---|---|---|
| Hybrid retrieval with RRF | `store.py` | cosine and BM25 scores live on different scales; normalising and summing them is where hybrid pipelines quietly break |
| Numbers never come from the LLM | `tools.py`, `llm.py` | an LLM asked "which is higher" answers confidently and is sometimes wrong; a parser is right or it throws |
| Permission filter inside the SQL query | `store.py` → `PG_QUERY` | a post-filter still puts the restricted chunk in the context window. that is the bug most RAG systems ship |
| Writes suspend for approval | `graph.py` → `approve` | `interrupt()` suspends the graph; without a resume the write never happens |

---

## Run it

```
pip install -r requirements.txt
python demo.py
```

`demo.py` walks through four stages in order, and is the script to run on camera:

1. compare two invoices — deterministic arithmetic
2. propose a write action — **the graph suspends and prints what it wants to do**
3. human approves — the graph resumes
4. human denies — nothing is written

Single questions through the CLI:

```
python cli.py ask "Compare INV-2026-0413 and INV-2026-0418"
python cli.py ask "Flag INV-2026-0418 for finance review"            # suspends
python cli.py ask "Flag INV-2026-0418 for finance review" --approve  # resumes
```

---

## No API key required

Two defaults are deliberately offline:

- `EMBED_BACKEND=hashing` — hashing-trick token vectors. This is lexical matching, **not**
  a semantic embedding model. Stated plainly because calling it one would be dishonest.
  Switch to a real model with `EMBED_BACKEND=openai`, or drop `bge-m3` into `embed.py`.
- `LLM_BACKEND=mock` — template prose. In real mode the LLM writes the sentence around
  the figures and is explicitly instructed not to restate or recompute them.

Real model:

```
set LLM_BACKEND=openai
set EMBED_BACKEND=openai
set OPENAI_API_KEY=sk-...
```

---

## pgvector (the production path)

`PgVectorStore` implements the same interface. Switching is one environment variable:

```
set DATABASE_URL=postgresql://user:pass@host:5432/db
```

`PG_DDL` contains the table, the `tenant_id` / `visibility` permission columns, and an
HNSW index. Read `PG_QUERY`: permission filtering happens in the `WHERE` clause, not after
retrieval.

---

## Repository layout

```
config.py      environment and defaults
embed.py       embeddings (offline hashing / OpenAI)
store.py       hybrid retrieval, SQLite + pgvector backends
ingest.py      corpus indexing
tools.py       deterministic tools: field extraction, invoice comparison, flag action
llm.py         prose only — never a number
graph.py       LangGraph: retrieve → route → run_tool → approve → answer
cli.py         command line
demo.py        the scripted demo
data/          four sample documents
```

---

## Honest limitations

This is a reference implementation, not a product. Specifically:

- **Field extraction is regex.** It recognises `INV-2026-0413` style identifiers and
  standard money formatting. Real invoices need LLM extraction plus rule-based validation
  on top, run in parallel.
- **The hashing embedder has no notion of synonyms.** Words like "higher" are matched by
  BM25, not by meaning.
- **`flag_for_review` is simulated.** It does not write to a database. A real deployment
  starts a transaction after `approve`.
- **Total selection is a heuristic.** `compare_invoices` takes the largest amount in the
  document's own currency. A real system should locate the `Total due` field label first.
  This is not generalised here because invoice layouts differ per client and that mapping
  has to be built on their data, not guessed in advance.
- **No PDF parsing, no OCR, no multi-language.** That is the next layer, not this one.
