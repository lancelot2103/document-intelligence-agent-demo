"""LLM layer.

Only one job: turn already-computed facts into readable prose.
It never produces a number that ends up in the answer.

Two backends:
  mock   -- offline, deterministic, no API key. Used for the demo and for CI.
  openai -- real model, enabled by LLM_BACKEND=openai

The prompt explicitly forbids restating or recomputing figures, because a model that
"helpfully" re-derives a total is exactly how wrong numbers reach a client report.
"""

import json
import os

SYSTEM = """You write short, neutral summaries for a finance/procurement workflow.
You are given FACTS already computed by deterministic tools.
Rules:
- Never change, round, recompute or add any number.
- If a figure is absent from FACTS, say it is unavailable. Do not estimate.
- No marketing tone. State what is known, then what needs a human decision.
- Three sentences maximum.
"""


def _mock(facts: dict) -> str:
    if not facts.get("ok"):
        return (
            "The comparison could not be completed: "
            f"{facts.get('reason', 'unknown reason')}. "
            "A human needs to resolve this before any figure is trusted."
        )
    rows = facts.get("rows", [])
    high = facts.get("highest", {})
    low = facts.get("lowest", {})
    parts = [
        f"{r['invoice'] or r['doc_id']} is {facts['currency']} {r['amount']:,.2f}"
        + (f" (tax {r['tax_rate']:g}%)" if r.get("tax_rate") else "")
        for r in rows
    ]
    return (
        "; ".join(parts) + ". "
        f"{high.get('invoice')} is the higher one, by {facts['currency']} "
        f"{facts['delta']:,.2f} ({facts['delta_pct']}%). "
        "Figures were parsed from the source documents and compared arithmetically; "
        "the payment decision is yours."
    )


def summarise(facts: dict, question: str = "") -> str:
    if os.getenv("LLM_BACKEND", "mock") != "openai":
        return _mock(facts)

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"QUESTION: {question}\n\nFACTS:\n{json.dumps(facts, indent=2, default=str)}",
            },
        ],
    )
    return resp.choices[0].message.content.strip()
