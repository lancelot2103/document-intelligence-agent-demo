"""Deterministic tools -- the core of the pitch.

Rule: any number that a client will act on is produced by a parser or arithmetic,
never by the model. The LLM writes the explanation around the numbers; it never
produces the numbers.

Why: an LLM asked "which invoice is higher" will answer confidently and sometimes
be wrong. A regex plus a subtraction is either right or throws. On procurement data
that difference is the whole product.
"""

import re
from datetime import date, datetime

MONEY_RE = re.compile(r"(?:(USD|EUR|GBP|\$|€|£)\s?)?([0-9][0-9,]*\.\d{2})")
INVOICE_RE = re.compile(r"\b(INV-[0-9]{4}-[0-9]{3,})\b")
DATE_RE = re.compile(r"\b([0-9]{4}-[0-9]{2}-[0-9]{2})\b")
PCT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s?%")
ISO_RE = re.compile(r"\b(USD|EUR|GBP|CAD|AUD|INR|CNY|JPY)\b")
CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}


def extract_fields(text: str) -> dict:
    """Pull invoice number, every money amount, every date, tax rate."""
    invoices = sorted(set(INVOICE_RE.findall(text)))

    amounts = []
    for sym, num in MONEY_RE.findall(text):
        currency = CURRENCY_SYMBOLS.get(sym, sym) or "UNSPECIFIED"
        amounts.append({"currency": currency, "value": float(num.replace(",", ""))})

    dates = []
    for raw in DATE_RE.findall(text):
        try:
            dates.append(datetime.strptime(raw, "%Y-%m-%d").date())
        except ValueError:
            continue

    tax = [float(p) for p in PCT_RE.findall(text)]

    return {
        "invoice_numbers": invoices,
        "amounts": amounts,
        "dates": dates,
        "tax_rates": tax,
    }


def compare_invoices(docs, key_amount_currency="UNSPECIFIED") -> dict:
    """Compare the amount of each retrieved invoice. Pure arithmetic, no model.

    Fails loud instead of guessing:
      - mixed currencies -> refuse to compare, report why
      - no amount found  -> report which doc is missing data
    """
    rows = []
    for d in docs:
        f = extract_fields(d["text"])
        # Only documents that carry an invoice number take part. A retrieved PO
        # or email thread must not enter the arithmetic -- a unit price is not
        # an invoice total and comparing them produces a nonsense percentage.
        if not f["invoice_numbers"]:
            continue

        doc_currency = ISO_RE.search(d["text"])
        doc_ccy = doc_currency.group(1) if doc_currency else "UNSPECIFIED"

        # The total is the largest amount carrying the document's own currency
        # (a unit price or a subtotal is smaller by construction).
        same_ccy = [a["value"] for a in f["amounts"] if a["currency"] in (doc_ccy, "UNSPECIFIED")]
        amount = max(same_ccy) if same_ccy else None

        row = {
            "doc_id": d["doc_id"],
            "invoice": f["invoice_numbers"][0],
            "amount": amount,
            "currency": doc_ccy,
            "due_dates": [str(x) for x in f["dates"]],
            "tax_rate": f["tax_rates"][0] if f["tax_rates"] else None,
        }
        rows.append(row)

    currencies = {r["currency"] for r in rows if r["currency"] and r["currency"] != "UNSPECIFIED"}
    if len(currencies) > 1:
        return {
            "ok": False,
            "reason": f"mixed currencies {sorted(currencies)}; converted values required before comparison",
            "rows": rows,
        }

    missing = [r["invoice"] or r["doc_id"] for r in rows if r["amount"] is None]
    if missing:
        return {"ok": False, "reason": f"no amount parsed in {missing}", "rows": rows}

    totals = sorted(rows, key=lambda r: r["amount"], reverse=True)
    highest, lowest = totals[0], totals[-1]
    delta = highest["amount"] - lowest["amount"]
    pct = (delta / lowest["amount"] * 100) if lowest["amount"] else None

    return {
        "ok": True,
        "currency": next(iter(currencies), "UNSPECIFIED"),
        "rows": rows,
        "highest": highest,
        "lowest": lowest,
        "delta": round(delta, 2),
        "delta_pct": round(pct, 2) if pct is not None else None,
        "computed_at": date.today().isoformat(),
    }


def flag_for_review(invoice_id: str, reason: str) -> dict:
    """Simulated write action. Requires human approval before it 'happens'."""
    return {
        "action": "flag_for_finance_review",
        "params": {"invoice_id": invoice_id, "reason": reason},
        "effect": "invoice will be queued for finance review and locked from payment until cleared",
        "reversible": True,
    }


TOOL_REGISTRY = {
    "compare_invoices": compare_invoices,
    "flag_for_review": flag_for_review,
}
