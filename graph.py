"""LangGraph pipeline.

    retrieve -> route -> run_tool -> approve (human-in-the-loop) -> answer

Four deliberate design decisions, each one a thing clients get burned on:

1. retrieval is hybrid and returns the source text, so every claim can be traced back
2. routing is rule-based, not model-based -- the model does not get to decide whether
   a number needs arithmetic. Numbers always go through a deterministic tool.
3. any write action passes through interrupt() and waits for a human. The graph cannot
   silently mutate state.
4. the answer node receives computed facts, never raw text, for the numeric part.
"""

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

import llm
from config import TOP_K
from store import get_store
from tools import compare_invoices, extract_fields, flag_for_review

COMPARE_HINTS = ("compare", "higher", "cheaper", "expensive", "difference", "delta", "which one", "对账", "比较")
FLAG_HINTS = ("flag", "on hold", "hold payment", "block", "review", "标记", "挂起")


class State(TypedDict, total=False):
    question: str
    docs: List[Dict[str, Any]]
    tool_name: Optional[str]
    tool_result: Optional[Dict[str, Any]]
    approval: Optional[Dict[str, Any]]
    answer: str
    trace: List[str]


def _log(state: State, line: str) -> List[str]:
    return (state.get("trace") or []) + [line]


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------
def retrieve(state: State) -> Dict[str, Any]:
    store = get_store()
    docs = store.search(state["question"], k=TOP_K)
    return {
        "docs": docs,
        "trace": _log(state, f"retrieve: {len(docs)} chunks, top={docs[0]['doc_id'] if docs else None}"),
    }


def route(state: State) -> Dict[str, Any]:
    q = state["question"].lower()
    has_amounts = any(extract_fields(d["text"])["amounts"] for d in state.get("docs", []))
    invoice_count = len({d["doc_id"] for d in state.get("docs", [])})

    if any(h in q for h in FLAG_HINTS):
        return {"tool_name": "flag_for_review", "trace": _log(state, "route -> flag_for_review")}
    if any(h in q for h in COMPARE_HINTS) and has_amounts and invoice_count >= 2:
        return {"tool_name": "compare_invoices", "trace": _log(state, "route -> compare_invoices")}
    return {"tool_name": None, "trace": _log(state, "route -> no tool (retrieval-only answer)")}


def run_tool(state: State) -> Dict[str, Any]:
    name = state.get("tool_name")
    docs = state.get("docs", [])

    if name == "compare_invoices":
        result = compare_invoices(docs)
        return {"tool_result": result, "trace": _log(state, "tool: compare_invoices (deterministic)")}

    if name == "flag_for_review":
        target = extract_fields(docs[0]["text"])["invoice_numbers"]
        invoice_id = target[0] if target else docs[0]["doc_id"]
        action = flag_for_review(invoice_id, "amount deviation vs. peer invoice")
        return {"tool_result": action, "trace": _log(state, "tool: flag_for_review (proposed write)")}

    return {"tool_result": None, "trace": _log(state, "tool: skipped")}


def approve(state: State) -> Dict[str, Any]:
    action = state["tool_result"]
    decision = interrupt(
        {
            "type": "approval_required",
            "action": action["action"],
            "params": action["params"],
            "effect": action["effect"],
            "reversible": action["reversible"],
        }
    )
    return {"approval": decision, "trace": _log(state, f"approve: {decision.get('approved')}")}


def answer(state: State) -> Dict[str, Any]:
    docs = state.get("docs", [])
    result = state.get("tool_result")
    approval = state.get("approval")

    if result and result.get("action"):
        if approval and approval.get("approved"):
            body = (
                f"Action `{result['action']}` approved for {result['params']['invoice_id']}. "
                f"Effect: {result['effect']}."
            )
            if approval.get("note"):
                body += f" Reviewer note: {approval['note']}"
        else:
            body = f"Action `{result['action']}` was not approved; nothing was written."
        body += "\n\nSources:\n" + "\n".join(f"- {d['source']} ({d['doc_id']})" for d in docs)
        return {"answer": body, "trace": _log(state, "answer: action outcome")}

    if result:
        prose = llm.summarise(result, state["question"])
        body = prose
        if not result.get("ok"):
            body += "\n\nComparison refused: " + str(result.get("reason"))
        else:
            body += "\n\nFigures:\n" + "\n".join(
                f"- {r['invoice'] or r['doc_id']}: {r['currency']} {r['amount']:,.2f}"
                + (f", tax {r['tax_rate']:g}%" if r.get("tax_rate") else "")
                for r in result.get("rows", [])
                if r.get("amount") is not None
            )
    else:
        body = "Retrieved context (no numeric tool applied):\n" + "\n".join(
            f"- {d['title']}: {d['text'][:160].replace(chr(10), ' ')}..." for d in docs
        )

    body += "\n\nSources:\n" + "\n".join(f"- {d['source']} ({d['doc_id']})" for d in docs)
    return {"answer": body, "trace": _log(state, "answer: composed from deterministic facts")}


def _after_route(state: State):
    return "run_tool" if state.get("tool_name") else "answer"


def _after_tool(state: State):
    return "approve" if (state.get("tool_result") or {}).get("action") else "answer"


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("route", route)
    g.add_node("run_tool", run_tool)
    g.add_node("approve", approve)
    g.add_node("answer", answer)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "route")
    g.add_conditional_edges("route", _after_route, {"run_tool": "run_tool", "answer": "answer"})
    g.add_conditional_edges("run_tool", _after_tool, {"approve": "approve", "answer": "answer"})
    g.add_edge("approve", "answer")
    g.add_edge("answer", END)

    return g.compile(checkpointer=MemorySaver())


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def ask(question: str, thread_id: str = "demo", auto_approve: Optional[bool] = None):
    """Run one question through the graph.

    Returns (answer, trace, interrupted_action).
    If the graph hits the approval node, pass auto_approve=True to resume it --
    the resume goes through Command(resume=...), exactly what a UI approve button sends.
    """
    app = get_graph()
    cfg = {"configurable": {"thread_id": thread_id}}
    state = app.invoke({"question": question, "trace": []}, cfg)

    pending = state.get("__interrupt__")
    if pending:
        payload = pending[0].value if hasattr(pending[0], "value") else pending[0]
        if auto_approve is None:
            return None, state.get("trace", []), payload
        decision = {"approved": bool(auto_approve), "note": "auto-approved by demo runner"}
        state = app.invoke(Command(resume=decision), cfg)

    return state.get("answer", ""), state.get("trace", []), None
