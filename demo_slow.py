"""Slow-paced demo for screen recording.

Same pipeline as demo.py, but prints in stages with pauses so a viewer can read
each step as it happens. No extra logic -- every number still comes from tools.py.
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from graph import ask
from ingest import ingest

PAUSE = {
    "header": 2.0,
    "after_retrieve": 2.5,
    "after_route": 2.5,
    "after_tool": 3.0,
    "after_answer": 4.0,
    "suspend": 6.0,
    "approved": 5.0,
    "denied": 5.0,
    "takeaway": 8.0,
}


def line(s=""):
    print(s, flush=True)


def banner(title):
    line("=" * 78)
    line(f"  {title}")
    line("=" * 78)


def main():
    line()
    banner("DOCUMENT INTELLIGENCE AGENT  /  deterministic tools + human approval")
    line()
    line("  question:")
    line('    "Compare INV-2026-0413 and INV-2026-0418. Which is higher and by how much?"')
    line()
    time.sleep(PAUSE["header"])

    count = ingest(reset=True)
    line(f"  indexed {count} documents  |  hybrid retrieval: vector rank + BM25 rank, fused with RRF")
    line()
    time.sleep(PAUSE["after_retrieve"])

    q1 = "Compare INV-2026-0413 and INV-2026-0418. Which is higher and by how much?"
    answer, trace, _ = ask(q1, thread_id="rec-1")

    line("  trace:")
    for t in trace:
        line(f"    > {t}")
        time.sleep(0.8 if "route" in t else 0.35)
    time.sleep(PAUSE["after_route"])

    line()
    line("  answer:")
    for row in answer.splitlines():
        line(f"    {row}")
        time.sleep(0.35)
    line()
    line("  -> 19,120.00 - 18,450.00 = 670.00     the model computed no number here")
    line()
    time.sleep(PAUSE["after_answer"])

    banner("NOW ASK IT TO CHANGE SOMETHING")
    line()
    q2 = "Flag INV-2026-0418 for finance review because 6 units failed inspection."
    line('  question: "Flag INV-2026-0418 for finance review because 6 units failed inspection."')
    line()
    time.sleep(2.0)

    _, trace2, pending = ask(q2, thread_id="rec-2")
    line("  trace:")
    for t in trace2:
        line(f"    > {t}")
        time.sleep(0.5)

    line()
    line("  *** GRAPH SUSPENDED -- nothing has been written ***")
    for k, v in pending.items():
        line(f"    {k}: {v}")
        time.sleep(0.6)
    line()
    line("  it stops here and waits for a person.")
    time.sleep(PAUSE["suspend"])

    banner("A HUMAN APPROVES")
    answer3, trace3, _ = ask(q2, thread_id="rec-2", auto_approve=True)
    for t in trace3[-2:]:
        line(f"    > {t}")
        time.sleep(0.6)
    line()
    line("  " + answer3.splitlines()[0])
    line()
    time.sleep(PAUSE["approved"])

    banner("A HUMAN DENIES")
    answer4, trace4, _ = ask(q2, thread_id="rec-4", auto_approve=False)
    for t in trace4[-2:]:
        line(f"    > {t}")
        time.sleep(0.6)
    line()
    line("  " + answer4.splitlines()[0])
    line()
    time.sleep(PAUSE["denied"])

    line()
    line("  every number came from a parser or a subtraction.")
    line("  every write waited for a human.")
    line()
    line("  github.com/lancelot2103/document-intelligence-agent-demo")
    line()
    time.sleep(PAUSE["takeaway"])


if __name__ == "__main__":
    main()
