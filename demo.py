"""Scripted demo -- this is the one to run on camera.

Shows four things in order:
  1. hybrid retrieval + deterministic arithmetic on invoice amounts
  2. a write action suspending the graph for human approval
  3. a human approves, the graph resumes
  4. a human denies, nothing is written

Run: python demo.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")

from graph import ask
from ingest import ingest


def section(n, title):
    print("\n" + "=" * 68)
    print(f"  {n}. {title}")
    print("=" * 68)


def show(trace):
    for line in trace:
        print("  [trace]", line)


def main():
    count = ingest(reset=True)
    print(f"indexed {count} documents (hybrid: vector rank + BM25 rank, fused with RRF)")

    section(1, "Ask a question that needs arithmetic")
    q1 = "Compare INV-2026-0413 and INV-2026-0418. Which is higher and by how much?"
    answer, trace, _ = ask(q1, thread_id="demo-1")
    show(trace)
    print("\n" + answer)

    section(2, "Propose a write action -- the graph must stop and wait")
    q2 = "Flag INV-2026-0418 for finance review because 6 units failed inspection."
    answer, trace, pending = ask(q2, thread_id="demo-2")
    show(trace)
    print("\n  SUSPENDED. Waiting for a human:")
    for k, v in pending.items():
        print(f"    {k}: {v}")

    section(3, "A human approves -- graph resumes")
    answer, trace, _ = ask(q2, thread_id="demo-2", auto_approve=True)
    show(trace)
    print("\n" + answer)

    section(4, "A human denies -- nothing is written")
    answer, trace, _ = ask(q2, thread_id="demo-4", auto_approve=False)
    show(trace)
    print("\n" + answer)

    print("\n" + "=" * 68)
    print("  Takeaway: the model wrote zero numbers. Every figure came from a")
    print("  parser or a subtraction. Writes cannot happen without approval.")
    print("=" * 68)


if __name__ == "__main__":
    main()
