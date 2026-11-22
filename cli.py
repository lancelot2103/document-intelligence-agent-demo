"""CLI.

  python cli.py ingest
  python cli.py ask "Compare INV-2026-0413 and INV-2026-0418"
  python cli.py ask "Flag INV-2026-0418 for review"            # stops at approval
  python cli.py ask "Flag INV-2026-0418 for review" --approve  # resumes and completes
"""

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--approve", action="store_true")
    a.add_argument("--deny", action="store_true")
    args = p.parse_args()

    if args.cmd == "ingest":
        from ingest import ingest

        print(f"indexed {ingest()} documents")
        return

    from ingest import ingest
    from graph import ask

    ingest(reset=True)

    if not args.approve and not args.deny:
        answer, trace, pending = ask(args.question)
    else:
        answer, trace, pending = ask(args.question, auto_approve=args.approve)

    print("\n--- trace ---")
    for line in trace:
        print(" ", line)

    if pending:
        print("\n--- APPROVAL REQUIRED (graph suspended) ---")
        for k, v in pending.items():
            print(f"  {k}: {v}")
        print("\nRe-run with --approve or --deny to resume.")
        return

    print("\n--- answer ---")
    print(answer)


if __name__ == "__main__":
    main()
