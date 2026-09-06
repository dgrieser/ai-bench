#!/usr/bin/env python3
"""Answer the pipeline's queued questions without a terminal or a pull request.

`update-all --collect-prompts` leaves a queue of questions nobody answered, and
propose.py turns it into a reviewable PR. This is the direct route: hand it the
answers and it writes them through each source's own writer.

  ./answer.py --stdin < answers.json          a batch, as JSON
  ./answer.py tbench "Fable 5.1" __unmappable__   one answer, by hand
  ./answer.py --stdin -w < answers.json       and actually write it

Dry run by default, like every other script here: without -w it validates the
whole batch and prints what it would do. A batch is all or nothing -- one bad
record and nothing is written -- because half an answered queue is the one
outcome nobody could reconstruct afterwards.

The record shapes are documented in _answers.py, which is also where the
validation that makes this safe to expose lives.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _answers
import propose
from _answers import AnswerError

HERE = Path(__file__).resolve().parent
DEFAULT_QUEUE = HERE / "_pending" / "pending.json"
LEGACY_QUEUE = HERE / "pending-prompts.jsonl"


def route_from_shorthand(name: str) -> str:
    """'tbench' -> 'update_tbench_mapping.py', if that is unambiguous."""
    if name in propose.ROUTES:
        return name
    exact = f"update_{name}_mapping.py"
    if exact in propose.ROUTES:
        return exact
    matches = [r for r in propose.ROUTES if name in r]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"error: no route matches {name!r}")
    raise SystemExit(f"error: {name!r} matches several routes: {', '.join(sorted(matches))}")


def default_queue() -> Path | None:
    for candidate in (DEFAULT_QUEUE, LEGACY_QUEUE):
        if candidate.exists():
            return candidate
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply answers to the pipeline's queued questions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "shorthand",
        nargs="*",
        metavar="ROUTE SUBJECT ANSWER",
        help="One mapping answer by hand, e.g. tbench 'Fable 5.1' __unmappable__",
    )
    parser.add_argument("--stdin", action="store_true", help="Read a JSON batch from stdin.")
    parser.add_argument("--json", metavar="FILE", help="Read a JSON batch from FILE.")
    parser.add_argument(
        "-w", "--write", action="store_true", help="Actually write. Without it, nothing changes."
    )
    parser.add_argument(
        "--llm-json", default=str(_answers.DEFAULT_LLM_JSON), help="Path to llm.json."
    )
    parser.add_argument(
        "--queue",
        metavar="FILE",
        help=f"The queue an answer must appear in (default: {DEFAULT_QUEUE.name} or "
        f"{LEGACY_QUEUE.name}, whichever exists).",
    )
    parser.add_argument(
        "--no-queue-check",
        action="store_true",
        help="Allow answers to names the queue never asked about. For answering by hand; "
        "never pass it for input that came from somewhere else.",
    )
    parser.add_argument(
        "--result", metavar="FILE", help="Write a machine-readable result summary to FILE."
    )
    return parser.parse_args(argv)


def read_records(args: argparse.Namespace) -> list:
    sources = [bool(args.stdin), bool(args.json), bool(args.shorthand)]
    if sum(sources) != 1:
        raise SystemExit("error: give exactly one of --stdin, --json FILE, or ROUTE SUBJECT ANSWER")

    if args.shorthand:
        if len(args.shorthand) != 3:
            raise SystemExit("error: the shorthand form takes exactly ROUTE SUBJECT ANSWER")
        route, subject, value = args.shorthand
        return [
            {
                "kind": _answers.MAPPING,
                "route": route_from_shorthand(route),
                "route_kind": "*",
                "subject": subject,
                "answer": value,
                # Answering by hand means answering what the file says now.
                "if_previous": _answers.current_value(
                    _answers.resolve_route(route_from_shorthand(route), "*"), subject
                ),
            }
        ]

    text = sys.stdin.read() if args.stdin else Path(args.json).read_text(encoding="utf-8")
    try:
        records = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: the batch is not valid JSON: {exc}")
    # A single record is a convenient shorthand for a batch of one.
    return records if isinstance(records, list) else [records]


def write_result(path: str, ok: bool, log: list[str], failures: list) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "ok": ok,
                "applied": log,
                "errors": [{"record": f.index, "message": f.message} for f in failures],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = read_records(args)

    queue_path = args.queue or (None if args.no_queue_check else default_queue())
    if not args.no_queue_check and queue_path is None:
        raise SystemExit(
            "error: no queue to check against. Run update-all --collect-prompts first, "
            "or pass --no-queue-check to answer a name it never asked about."
        )

    try:
        queue = _answers.load_queue(queue_path)
        answers, failures = _answers.validate(
            records,
            llm_path=Path(args.llm_json),
            queue=queue,
            require_queue=not args.no_queue_check,
        )
    except AnswerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if failures:
        print(f"{len(failures)} record(s) rejected; nothing was written:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        if args.result:
            write_result(args.result, False, [], failures)
        return 1

    if not args.write:
        print(f"Would apply {len(answers)} answer(s):")
        for answer in answers:
            print(f"  {answer.describe()}")
        print("\nNothing written. Pass -w to apply.")
        if args.result:
            write_result(args.result, True, [a.describe() for a in answers], [])
        return 0

    try:
        log = _answers.apply(answers, llm_path=Path(args.llm_json))
    except AnswerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("Nothing was written: the batch was rolled back.", file=sys.stderr)
        if args.result:
            write_result(args.result, False, [], [_answers.Failure(0, str(exc))])
        return 1

    print(f"Applied {len(log)} answer(s):")
    for line in log:
        print(f"  {line}")
    if args.result:
        write_result(args.result, True, log, [])
    return 0


def run() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(run())
