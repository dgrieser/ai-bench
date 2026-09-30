#!/usr/bin/env python3
"""Keep _log/mappings.json, the record of who changed which mapping and when.

  ./mapping_log.py record --by auto -w                  what the working tree
                                                        changes since HEAD
  ./mapping_log.py record --by admin --batch R.json -w  the same, classified
                                                        against answer.py's result
  ./mapping_log.py commit SHA -w                        what one commit changed,
                                                        e.g. a merged pull request
  ./mapping_log.py backfill -w                          seed from git history

Dry run by default, like every other script here: without -w it prints what it
would add. The workflow calls `record` just before each of its commits; see
_mapping_log.py for the entry shape and what each `by` means.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _mapping_log


def describe(entry: dict) -> str:
    source = entry["route"].removeprefix("update_").removesuffix("_mapping.py")
    why = f"  ({entry['why']})" if entry.get("why") else ""
    return (f"{entry['at']}  {entry['by']:<8} {source}: {entry['subject']!r} "
            f"{entry.get('previous')!r} -> {entry.get('value')!r}{why}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="Log what the working tree changes since a revision.")
    rec.add_argument("--by", required=True, choices=sorted(_mapping_log.BY),
                     help="Who an unexplained change is credited to.")
    rec.add_argument("--base", default="HEAD", help="The revision to compare against (default: HEAD).")
    rec.add_argument("--batch", metavar="FILE",
                     help="answer.py's --result file: changes it asked for are credited to "
                     "admin, the rest are its side effects.")
    rec.add_argument("--run", metavar="URL", help="The workflow run making the change.")

    one = sub.add_parser("commit", help="Log what one commit changed against its first parent.")
    one.add_argument("sha")
    one.add_argument("--by", choices=sorted(_mapping_log.BY),
                     help="Who made it (default: read off its author and subject).")

    back = sub.add_parser("backfill", help="Rebuild the log from the first-parent history.")
    back.add_argument("--rev", default="HEAD", help="Where the history ends (default: HEAD).")

    for command in (rec, one, back):
        command.add_argument("-w", "--write", action="store_true", help="Actually write the log.")
        command.add_argument("--log", default=str(_mapping_log.LOG_PATH), help="The log file.")
    args = parser.parse_args(argv)
    log_path = Path(args.log)

    if args.command == "record":
        batch = _mapping_log.Batch.from_result(Path(args.batch)) if args.batch else None
        found = _mapping_log.record(args.by, base=args.base, batch=batch, run=args.run)
        entries = _mapping_log.load(log_path) + found
    elif args.command == "commit":
        found = _mapping_log.commit_changes(args.sha, by=args.by)
        existing = _mapping_log.load(log_path)
        # A re-run of the same merge's workflow must not log it twice.
        short = args.sha[:7]
        if any(e.get("commit") == short for e in existing):
            found = []
        entries = existing + found
    else:
        found = _mapping_log.backfill(args.rev)
        # Replaces the file rather than merging into it: it is for seeding the
        # log once, and a live entry carries no commit to deduplicate against.
        entries = found

    for entry in found[-50:]:
        print(describe(entry))
    if len(found) > 50:
        print(f"... and {len(found) - 50} earlier")
    print(f"{len(found)} mapping change(s).", file=sys.stderr)
    if not args.write:
        print("Nothing written. Pass -w to write.", file=sys.stderr)
        return 0
    if found or args.command == "backfill":
        _mapping_log.save(entries, log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
