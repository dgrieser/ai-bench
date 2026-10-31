#!/usr/bin/env python3
"""Replay llm.json's git history and list every column whose scores moved together.

update.py reports such a shift on the run it happens in (_switch_detector);
this is the same check after the fact, over every day the file changed: the
last revision of each day against the one before. It is how the in-place
switches already in the table were found -- AA's Intelligence Index v4.1, v4.2,
v4.3 and v5.0, AA-Omniscience v1.1, Real-SWE 1.1, GDPval-AA's re-anchoring --
and it is the first thing to run on a column whose numbers look wrong.

    ./audit_switches.py                       # every flagged shift, oldest first
    ./audit_switches.py --since 2026-10-01    # a window
    ./audit_switches.py --column aa_lcr       # one column, flagged or not

A clone has to carry the history: in a shallow one, `git fetch --unshallow`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import _switch_detector

HERE = Path(__file__).resolve().parent


def daily_revisions(ref: str, since: str | None) -> list[tuple[str, str]]:
    """(date, commit) for the last commit of each day that touched llm.json."""
    log = subprocess.run(
        ["git", "-C", str(HERE), "log", "--format=%H %ad", "--date=short", ref, "--", "llm.json"],
        capture_output=True, text=True, check=True,
    ).stdout.split("\n")
    last: dict[str, str] = {}
    for line in log:
        if line.strip():
            commit, day = line.split()
            last.setdefault(day, commit)  # newest first: the first seen is the day's last
    days = sorted(d for d in last if since is None or d >= since)
    return [(d, last[d]) for d in days]


def state(commit: str) -> tuple[dict, dict, set[str]] | None:
    """({model: scores}, {model: sources}, derived keys) at a commit, or None."""
    raw = subprocess.run(
        ["git", "-C", str(HERE), "show", f"{commit}:llm.json"], capture_output=True, text=True
    ).stdout
    try:
        doc = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(doc, dict):
        return None
    return (
        _switch_detector.doc_scores(doc),
        _switch_detector.doc_scores(doc, "scores_source"),
        _switch_detector.derived_keys(doc),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="HEAD", help="history to replay (default: HEAD)")
    parser.add_argument("--since", help="first day to report (YYYY-MM-DD)")
    parser.add_argument("--column", help="report this column only")
    args = parser.parse_args()

    revisions = daily_revisions(args.ref, None)
    previous = None
    found = 0
    for day, commit in revisions:
        current = state(commit)
        if current is None:
            continue
        if previous is not None and (args.since is None or day >= args.since):
            shifts = _switch_detector.detect(
                previous[0], current[0], skip=current[2] | previous[2],
                sources_before=previous[1], sources_after=current[1],
            )
            for shift in shifts:
                if args.column and shift.key != args.column:
                    continue
                found += 1
                print(f"{day}  {shift.describe()}")
        previous = current
    if len(revisions) < 2:
        print("fewer than two revisions of llm.json in this clone's history "
              "(a shallow clone? run `git fetch --unshallow`)", file=sys.stderr)
    print(f"{found} shift(s) over {len(revisions)} day(s) of history", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
