#!/usr/bin/env python3
"""Rebuild llm.json's index_rankings from the file's own git history.

From the day it is first written, _rankings.sync() keeps ``index_rankings`` up
to date every time the derived indexes are recomputed. This gives it a past:
every revision of llm.json committed in the last --days days is replayed in
order through that same sync(), with the commit's author date standing in for
today, so a backfilled change obeys exactly the rules a live one does.

The revision just before the window is the starting point: its ranking is
taken as given and reported as nothing, so the first change on record is the
first one inside the window. The working copy is replayed last, so the stored
ranking ends up matching the file as it stands.

Only revisions on the first-parent line of HEAD are read -- what main actually
published, one refresh after another, not the intermediate states of a merged
branch. The repository has to hold that much history; in a shallow clone,
`git fetch --shallow-since=<date> origin main` first.

A model renamed inside the window is followed under each of its names, so it
reads as leaving under the old one and entering under the new one on the day
of the rename.

Default is a dry-run; pass -w/--write to persist changes (same convention as
backfill_history.py, prune.py and update.py).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import _rankings

DEFAULT_LLM_JSON = Path(__file__).resolve().parent / "llm.json"
JSON_DUMP_KWARGS = {"indent": 2, "ensure_ascii": False}


def revisions(path: Path, since: str) -> list[tuple[str, str]]:
    """(commit, ISO date) for every first-parent revision of `path` from
    `since` on, oldest first, preceded by the last revision before it (dated
    the day before `since`) when there is one.

    Author dates, as backfill_history.py uses: the day the refresh ran.
    """
    def log(*extra: str) -> list[tuple[str, str]]:
        result = subprocess.run(
            ["git", "log", "--first-parent", "--format=%H %ad", "--date=short", *extra, "--", path.name],
            cwd=path.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return [
            (parts[0], parts[1])
            for parts in (line.split() for line in result.stdout.splitlines())
            if len(parts) == 2
        ]

    inside = [rev for rev in reversed(log(f"--since={since}T00:00:00")) if rev[1] >= since]
    before = [rev for rev in log(f"--until={since}T00:00:00", "-n", "5") if rev[1] < since]
    return before[:1] + inside


def revision_doc(path: Path, commit: str) -> dict[str, Any] | None:
    """llm.json as of one commit, or None if it cannot be read as JSON."""
    result = subprocess.run(
        ["git", "show", f"{commit}:{path.name}"],
        cwd=path.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def replay(path: Path, head: dict[str, Any], since: str) -> dict[str, Any]:
    """The index_rankings record a sync on every revision from `since` on
    would have built, ending at `head`."""
    carried: dict[str, Any] | None = None
    revs = revisions(path, since)
    if not revs:
        raise SystemExit(
            f"no revision of {path.name} since {since}; is the clone shallow? "
            f"(git fetch --shallow-since=<date> origin main)"
        )
    for position, (commit, when) in enumerate(revs):
        doc = revision_doc(path, commit)
        if doc is None:
            continue
        if carried is None:
            # The baseline: its ranking is the starting point, not news.
            doc.pop(_rankings.ROOT_FIELD, None)
            _rankings.sync(doc, today=max(when, since))
            carried = doc[_rankings.ROOT_FIELD]
            carried["since"] = since
            carried["changes"] = []
            continue
        doc[_rankings.ROOT_FIELD] = carried
        recorded = _rankings.sync(doc, today=when)
        carried = doc[_rankings.ROOT_FIELD]
        if recorded:
            print(f"  {when} {commit[:8]}: {recorded} change(s)")

    head[_rankings.ROOT_FIELD] = carried
    _rankings.sync(head)
    return head[_rankings.ROOT_FIELD]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(DEFAULT_LLM_JSON),
        help='Path to llm.json (default: "./llm.json" next to this script)',
    )
    parser.add_argument(
        "--days",
        type=int,
        default=31,
        help="How far back to replay, in days (default: 31).",
    )
    parser.add_argument(
        "--write",
        "-w",
        action="store_true",
        help="Write the rebuilt record back to the file (default is dry-run).",
    )
    args = parser.parse_args()

    path = Path(args.json_file)
    head = json.loads(path.read_text(encoding="utf-8"))
    since = (date.today() - timedelta(days=args.days)).isoformat()
    print(f"Replaying {path.name} since {since}:")
    record = replay(path, head, since)

    entries = record.get("changes") or []
    by_kind = {"entered": 0, "left": 0, "moved": 0}
    for entry in entries:
        if entry.get("from") is None:
            by_kind["entered"] += 1
        elif entry.get("to") is None:
            by_kind["left"] += 1
        else:
            by_kind["moved"] += 1
    print(
        f"\n{len(entries)} ranking change(s) since {record.get('since')}: "
        f"{by_kind['entered']} entered, {by_kind['moved']} moved, {by_kind['left']} left"
    )

    if not args.write:
        print("dry-run only, pass --write to persist")
        return 0
    path.write_text(json.dumps(head, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
