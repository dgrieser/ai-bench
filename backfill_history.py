#!/usr/bin/env python3
"""Seed models[].scores_history in llm.json from the file's own git history.

``scores_history`` starts empty, and from then on _history.sync() keeps it as
scores move. This is the one-time step that gives it a past: every committed
revision of llm.json is replayed in order through the very same sync(), with
the commit's date standing in for today, so the entries it lays down obey the
rules in _history.py rather than a second copy of them written here.

What the replay can and cannot recover:

* A score that changed while the file was under version control yields a real
  transition -- the old number, the new one, and the day it moved.
* A score already present in the earliest revision yields one entry, dated by
  its own ``scores_updated`` stamp and credited to its own ``scores_source``.
  That stamp reaches back well before the repository does, so the day the score
  was last written survives; what it replaced, if anything, does not. The
  earliest revision's date is written to llm.json as ``history_since``, and the
  site reads an entry older than that as a first sighting rather than as an
  arrival -- the write it stands for may well have been an update whose old
  value nobody kept.
* A model renamed before the earliest revision it appears under is followed
  only from that name onwards -- the replay keys by ``name``, which is what
  every other consumer of llm.json does.

Default is a dry-run; pass -w/--write to persist changes (same convention as
prune.py, sync_score_dates.py and update.py).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import _history

DEFAULT_LLM_JSON = Path(__file__).resolve().parent / "llm.json"
JSON_DUMP_KWARGS = {"indent": 2, "ensure_ascii": False}


def revisions(path: Path) -> list[tuple[str, str]]:
    """(commit, ISO date) for every revision of `path`, oldest first.

    Author dates, not committer dates: the day the data was collected is the
    day the refresh ran, and a rebase must not move a score's history.
    """
    result = subprocess.run(
        ["git", "log", "--reverse", "--format=%H %ad", "--date=short", "--", path.name],
        cwd=path.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    revs = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            revs.append((parts[0], parts[1]))
    return revs


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


def replay(path: Path, head: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Walk every revision of llm.json, returning name -> history map.

    The accumulated history rides along from revision to revision: each doc is
    handed the history built so far, sync() extends it with whatever that
    revision changed, and the result is carried to the next one. The working
    copy is replayed last, so uncommitted edits are recorded too.
    """
    carried: dict[str, dict[str, Any]] = {}

    for commit, when in revisions(path):
        doc = revision_doc(path, commit)
        if doc is None:
            continue
        carried = _apply(doc, carried, when)

    # The head revision may already be committed, in which case this is a
    # no-op; when it is not, it is the last day of the replay.
    carried = _apply(head, carried, None)
    return carried


def _apply(
    doc: dict[str, Any],
    carried: dict[str, dict[str, Any]],
    when: str | None,
) -> dict[str, dict[str, Any]]:
    for model in doc.get("models") or []:
        name = model.get("name")
        if isinstance(name, str) and name in carried:
            model[_history.HISTORY_FIELD] = json.loads(json.dumps(carried[name]))
        else:
            model.pop(_history.HISTORY_FIELD, None)

    _history.sync(doc, today=when)

    harvested = dict(carried)
    for model in doc.get("models") or []:
        name = model.get("name")
        if not isinstance(name, str):
            continue
        stored = model.get(_history.HISTORY_FIELD)
        if isinstance(stored, dict) and stored:
            harvested[name] = stored
        else:
            harvested.pop(name, None)
    return harvested


def with_history_since(doc: dict[str, Any], since: str) -> dict[str, Any]:
    """The document with history_since set, placed among the other preamble
    keys rather than after the models array a reader has to scroll past."""
    rebuilt: dict[str, Any] = {}
    for key, value in doc.items():
        if key == _history.SINCE_FIELD:
            continue
        rebuilt[key] = value
        if key == "defaultSort":
            rebuilt[_history.SINCE_FIELD] = since
    if _history.SINCE_FIELD not in rebuilt:
        rebuilt = {_history.SINCE_FIELD: since, **rebuilt}
    return rebuilt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(DEFAULT_LLM_JSON),
        help='Path to JSON file to read/update (default: "./llm.json" next to this script)',
    )
    parser.add_argument(
        "--write",
        "-w",
        action="store_true",
        help="Write changes back to the input JSON file (default is dry-run).",
    )
    args = parser.parse_args()

    path = Path(args.json_file).resolve()
    doc = json.loads(path.read_text(encoding="utf-8"))

    revs = revisions(path)
    if not revs:
        print(f"No git history for {path.name}; nothing to replay.", file=sys.stderr)
        return 1
    print(f"Replaying {len(revs)} revision(s) of {path.name}, {revs[0][1]} to {revs[-1][1]}")

    built = replay(path, json.loads(json.dumps(doc)))

    models = 0
    keys = 0
    total = 0
    transitions = 0
    for model in doc.get("models") or []:
        name = model.get("name")
        log = built.get(name) if isinstance(name, str) else None
        if not log:
            model.pop(_history.HISTORY_FIELD, None)
            continue
        models += 1
        keys += len(log)
        for entries in log.values():
            total += len(entries)
            transitions += len(entries) - 1
        _history._place(model, log)

    # The replay reconstructs history from the scores as they were; the file
    # being written is the one on disk now, so sync() has the last word on it.
    _history.sync(doc)
    doc = with_history_since(doc, revs[0][1])
    problems = _history.validate(doc)

    print(
        f"{total} entr(ies) across {keys} score(s) of {models} model(s); "
        f"{transitions} recorded change(s) beyond each score's first sighting"
    )
    for problem in problems:
        print(f"WARN  {problem}", file=sys.stderr)

    if not args.write:
        print("\ndry-run only, pass --write to persist changes")
        return 0

    path.write_text(json.dumps(doc, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
