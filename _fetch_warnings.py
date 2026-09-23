#!/usr/bin/env python3
"""Fetcher failures as warnings: recorded during a refresh, annotated on CI.

A source whose page moved, whose host is down or whose API refused is one
source to look at, not a reason to red the whole run. update.py already skips
such a source and writes every other score; what used to follow was a non-zero
exit, a red run, and a scheduled refresh that looked broken every three hours
until someone fixed a scraper they may not control.

So those failures are recorded here instead, one JSON line each, in the file
AI_BENCH_FETCH_WARNINGS names:

  update.py          one line per source whose fetch raised (source_data())
  update-all         one line per update_*_mapping.py that exited non-zero --
                     each of those is a scrape of one board's names

and the workflow turns every line into a `::warning` annotation titled
"Fetcher failed: ...". The admin page reads those annotations back through
api.php and shows them beside the run they came from.

Everything else that fails -- an ingest bug, a refused value, the index fit,
check_new.py -- still fails the run: those are this repository's bugs, and a
warning is how they would go unnoticed.

    _fetch_warnings.py record FILE --step S [--source NAME] --message M
    _fetch_warnings.py annotate FILE [--summary PATH]
    _fetch_warnings.py routes FILE      the update_*_mapping.py steps that failed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ENV_VAR = "AI_BENCH_FETCH_WARNINGS"
# What api.php filters the run's annotations on. Anything else a run annotates
# -- a deprecated action, a runner notice -- is not a fetcher and stays off the
# page.
TITLE_PREFIX = "Fetcher failed"
# update-all's exit status when every failure was a fetcher: distinct from 1 so
# the workflow can tell "some sources are stale" from "something is broken".
WARNINGS_ONLY_EXIT = 3


def target() -> Path | None:
    value = os.environ.get(ENV_VAR, "").strip()
    return Path(value) if value else None


def record(step: str, message: str, source: str | None = None, path: Path | None = None) -> None:
    """Append one failure. A no-op when nothing asked for the file.

    Never raises: this runs on the failure path, and a warning that could not be
    written must not turn into a second, unrelated failure.
    """
    path = path or target()
    if path is None:
        return
    entry = {"step": step, "source": source, "message": message.strip()}
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"warning: could not record a fetch warning in {path}: {exc}", file=sys.stderr)


def load(path: Path) -> list[dict[str, Any]]:
    """Every recorded failure, in order, skipping lines that do not parse."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get("step"):
            entries.append(entry)
    return entries


def title(entry: dict[str, Any]) -> str:
    return f"{TITLE_PREFIX}: {entry.get('source') or entry['step']}"


def _escape_data(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(value: str) -> str:
    return _escape_data(value).replace(":", "%3A").replace(",", "%2C")


def annotation(entry: dict[str, Any]) -> str:
    """The workflow command that makes one entry a warning on the run."""
    message = entry.get("message") or f"{entry['step']} failed; see the log."
    return f"::warning title={_escape_property(title(entry))}::{_escape_data(message)}"


def summary(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return ""
    lines = [
        "## Fetcher warnings",
        "",
        f"{len(entries)} source(s) could not be read this run. Their scores stay "
        "where the last good run put them; everything else landed.",
        "",
    ]
    for entry in entries:
        # The last line: for a traceback that is the exception, which names the cause.
        message = (entry.get("message") or "").strip().splitlines()
        last = message[-1] if message else "see the log"
        lines.append(f"- **{entry.get('source') or entry['step']}** ({entry['step']}): {last}")
    return "\n".join(lines) + "\n"


def failed_routes(entries: list[dict[str, Any]]) -> list[str]:
    """The mapping updaters that failed: whose questions this run never asked."""
    return sorted({
        entry["step"] for entry in entries
        if entry["step"].startswith("update_") and entry["step"].endswith(".py")
        and not entry.get("source")
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="Append one fetcher failure.")
    rec.add_argument("file")
    rec.add_argument("--step", required=True)
    rec.add_argument("--source")
    rec.add_argument("--message", default="")
    ann = sub.add_parser("annotate", help="Print a ::warning per recorded failure.")
    ann.add_argument("file")
    ann.add_argument("--summary", metavar="PATH", help="Also append a markdown summary here.")
    rts = sub.add_parser("routes", help="Print the mapping updaters that failed, one per line.")
    rts.add_argument("file")
    args = parser.parse_args()

    if args.command == "record":
        record(args.step, args.message, args.source, Path(args.file))
        return 0
    entries = load(Path(args.file))
    if args.command == "routes":
        for route in failed_routes(entries):
            print(route)
        return 0
    for entry in entries:
        print(annotation(entry))
    if args.summary and entries:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(summary(entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
