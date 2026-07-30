#!/usr/bin/env python3
"""Collect mode: queue the pipeline's questions instead of asking them.

`update-all --collect-prompts` runs the whole pipeline unattended (a cron job, a
GitHub Actions run) and needs the scores without a human at the keyboard. Every
prompt in the pipeline is then recorded here as one JSON line and answered with
"no answer", and -- the part that matters -- *no decision is persisted*: the
mapping-file writers turn into no-ops via freeze_decisions(), so the same
questions come back on the next local interactive run. Scores still land in
llm.json, so the benchmarks stay current while new models and new mappings wait
for a person.

Activated by AI_BENCH_PENDING_PROMPTS (holding the report path) rather than a
module flag, so it survives the check_new.py -> add.py subprocess hop. The
--collect-prompts CLI flag added by add_cli_flag() only sets that variable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

ENV_VAR = "AI_BENCH_PENDING_PROMPTS"
DEFAULT_REPORT = Path(__file__).resolve().with_name("pending-prompts.jsonl")

# (command, kind, subject) already written by this process. The report is
# append-only across processes, so load() dedupes again on read.
_seen: set[tuple[str, str, str]] = set()


def collecting() -> Path | None:
    """Report path when collect mode is on, else None."""
    value = os.environ.get(ENV_VAR, "").strip()
    return Path(value) if value else None


def freeze_decisions() -> bool:
    """True when no answer may be persisted (alias of collecting(), read at the writers)."""
    return collecting() is not None


def enable(path: str | Path | None = None) -> Path:
    """Turn on collect mode for this process and every subprocess it spawns."""
    report = Path(path).expanduser().resolve() if path else DEFAULT_REPORT
    os.environ[ENV_VAR] = str(report)
    return report


def reset(path: str | Path | None = None) -> Path:
    """Truncate the report so a fresh run does not inherit a stale queue."""
    report = Path(path).expanduser().resolve() if path else DEFAULT_REPORT
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("", encoding="utf-8")
    _seen.clear()
    return report


def default_command() -> str:
    """The local command that would ask this question, for the report."""
    name = Path(sys.argv[0]).name or "update-all"
    write_flag = " -w" if {"-w", "--write"} & set(sys.argv[1:]) else ""
    return f"./{name}{write_flag}"


def record(
    *,
    kind: str,
    subject: str,
    question: str,
    candidates: Iterable[str] = (),
    default: str | None = None,
    note: str | None = None,
    command: str | None = None,
) -> None:
    """Queue one unanswered question. No-op outside collect mode."""
    report = collecting()
    if report is None:
        return

    entry = {
        "command": command or default_command(),
        "kind": kind,
        "subject": subject,
        "question": question,
        "candidates": [c for c in candidates],
        "default": default,
        "note": note,
    }
    key = (entry["command"], kind, subject)
    if key in _seen:
        return
    _seen.add(key)

    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Read the report, dropping duplicates from repeated runs of the same script."""
    report = Path(path) if path else (collecting() or DEFAULT_REPORT)
    if not report.exists():
        return []

    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for line in report.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        key = (
            str(entry.get("command", "")),
            str(entry.get("kind", "")),
            str(entry.get("subject", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        entries.append(entry)
    return entries


def add_cli_flag(parser: argparse.ArgumentParser) -> None:
    """Add --collect-prompts to a script that prompts."""
    parser.add_argument(
        "--collect-prompts",
        metavar="FILE",
        nargs="?",
        const=str(DEFAULT_REPORT),
        help="Do not ask anything: append every prompt to FILE and record no "
        f"answers, so the questions are asked again next time (default: "
        f"{DEFAULT_REPORT.name}). Also picked up from ${ENV_VAR}.",
    )


def apply_cli_flag(args: argparse.Namespace) -> None:
    """Honour --collect-prompts. Call right after parse_args()."""
    requested = getattr(args, "collect_prompts", None)
    if not requested:
        return
    # A parent (update-all) that already exported the variable owns the report
    # and has truncated it; only a standalone run starts a fresh one.
    inherited = bool(os.environ.get(ENV_VAR, "").strip())
    enable(requested)
    if not inherited:
        reset(requested)
