#!/usr/bin/env python3
"""Render the queue that `update-all --collect-prompts` left behind.

Groups the questions by the local command that asks them, so the report doubles
as a to-do list: run that command in a terminal and answer what it asks.

`--format json` is the same queue for something that is not a person reading a
PR: it carries the route each question belongs to, the candidates ranked by
_matching.grade() with the reason each one matched, and what the mapping file
says today. answer.py checks an answer's subject against it, and a page can
render it as a form.

No output here carries a timestamp, on purpose. The markdown is diffed against
the live PR body to decide whether the queue actually changed, and the JSON is
committed, so both must differ only when the questions differ.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import _matching
import _prompts
import propose

# Enough to choose from without the tail reshuffling every time a model is
# added: grade() returns a total order, so a fixed cut is stable.
MAX_CANDIDATES = 5

KIND_LABELS = {
    "new-model": "new model",
    "mapping": "mapping",
    "aa-mapping": "Artificial Analysis mapping",
    "llmstats-model": "llm-stats model",
    "llmstats-benchmark": "llm-stats benchmark",
}


def group(entries: list[dict[str, Any]]) -> "OrderedDict[str, list[dict[str, Any]]]":
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for entry in sorted(entries, key=lambda e: (e.get("command", ""), e.get("subject", ""))):
        grouped.setdefault(entry.get("command") or "(unknown)", []).append(entry)
    return grouped


def render_markdown(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "Nothing pending -- every source name and benchmark label is mapped.\n"

    lines = [
        f"@dgrieser -- **{len(entries)}** question(s) the daily run could not answer.",
        "",
        "Scores in `llm.json` are up to date. These need a person: run the command "
        "for a section in a terminal and answer its prompts.",
        "",
    ]
    for command, items in group(entries).items():
        lines.append(f"### `{command}` &mdash; {len(items)}")
        lines.append("")
        for entry in items:
            subject = entry.get("subject") or "?"
            kind = KIND_LABELS.get(entry.get("kind", ""), entry.get("kind", ""))
            lines.append(f"- **`{subject}`** ({kind})")
            note = entry.get("note")
            if note:
                lines.append(f"  {note}")
            default = entry.get("default")
            if default:
                lines.append(f"  suggested: `{default}`")
            others = [c for c in entry.get("candidates") or [] if c != default]
            if others:
                lines.append("  candidates: " + ", ".join(f"`{c}`" for c in others))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_text(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "nothing pending\n"

    lines = [f"{len(entries)} pending prompt(s):", ""]
    for command, items in group(entries).items():
        lines.append(f"{command}  ({len(items)})")
        for entry in items:
            subject = entry.get("subject") or "?"
            hint = entry.get("default") or ", ".join(entry.get("candidates") or [])
            lines.append(f"  {subject}" + (f"  -> {hint}" if hint else ""))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_json(entries: list[dict[str, Any]], llm_path: Path, skip_aa: bool) -> str:
    """The queue as data: one question per entry, with its route and candidates.

    Universes are deliberately left out. The model list alone is hundreds of
    names and the Artificial Analysis list is thousands, both of which change on
    their own schedule -- committing them would rewrite this file on runs where
    no question changed, and empty it entirely whenever AA was unreachable. A
    page already has llm.json, so it can derive the choices; answer.py
    revalidates server-side, which is the check that counts.
    """
    universes = propose.build_universes(llm_path, skip_aa=skip_aa)
    questions = []
    for entry in sorted(
        entries,
        key=lambda e: (e.get("command", ""), e.get("kind", ""), e.get("subject", "")),
    ):
        subject = entry.get("subject") or ""
        route_name = propose.script_of(entry)
        route = propose.route_for(entry)
        candidates: list[dict[str, str]] = []
        previous = None
        if route is not None:
            options = universes.get(route.universe) or entry.get("candidates") or []
            candidates = [
                {"option": m.option, "confidence": m.confidence, "reason": m.reason}
                for m in _matching.grade(propose.match_subject(entry, route), options)[
                    :MAX_CANDIDATES
                ]
            ]
            previous = _answers_current_value(route, subject)
        questions.append(
            {
                "route": route_name,
                "route_kind": entry.get("kind") or "",
                "subject": subject,
                "question": entry.get("question") or "",
                "note": entry.get("note"),
                "default": entry.get("default"),
                # What the file says now, so an answer can be rejected as stale
                # rather than clobbering whoever got there first.
                "if_previous": previous,
                "candidates": candidates or [
                    {"option": c, "confidence": "recorded", "reason": "offered by the prompt"}
                    for c in (entry.get("candidates") or [])[:MAX_CANDIDATES]
                ],
            }
        )
    return (
        json.dumps({"questions": questions}, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )


def _answers_current_value(route, subject: str):
    """_answers.current_value, imported late to keep this script standalone."""
    import _answers

    return _answers.current_value(route, subject)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        nargs="?",
        help=f"Path to the collected report (default: ${_prompts.ENV_VAR} "
        f"or ./{_prompts.DEFAULT_REPORT.name}).",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        help="Write the output to FILE instead of stdout, creating its directory.",
    )
    parser.add_argument(
        "--llm-json",
        default=str(propose.DEFAULT_LLM_JSON),
        help="Path to llm.json, for the candidate lists (--format json).",
    )
    parser.add_argument(
        "--skip-aa",
        action="store_true",
        help="Do not fetch the Artificial Analysis slug list (--format json).",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Print only the number of pending prompts.",
    )
    args = parser.parse_args()

    entries = _prompts.load(args.report)
    if args.count:
        print(len(entries))
        return 0

    if args.format == "json":
        output = render_json(entries, Path(args.llm_json), args.skip_aa)
    elif args.format == "markdown":
        output = render_markdown(entries)
    else:
        output = render_text(entries)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
