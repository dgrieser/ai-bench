#!/usr/bin/env python3
"""Render the queue that `update-all --collect-prompts` left behind.

Groups the questions by the local command that asks them, so the report doubles
as a to-do list: run that command in a terminal and answer what it asks.

The markdown output carries no timestamp on purpose -- the GitHub Actions
workflow diffs it against the live issue body to decide whether the queue
actually changed, so it must only differ when the questions differ.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from typing import Any

import _prompts

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
        choices=("markdown", "text"),
        default="text",
        help="Output format (default: text).",
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

    renderer = render_markdown if args.format == "markdown" else render_text
    print(renderer(entries), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
