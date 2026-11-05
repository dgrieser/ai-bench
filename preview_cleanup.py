#!/usr/bin/env python3
"""List the preview models that look released, and what cleaning each up means.

  ./preview_cleanup.py                 a readable summary
  ./preview_cleanup.py --json          the report, as the admin page reads it
  ./preview_cleanup.py --publish       write it to _pending/previews.json

Reads only. The cleanup itself is a `preview-cleanup` record, sent from the
admin page's Previews tab or by hand:

  echo '[{"kind": "preview-cleanup", "name": "hy4-preview", "rename": "hy4",
          "fields": {"url": "https://huggingface.co/tencent/Hy4"},
          "scores": {"hle": {"do": "recredit", "url": "https://huggingface.co/tencent/Hy4"}},
          "park": [{"route": "update_openrouter_mapping.py", "subject": "tencent/hy4-preview"}]}]' \\
    | ./answer.py --stdin --no-queue-check -w

  echo '[{"kind": "preview-cleanup", "name": "step-5-preview", "action": "dismiss"}]' \\
    | ./answer.py --stdin --no-queue-check -w

See _preview_cleanup.py for what is detected and why.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _preview_cleanup as cleanup


def summary(report: dict) -> str:
    lines: list[str] = []
    cases = report["cases"]
    if not cases:
        return "No preview entries to clean up.\n"
    for case in cases:
        head = case["name"]
        if case["released"]:
            head += f"  (released {case['released']})"
        elif not case["evidence"]:
            head += "  (no sign of a release yet)"
        lines.append(head)
        for item in case["evidence"]:
            lines.append(f"  + {item['text']}")
        if case["renames"]:
            names = ", ".join(r["name"] + (" (taken)" if r["taken"] else "") for r in case["renames"])
            lines.append(f"  rename to: {names}")
        if case["new_url"]:
            lines.append(f"  url: {case['url']} -> {case['new_url']}")
        acted = [s for s in case["scores"] if s["suggest"] != cleanup.KEEP]
        preview = [s for s in case["scores"] if s["era"] == cleanup.ERA_PREVIEW]
        lines.append(f"  scores: {len(case['scores'])}, {len(preview)} from the preview, "
                     f"{len(acted)} to re-credit")
        for row in acted:
            lines.append(f"    {row['key']} {row['value']}: {row['source']} -> {row['recredit_to']}")
        parks = [m for m in case["mappings"] if m["suggest"] == cleanup.PARK]
        for m in parks:
            lines.append(f"  park: {m['source']} {m['subject']!r}")
        for q in case["queue"]:
            lines.append(f"  queued: {q['route']} {q['subject']!r}")
        lines.append("")
    if report["done"]:
        lines.append("Checked off: " + ", ".join(d["name"] for d in report["done"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List preview models that look released.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    parser.add_argument(
        "--publish", nargs="?", const=str(cleanup.DEFAULT_REPORT), metavar="FILE",
        help=f"Write the report for the admin page (default {cleanup.DEFAULT_REPORT.relative_to(cleanup.HERE)}).",
    )
    parser.add_argument("--llm-json", default=str(cleanup.DEFAULT_LLM_JSON))
    parser.add_argument("--queue", default=str(cleanup.DEFAULT_QUEUE))
    args = parser.parse_args(argv)

    report = cleanup.detect(cleanup.Inputs.from_repo(Path(args.llm_json), Path(args.queue)))
    if args.publish:
        changed = cleanup.write_report(report, Path(args.publish))
        print(f"{args.publish}: {len(report['cases'])} case(s){'' if changed else ', unchanged'}")
        return 0
    if args.json:
        json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0
    sys.stdout.write(summary(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
