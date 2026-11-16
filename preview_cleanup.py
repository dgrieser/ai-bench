#!/usr/bin/env python3
"""List the preview models that look released, and what cleaning each up means.

  ./preview_cleanup.py                 a readable summary
  ./preview_cleanup.py --json          the report, as the admin page reads it
  ./preview_cleanup.py --publish       write it to _pending/previews.json
  ./preview_cleanup.py --publish --hub also ask the Hugging Face Hub, for each
                                       case with a release rule, whether the
                                       repo it waits for is there yet

Reads only. The cleanup itself is a `preview-cleanup` record, sent from the
admin page's Cleanup tab or by hand:

  echo '[{"kind": "preview-cleanup", "name": "hy4-preview", "rename": "hy4",
          "fields": {"url": "https://huggingface.co/tencent/Hy4"},
          "scores": {"hle": {"do": "recredit", "url": "https://huggingface.co/tencent/Hy4"}},
          "park": [{"route": "update_openrouter_mapping.py", "subject": "tencent/hy4-preview"}]}]' \\
    | ./answer.py --stdin --no-queue-check -w

  echo '[{"kind": "preview-cleanup", "name": "step-5-preview", "action": "dismiss"}]' \\
    | ./answer.py --stdin --no-queue-check -w

  echo '[{"kind": "preview-watch", "name": "ernie-5-5-flash-preview",
          "watch": {"all": [{"op": "starts_with", "value": "baidu/ERNIE-5.5-Flash"},
                            {"op": "ends_with", "value": "-Base", "not": true}]}}]' \\
    | ./answer.py --stdin --no-queue-check -w

See _preview_cleanup.py for what is detected and why.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import _preview_cleanup as cleanup


HF_API = "https://huggingface.co/api/models"
USER_AGENT = "ai-bench-preview-cleanup/1.0"
HUB_LIMIT = 100


def _hub_ids(params: dict[str, str], timeout: int = 20) -> list[str]:
    url = f"{HF_API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read(2_000_000).decode("utf-8"))
    return [m["id"] for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)
            and len(m["id"]) <= cleanup.MAX_REPO_ID]


def hub_lookup(inputs: cleanup.Inputs) -> dict[str, list[str]]:
    """Each ruled case's owner's newest repos, and those the rule's literal names.

    One or two requests per case with a rule -- there are a handful at most.
    A case whose lookup fails is left out, so its last report carries over.
    """
    found: dict[str, list[str]] = {}
    models = {m.get("name"): m for m in inputs.doc.get("models") or [] if isinstance(m, dict)}
    for name, rule in sorted(inputs.rules.items()):
        model = models.get(name)
        if model is None or name in inputs.done:
            continue
        owner = cleanup.rule_owner(rule, model.get("url"))
        if not owner:
            continue
        base = {"author": owner, "limit": str(HUB_LIMIT)}
        queries = [{**base, "sort": "createdAt", "direction": "-1"}]
        term = cleanup.rule_search_term(rule)
        if term:
            queries.append({**base, "search": term})
        try:
            ids: list[str] = []
            for query in queries:
                ids.extend(i for i in _hub_ids(query) if i not in ids)
        except (OSError, ValueError) as exc:
            print(f"warning: Hugging Face lookup for {name!r} failed: {exc}", file=sys.stderr)
            continue
        found[name] = ids
    return found


def carried_over(inputs: cleanup.Inputs, previous: Path) -> dict[str, list[str]]:
    """What the last report matched, for each case whose rule has not changed since."""
    try:
        report = json.loads(previous.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    kept: dict[str, list[str]] = {}
    for case in report.get("cases") or [] if isinstance(report, dict) else []:
        watch = case.get("watch") if isinstance(case, dict) else None
        rule = inputs.rules.get(case.get("name")) if isinstance(case, dict) else None
        if not (isinstance(watch, dict) and rule and watch.get("hub") != "not checked"):
            continue
        if watch.get("all") != rule["all"] or watch.get("owner") != rule.get("owner"):
            continue
        kept[case["name"]] = [r for r in watch.get("matched") or [] if isinstance(r, str)]
    return kept


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
        if case.get("watch"):
            watch = case["watch"]
            lines.append(f"  waits for {watch['describe']} (Hub: {watch['hub']})")
        for item in case["evidence"]:
            lines.append(f"  + {item['text']}")
        for item in case.get("hints") or []:
            lines.append(f"  ~ {item['text']} (not counted)")
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
    parser.add_argument(
        "--hub", action="store_true",
        help="Ask the Hugging Face Hub whether a case with a release rule has its repo yet "
             "(one or two requests per such case). Without it, or when a lookup fails, the "
             "published report's last answer carries over.",
    )
    parser.add_argument("--llm-json", default=str(cleanup.DEFAULT_LLM_JSON))
    parser.add_argument("--queue", default=str(cleanup.DEFAULT_QUEUE))
    args = parser.parse_args(argv)

    inputs = cleanup.Inputs.from_repo(Path(args.llm_json), Path(args.queue))
    if inputs.rules:
        hub = carried_over(inputs, Path(args.publish or cleanup.DEFAULT_REPORT))
        fresh = hub_lookup(inputs) if args.hub else {}
        inputs.hub_status = {name: "carried over" for name in hub if name not in fresh}
        hub.update(fresh)
        inputs.hub = hub
    report = cleanup.detect(inputs)
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
