#!/usr/bin/env python3
"""
Fetch FrontierSWE Dominance (%) scores from https://www.frontierswe.com/

The site is a Next.js app that embeds the leaderboard as React Server Component
flight data in self.__next_f.push() script tags. This script concatenates the
decoded flight chunks and extracts the "entries" object without a headless browser.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request


URL = "https://www.frontierswe.com/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,(".*?")\]\)', re.DOTALL)


def fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def extract_entries(html: str) -> dict:
    """Extract the leaderboard 'entries' object from the Next.js RSC flight data."""
    decoded = ""
    for chunk in _PUSH_RE.findall(html):
        try:
            decoded += json.loads(chunk)
        except json.JSONDecodeError:
            continue

    key_idx = decoded.find('"entries":')
    if key_idx == -1:
        raise ValueError("Could not find 'entries' in page flight data")

    obj_start = decoded.index("{", key_idx)
    depth, end = 0, obj_start
    for i, c in enumerate(decoded[obj_start:], obj_start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    return json.loads(decoded[obj_start : end + 1])


def get_scores(group: str = "mean") -> list[dict]:
    """Return a list of dicts: rank, model, harness, dominance, overall, score.

    score = dominance * 100 (win-rate %). group selects the leaderboard view;
    'mean' matches the site's public leaderboard, 'best' uses each model's best harness.
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    html = fetch_html(URL)

    print("Parsing leaderboard data...", file=sys.stderr)
    entries = extract_entries(html)

    items = entries.get(group)
    if not isinstance(items, list):
        raise ValueError(f"Group {group!r} not found; available: {sorted(entries)}")

    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model = item.get("model")
        dominance = item.get("dominance")
        if not isinstance(model, str) or not model or not isinstance(dominance, (int, float)):
            continue
        results.append(
            {
                "model": model,
                "harness": item.get("harness"),
                "dominance": round(dominance, 4),
                "overall": item.get("overall"),
                "score": round(dominance * 100, 1),
            }
        )

    results.sort(key=lambda r: -r["score"])
    for i, r in enumerate(results, 1):
        r["rank"] = i
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch FrontierSWE leaderboard scores.")
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--group",
        choices=["mean", "best"],
        default="mean",
        help="Leaderboard view: 'mean' (public default) or 'best' harness (default: mean).",
    )
    args = parser.parse_args()

    scores = get_scores(group=args.group)

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for entry in scores:
            print(entry["model"])
    else:
        col_widths = [
            max(len("MODEL"), max((len(e["model"]) for e in scores), default=0)),
            7,
            max(len("HARNESS"), max((len(e.get("harness") or "") for e in scores), default=0)),
        ]
        fmt = f"{{:<{col_widths[0]}}}  {{:>{col_widths[1]}}}  {{:<{col_widths[2]}}}"
        print(fmt.format("MODEL", "SCORE", "HARNESS"))
        for entry in scores:
            print(fmt.format(entry["model"], str(entry["score"]), entry.get("harness") or ""))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
