#!/usr/bin/env python3
"""
Fetch DeepSWE (pass@1 %) scores from https://benchlm.ai/benchmarks/deepSwe

The site is a Next.js app that embeds the leaderboard as JSON in a
<script id="__NEXT_DATA__" type="application/json"> tag. This script extracts it
directly without needing a headless browser.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request


URL = "https://benchlm.ai/benchmarks/deepSwe"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)


def fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def extract_leaderboard(html: str) -> list[dict]:
    """Extract the leaderboard array from the Next.js __NEXT_DATA__ blob."""
    match = _NEXT_DATA_RE.search(html)
    if match is None:
        raise ValueError("Could not find __NEXT_DATA__ script in page HTML")

    data = json.loads(match.group(1))
    leaderboard = data.get("props", {}).get("pageProps", {}).get("leaderboard")
    if not isinstance(leaderboard, list):
        raise ValueError("Could not find 'leaderboard' array in __NEXT_DATA__")
    return leaderboard


def get_scores() -> list[dict]:
    """Return a list of dicts with keys: rank, model, slug, creator, score, context_window."""
    print(f"Fetching {URL} ...", file=sys.stderr)
    html = fetch_html(URL)

    print("Parsing leaderboard data...", file=sys.stderr)
    leaderboard = extract_leaderboard(html)

    results = []
    for item in leaderboard:
        if not isinstance(item, dict):
            continue
        model = item.get("model")
        score = item.get("score")
        if not isinstance(model, str) or not model or not isinstance(score, (int, float)):
            continue
        results.append(
            {
                "model": model,
                "slug": item.get("slug"),
                "creator": item.get("creator"),
                "score": score,
                "context_window": item.get("contextWindow"),
            }
        )

    results.sort(key=lambda r: -r["score"])
    for i, r in enumerate(results, 1):
        r["rank"] = i
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch DeepSWE leaderboard scores.")
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table).",
    )
    args = parser.parse_args()

    scores = get_scores()

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for entry in scores:
            print(entry["model"])
    else:
        col_widths = [
            max(len("MODEL"), max((len(e["model"]) for e in scores), default=0)),
            7,
            max(len("CREATOR"), max((len(e.get("creator") or "") for e in scores), default=0)),
        ]
        fmt = f"{{:<{col_widths[0]}}}  {{:>{col_widths[1]}}}  {{:<{col_widths[2]}}}"
        print(fmt.format("MODEL", "SCORE", "CREATOR"))
        for entry in scores:
            print(fmt.format(entry["model"], str(entry["score"]), entry.get("creator") or ""))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
