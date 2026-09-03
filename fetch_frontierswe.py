#!/usr/bin/env python3
"""
Fetch FrontierSWE scores from https://www.frontierswe.com/

The site is a Next.js app that embeds the leaderboard as React Server Component
flight data in self.__next_f.push() script tags. This script concatenates the
decoded flight chunks and extracts the "entries" object without a headless browser.

The root page serves the V2 leaderboard: 34 hand-crafted tasks, five trials each
under a 20-hour budget, scored as a percentage. V1 -- scored by pairwise
Dominance, a win rate on an entirely different scale -- still lives at /v1 and is
deliberately not what this reads.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request


URL = "https://www.frontierswe.com/"
# entries are keyed by score mode first: "abs" is the site's own percentage scale.
SCORE_MODE = "abs"
# ... and by trial aggregation second. The public leaderboard prints mean@5 and
# spans its whiskers from worst@5 to best@5.
GROUPS = ("mean", "best", "worst")
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


def select_group(entries: dict, group: str) -> list:
    """One leaderboard view out of the flight payload.

    The two levels are read separately so a failure names the one that moved:
    the site nested the views under a score mode when it shipped V2, and the
    error a reader gets should say whether the mode or the view went missing.
    """
    board = entries.get(SCORE_MODE)
    if not isinstance(board, dict):
        raise ValueError(f"Score mode {SCORE_MODE!r} not found; available: {sorted(entries)}")

    items = board.get(group)
    if not isinstance(items, list):
        raise ValueError(f"Group {group!r} not found; available: {sorted(board)}")
    return items


def get_scores(group: str = "mean") -> list[dict]:
    """Return a list of dicts: rank, model, harness, overall, score.

    score is the site's own percentage, rounded the way the leaderboard prints
    it. group selects the trial aggregation: 'mean' is the public leaderboard's
    mean@5, 'best' and 'worst' the ends of its whiskers.
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    html = fetch_html(URL)

    print("Parsing leaderboard data...", file=sys.stderr)
    entries = extract_entries(html)
    items = select_group(entries, group)

    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model = item.get("model")
        overall = item.get("overall")
        if not isinstance(model, str) or not model or not isinstance(overall, (int, float)):
            continue
        results.append(
            {
                "model": model,
                "harness": item.get("harness"),
                "overall": overall,
                "score": round(overall, 1),
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
        choices=list(GROUPS),
        default="mean",
        help="Trial aggregation: 'mean' (the public leaderboard's mean@5), "
        "'best' or 'worst' (default: mean).",
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
