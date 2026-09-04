#!/usr/bin/env python3
"""
Fetch DeepSWE (pass@1 %) scores from https://benchlm.ai/benchmarks/deepSwe

The site is a Next.js app that embeds the leaderboard as JSON in a
<script id="__NEXT_DATA__" type="application/json"> tag. This script extracts it
directly without needing a headless browser.

Rows carry a ``sourceType`` ("Open Weight" / "Proprietary", or null while a
model is pending review), reported as ``open_weights``.

benchlm mirrors one revision of Datacurve's leaderboard rather than running the
benchmark, and names the artifact it mirrored in the page's own metadata
("https://deepswe.datacurve.ai/artifacts/v1.1/leaderboard-live.json"). That
path is what every row's ``revision`` is read from, so the scores land in the
DeepSWE column of the revision they actually measure (see _revisions.py). If
the page ever stops naming an artifact, ``revision`` is None and the ingest
skips the rows rather than guessing a column for them -- the rule the README's
version traps state for every unlabelled source.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

from _openness import source_type_open
from _revisions import revision_label


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
# The mirrored artifact's revision directory, in whichever reference link
# carries it: ".../artifacts/v1.1/leaderboard-live.json" -> "v1.1".
_ARTIFACT_REVISION_RE = re.compile(r"/artifacts/([^/\s\"']+)/[^/\s\"']*leaderboard-live\.json")


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


def extract_revision(html: str) -> str | None:
    """The DeepSWE revision benchlm mirrored, as a label ("1.1"), or None.

    Read from the page's own reference links rather than pinned here, so a
    mirror that moves to a later artifact reports the revision it moved to.
    """
    match = _NEXT_DATA_RE.search(html)
    if match is None:
        return None
    page = json.dumps(json.loads(match.group(1)).get("props", {}).get("pageProps", {}))
    directories = _ARTIFACT_REVISION_RE.findall(page)
    if not directories:
        return None
    # One page mirrors one artifact; more than one distinct revision named
    # means the page no longer identifies a single source, so name none.
    unique = {revision_label(directory) for directory in directories}
    return unique.pop() if len(unique) == 1 else None


def get_scores() -> list[dict]:
    """Return a list of dicts with keys: rank, model, slug, creator, score,
    context_window, source_type, open_weights (None while sourceType is unset),
    revision (the mirrored DeepSWE revision, None when the page names none)."""
    print(f"Fetching {URL} ...", file=sys.stderr)
    html = fetch_html(URL)

    print("Parsing leaderboard data...", file=sys.stderr)
    leaderboard = extract_leaderboard(html)
    revision = extract_revision(html)
    if revision is None:
        print("  warning: page names no mirrored artifact; revision unknown",
              file=sys.stderr)
    else:
        print(f"  mirrors DeepSWE revision {revision}", file=sys.stderr)

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
                "revision": revision,
                "slug": item.get("slug"),
                "creator": item.get("creator"),
                "score": score,
                "context_window": item.get("contextWindow"),
                "source_type": item.get("sourceType"),
                "open_weights": source_type_open(item.get("sourceType")),
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
