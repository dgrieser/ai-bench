#!/usr/bin/env python3
"""
Fetch FrontierCode scores from Cognition's leaderboard, https://cognition.com/frontiercode

The leaderboard renders client-side from a static JSON the page loads separately
(the HTML only ships a "Loading leaderboard…" placeholder), so this reads that
JSON directly instead of parsing markup:

    {"v1_1": {"models": [...], "harness": {...}, "efforts": {...},
              "subsets": {"main": 100, "extended": 150},
              "data": {"<model>": {"<effort>": {"<subset>": {...}}}}},
     "v1": {...}}

Each leaf carries both metrics the site shows: `new_score` (the Score column,
which is what llm.json stores in its FrontierCode columns) and `correct`
(the raw pass rate, a few points higher), plus tokens and cost.

A model may be published at several reasoning efforts (Claude/GPT ship
low..max); the best-scoring effort is reported, matching how the other
harness-variant sources in this repo resolve one row per model. `effort` and
`efforts` name what was picked and what was on offer.

Every revision in the payload is reported, newest first, and every row names
the `revision` it came from. Revisions are *not* merged: a 1.0 number was
measured against that revision's task set and scoring, so it is a different
measurement from a 1.1 number, and llm.json gives each its own column
(`frontiercode_1_0`, `frontiercode_1_1`) rather than blending them -- see
_revisions.py. A model re-run in 1.1 therefore appears twice, once per
revision, and models retired before the re-run (Kimi K2.5, MiniMax M2.5 and
friends) appear under 1.0 alone.

Revisions are ordered by the numbers in their key, so a future "v1_2" or "v2"
sorts ahead of today's "v1_1" without an edit here. --revision pins a single
one.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

from _revisions import revision_label, revision_rank


URL = "https://cognition.com/data/frontiercode-leaderboard/data.json"
# The human-facing leaderboard, stored as the score's source page.
LEADERBOARD_URL = "https://cognition.com/frontiercode"

# --revision value that reports every revision instead of pinning one.
ALL_REVISIONS = "all"
DEFAULT_REVISION = ALL_REVISIONS

# Task subset: "main" is the 100-task set the leaderboard opens on, "extended"
# the 150-task superset.
DEFAULT_SUBSET = "main"

# Metric read off each leaf. "new_score" is the site's Score column; "correct"
# is the raw pass rate, offered for comparison but not what llm.json stores.
METRICS = ("new_score", "correct")
DEFAULT_METRIC = METRICS[0]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected payload from {url}: expected an object")
    return payload


def revisions_newest_first(payload: dict) -> list[tuple[str, dict]]:
    """(key, block) for every revision in the payload, newest first."""
    blocks = [(key, block) for key, block in payload.items() if isinstance(block, dict)]
    blocks.sort(key=lambda item: revision_rank(item[0]), reverse=True)
    return blocks


def revision_block(payload: dict, revision: str) -> tuple[str, dict]:
    """One revision's (key, block), addressed by label ("1.1") or key ("v1_1")."""
    for key, block in revisions_newest_first(payload):
        if revision in (key, revision_label(key)):
            return key, block
    available = ", ".join(
        f"{revision_label(key)} ({key})" for key, _ in revisions_newest_first(payload)
    )
    raise ValueError(
        f"Revision {revision!r} not in payload; available: {available or '(none)'}"
    )


def best_effort(
    efforts: dict, subset: str, metric: str
) -> tuple[str | None, dict | None]:
    """The (effort, leaf) with the highest metric for one subset.

    A model published at one effort only ("none") goes through this unchanged.
    Efforts whose leaf has no numeric metric are ignored, so a partially
    published run cannot win by being empty.
    """
    best_name: str | None = None
    best_leaf: dict | None = None
    best_value: float | None = None
    for name, subsets in efforts.items():
        if not isinstance(subsets, dict):
            continue
        leaf = subsets.get(subset)
        if not isinstance(leaf, dict):
            continue
        value = leaf.get(metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if best_value is None or value > best_value:
            best_name, best_leaf, best_value = name, leaf, float(value)
    return best_name, best_leaf


def revision_rows(
    key: str, block: dict, subset: str, metric: str
) -> list[dict]:
    """One row per model published in this revision for the given subset."""
    subsets = block.get("subsets")
    if isinstance(subsets, dict) and subset not in subsets:
        available = ", ".join(sorted(subsets)) or "(none)"
        raise ValueError(
            f"Subset {subset!r} not in revision {revision_label(key)}; available: {available}"
        )

    data = block.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"Revision {revision_label(key)} has no 'data' object")
    harness = block.get("harness") if isinstance(block.get("harness"), dict) else {}

    rows: list[dict] = []
    for model, efforts in data.items():
        if not isinstance(model, str) or not isinstance(efforts, dict):
            continue
        effort, leaf = best_effort(efforts, subset, metric)
        if leaf is None:
            continue
        rows.append(
            {
                "model": model,
                "revision": revision_label(key),
                "harness": harness.get(model),
                "effort": effort,
                "efforts": sorted(efforts),
                "score": round(float(leaf[metric]) * 100, 2),
                "correct": leaf.get("correct"),
                "new_score": leaf.get("new_score"),
                "tokens": leaf.get("tokens"),
                "cost": leaf.get("cost"),
            }
        )
    return rows


def get_scores(
    revision: str = DEFAULT_REVISION,
    subset: str = DEFAULT_SUBSET,
    metric: str = DEFAULT_METRIC,
) -> list[dict]:
    """Return one dict per (model, revision): model, revision, harness, score, ...

    score = metric * 100, i.e. the percentage the leaderboard prints; the payload
    stores fractions.

    revision="all" (the default) reports every revision, newest first, without
    merging them: each row carries the revision it was measured under, and a
    model re-run in a later revision appears once per revision. Any other value
    pins that single revision. Rows are ranked within their own revision, since
    ranking across revisions would compare two different task sets.
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    payload = fetch_json(URL)

    if revision == ALL_REVISIONS:
        blocks = revisions_newest_first(payload)
        if not blocks:
            raise ValueError("Payload carries no revision blocks")
    else:
        blocks = [revision_block(payload, revision)]

    results: list[dict] = []
    for key, block in blocks:
        rows = sorted(revision_rows(key, block, subset, metric), key=lambda r: -r["score"])
        for i, row in enumerate(rows, 1):
            row["rank"] = i
        print(
            f"  revision {revision_label(key)} ({key}): {len(rows)} model(s)",
            file=sys.stderr,
        )
        results.extend(rows)

    print(
        f"  parsed {len(results)} rows across {len(blocks)} revision(s) ({subset}, {metric})",
        file=sys.stderr,
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch FrontierCode leaderboard scores.")
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"Pin one benchmark revision by label (1.1) or payload key (v1_1), "
        f"or {ALL_REVISIONS!r} to report every revision (default: {DEFAULT_REVISION}).",
    )
    parser.add_argument(
        "--subset",
        default=DEFAULT_SUBSET,
        help=f"Task subset, e.g. main or extended (default: {DEFAULT_SUBSET}).",
    )
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default=DEFAULT_METRIC,
        help=f"Metric to report as score (default: {DEFAULT_METRIC}, the site's Score column).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = get_scores(revision=args.revision, subset=args.subset, metric=args.metric)

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for entry in scores:
            print(entry["model"])
    else:
        model_width = max([len("MODEL"), *(len(e["model"]) for e in scores)])
        harness_width = max([len("HARNESS"), *(len(e.get("harness") or "") for e in scores)])
        fmt = f"{{:<{model_width}}}  {{:>6}}  {{:<4}}  {{:<{harness_width}}}  {{:<8}}"
        print(fmt.format("MODEL", "SCORE", "REV", "HARNESS", "EFFORT"))
        for entry in scores:
            print(
                fmt.format(
                    entry["model"],
                    f"{entry['score']:.2f}",
                    entry.get("revision") or "",
                    entry.get("harness") or "",
                    entry.get("effort") or "",
                )
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
