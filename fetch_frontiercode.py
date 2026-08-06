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
which is what llm.json stores in the "frontiercode" benchmark) and `correct`
(the raw pass rate, a few points higher), plus tokens and cost.

A model may be published at several reasoning efforts (Claude/GPT ship
low..max); the best-scoring effort is reported, matching how the other
harness-variant sources in this repo resolve one row per model. `effort` and
`efforts` name what was picked and what was on offer.

Every revision in the payload is read, newest first, and a model is taken from
the newest revision that publishes it: the current revision covers only the
models Cognition re-ran, while older ones are the sole source for models retired
before the re-run (Kimi K2.5, MiniMax M2.5 and friends live in v1 only).
Revisions are ordered by the numbers in their key, so a future "v1_2" or "v2"
outranks today's "v1_1" without an edit here. Each row names the `revision` it
came from, and --revision pins a single one -- worth doing when the mix matters,
since a 1.0 number was measured against that revision's task set and scoring,
not the current one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request


URL = "https://cognition.com/data/frontiercode-leaderboard/data.json"
# The human-facing leaderboard, stored as the score's source page.
LEADERBOARD_URL = "https://cognition.com/frontiercode"

# --revision value that merges every revision instead of pinning one.
ALL_REVISIONS = "all"
DEFAULT_REVISION = ALL_REVISIONS

_REVISION_NUM_RE = re.compile(r"\d+")

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


def revision_rank(key: str) -> tuple[int, ...]:
    """Sort key ordering revisions oldest to newest by the numbers in their name.

    "v1" -> (1,), "v1_1" -> (1, 1), "v2" -> (2,) -- and a shorter tuple sorts
    before its own extensions, so v1 < v1_1 < v1_2 < v2. A key carrying no digits
    cannot be placed among them and is treated as the oldest, so an unexpected
    name never silently outranks a real revision.
    """
    numbers = tuple(int(n) for n in _REVISION_NUM_RE.findall(key))
    return numbers or (-1,)


def revision_label(key: str) -> str:
    """The label the site prints for a payload key: "v1_1" -> "1.1", "v1" -> "1.0"."""
    numbers = _REVISION_NUM_RE.findall(key)
    if not numbers:
        return key
    major, *rest = numbers
    return ".".join([major, *(rest or ["0"])])


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
    """Return one dict per model: model, revision, harness, effort, score, ...

    score = metric * 100, i.e. the percentage the leaderboard prints; the payload
    stores fractions.

    revision="all" (the default) walks the revisions newest first and keeps the
    first row it finds per model, so a re-run supersedes the older number and a
    model dropped from the current revision still reports its last published one.
    Any other value pins that single revision.
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    payload = fetch_json(URL)

    if revision == ALL_REVISIONS:
        blocks = revisions_newest_first(payload)
        if not blocks:
            raise ValueError("Payload carries no revision blocks")
    else:
        blocks = [revision_block(payload, revision)]

    by_model: dict[str, dict] = {}
    for key, block in blocks:
        superseded = 0
        added = 0
        for row in revision_rows(key, block, subset, metric):
            if row["model"] in by_model:
                superseded += 1
                continue
            by_model[row["model"]] = row
            added += 1
        note = f", {superseded} superseded by a newer revision" if superseded else ""
        print(
            f"  revision {revision_label(key)} ({key}): {added} model(s){note}",
            file=sys.stderr,
        )

    results = sorted(by_model.values(), key=lambda r: -r["score"])
    for i, r in enumerate(results, 1):
        r["rank"] = i
    print(
        f"  parsed {len(results)} models ({subset}, {metric})",
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
        f"or {ALL_REVISIONS!r} to merge them newest first (default: {DEFAULT_REVISION}).",
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
