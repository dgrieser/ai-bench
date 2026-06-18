#!/usr/bin/env python3
"""Fetch benchmark scores from the llm-stats.com Open LLM Leaderboard.

The leaderboard page is backed by the zeroeval API. A single endpoint returns
every model as a flat record with one column per benchmark (keys ending in
``_score``, values on a 0-1 scale). The source benchmark label is that key with
the ``_score`` suffix stripped (e.g. ``gpqa_score`` -> ``gpqa``).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

URL = "https://api.zeroeval.com/leaderboard/models/full"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) ai-bench-fetcher/1.0"

_SCORE_SUFFIX = "_score"


def fetch_json(url: str, timeout: int = 60) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def get_scores() -> list[dict]:
    """Return a list of dicts with keys: model, name, scores.

    ``model`` is the llm-stats model_id, ``scores`` maps the source benchmark
    label (``_score`` suffix stripped) to its raw 0-1 value. Records without any
    non-null score are dropped.
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    payload = fetch_json(URL)
    if not isinstance(payload, list):
        raise ValueError("Unexpected response: expected a JSON list")

    results: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        model_id = item.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            continue
        scores: dict[str, float] = {}
        for key, value in item.items():
            if not key.endswith(_SCORE_SUFFIX):
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            label = key[: -len(_SCORE_SUFFIX)]
            scores[label] = value
        if not scores:
            continue
        results.append(
            {
                "model": model_id,
                "name": item.get("name"),
                "scores": scores,
            }
        )

    results.sort(key=lambda r: r["model"])
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch benchmark scores from the llm-stats.com Open LLM Leaderboard."
    )
    parser.add_argument(
        "--format",
        choices=["json", "table", "names"],
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--names",
        choices=["models", "benchmarks"],
        default="models",
        help="With --format names, list model ids or benchmark labels (default: models).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = get_scores()

    if args.format == "json":
        print(json.dumps(results, ensure_ascii=False))
    elif args.format == "names":
        if args.names == "models":
            for entry in results:
                print(entry["model"])
        else:
            labels: set[str] = set()
            for entry in results:
                labels.update(entry["scores"].keys())
            for label in sorted(labels):
                print(label)
    else:
        for entry in results:
            title = entry["model"]
            if entry.get("name"):
                title = f"{entry['model']} ({entry['name']})"
            print(f"\n## {title}")
            if not entry["scores"]:
                print("  (no scores)")
                continue
            width = max(len(k) for k in entry["scores"])
            for label, value in sorted(entry["scores"].items()):
                print(f"  {label:<{width}}  {value}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
