#!/usr/bin/env python3
"""
Fetch ARC-AGI scores from ARC Prize's own leaderboard, https://arcprize.org/leaderboard

The leaderboard draws from two static files the page loads:
``/media/data/evaluations.json`` (one row per model configuration and dataset:
``datasetId``, ``modelId``, ``score`` as a fraction, ``display``) and
``/media/data/models.json`` (``id`` -> ``displayName``). Each ARC-AGI version
is scored on its own semi-private set, which is what the board ranks by and what
ARC Prize verifies itself, so each feeds its own column:

  * ``v1_Semi_Private`` -> ``arc_agi_1``
  * ``v2_Semi_Private`` -> ``arc_agi_2``
  * ``v3_Semi_Private`` -> ``arc_agi_3`` -- the interactive successor, where
    the agent explores novel environments rather than completing grids.

The public-eval and private-eval rows are other question sets and are not
read. Two kinds of row are dropped from the semi-private boards too: the human
panel, which is not a model, and every "Provider Adapter" run -- a model run
through its provider's own agent adapter rather than ARC Prize's harness,
which on ARC-AGI-3 scores in another range altogether (GPT-6 Astra 98.6 there
against 62.7 in the standard harness). A dataset the file carries beyond these
is reported, not read.

Labels carry the reasoning effort in a parenthetical -- "GPT-6 Astra (Max)" --
so the reported ``model`` is the display name with run settings stripped,
lower-cased (the rule fetch_zerobench.py follows), and update.py keeps a
model's best configuration per column.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import _fetch_warnings
from _fetch_checks import check_fractions, require_rows
import _source_pages

SITE_URL = "https://arcprize.org"
LEADERBOARD_URL = f"{SITE_URL}/leaderboard"
EVALUATIONS_URL = f"{SITE_URL}/media/data/evaluations.json"
MODELS_URL = f"{SITE_URL}/media/data/models.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Dataset id -> llm.json column. Only the semi-private sets, one per version.
DATASETS: dict[str, str] = {
    "v1_Semi_Private": "arc_agi_1",
    "v2_Semi_Private": "arc_agi_2",
    "v3_Semi_Private": "arc_agi_3",
}
KEYS = tuple(DATASETS.values())

# Rows that are not a model run in ARC Prize's own harness.
_HUMAN_RE = re.compile(r"human", re.IGNORECASE)
_ADAPTER_RE = re.compile(r"provider[- ]adapter", re.IGNORECASE)

_MODIFIER_TOKENS = frozenset(
    {
        "none", "minimal", "low", "medium", "high", "xhigh", "x-high", "max",
        "default", "effort", "reasoning", "thinking", "non-thinking",
    }
)


def board_url(key: str) -> str:
    """The page a score in column ``key`` is credited to: the one leaderboard."""
    if key not in KEYS:
        raise KeyError(key)
    return LEADERBOARD_URL


def fetch_json(url: str, retries: int = 3, delay: float = 2.0) -> Any:
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as exc:
            if attempt == retries:
                raise
            wait = delay * attempt
            print(
                f"  attempt {attempt}/{retries} failed ({exc}); retrying in {wait:.0f}s ...",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def normalize_model(raw: str) -> str:
    """The mapping key for a label: a run-setting parenthetical dropped.

    "GPT-6 Astra (Max)" -> "gpt-6 astra"; "DeepSeek V4.1 Flash (None)" ->
    "deepseek v4.1 flash"; a parenthetical naming anything else is kept.
    """
    match = re.search(r"\(([^)]*)\)\s*$", raw)
    name = raw
    if match:
        tokens = [t for t in re.split(r"[\s,/]+", match.group(1).strip().lower()) if t]
        if tokens and all(t in _MODIFIER_TOKENS for t in tokens):
            name = raw[: match.start()]
    return re.sub(r"\s+", " ", name).strip().lower()


def parse_rows(evaluations: Any, models: Any) -> list[dict]:
    """One row per displayed semi-private run in ARC Prize's own harness."""
    if not isinstance(evaluations, list) or not isinstance(models, list):
        raise ValueError(f"{EVALUATIONS_URL} or {MODELS_URL} is not a list -- the layout changed.")
    names = {
        m.get("id"): m.get("displayName")
        for m in models
        if isinstance(m, dict) and isinstance(m.get("id"), str)
    }
    rows: list[dict] = []
    unread: set[str] = set()
    for entry in evaluations:
        if not isinstance(entry, dict) or entry.get("display") is False:
            continue
        dataset = entry.get("datasetId")
        key = DATASETS.get(dataset)
        if key is None:
            if isinstance(dataset, str) and dataset.endswith("_Semi_Private"):
                unread.add(dataset)
            continue
        model_id = entry.get("modelId")
        raw = names.get(model_id) or model_id
        score = entry.get("score")
        if not isinstance(raw, str) or not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        if _HUMAN_RE.search(raw) or _HUMAN_RE.search(str(model_id)):
            continue
        if _ADAPTER_RE.search(raw) or _ADAPTER_RE.search(str(model_id)):
            continue
        rows.append(
            {
                "key": key,
                "dataset": dataset,
                "model": normalize_model(raw),
                "raw": raw,
                "model_id": model_id,
                "fraction": float(score),
                "score": round(float(score) * 100, 2),
                "source": LEADERBOARD_URL,
            }
        )
    for dataset in sorted(unread):
        message = f"ARC Prize dataset {dataset} has no llm.json column; not read"
        print(f"  warning: {message}", file=sys.stderr)
        _fetch_warnings.record("fetch_arc_prize.py", message, source="new-version:arc_prize")
    return rows


def get_scores(fetch=fetch_json) -> list[dict]:
    """Rows for every tracked ARC-AGI version: key, dataset, model, raw,
    model_id, score (percent), source."""
    print(f"Fetching {EVALUATIONS_URL} and {MODELS_URL} ...", file=sys.stderr)
    rows = parse_rows(fetch(EVALUATIONS_URL), fetch(MODELS_URL))
    for key in KEYS:
        board = [r for r in rows if r["key"] == key]
        require_rows(board, f"{EVALUATIONS_URL}#{key}")
        check_fractions([r["fraction"] for r in board], f"{EVALUATIONS_URL}#{key}", "score")
    rows.sort(key=lambda r: (r["key"], -r["score"]))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch ARC-AGI scores from ARC Prize's leaderboard.")
    parser.add_argument(
        "--format",
        choices=["table", "json", "names", "pages"],
        default="table",
        help="Output format (default: table).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = get_scores()
    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for name in sorted({r["model"] for r in scores}):
            print(name)
    elif args.format == "pages":
        _source_pages.emit(sorted(scores, key=lambda r: r["model"]), lambda r: r["source"])
    else:
        width = max([len("MODEL"), *(len(r["raw"]) for r in scores)])
        print(f"{'COLUMN':<10}  {'MODEL':<{width}}  {'SCORE':>6}")
        for r in scores:
            print(f"{r['key']:<10}  {r['raw']:<{width}}  {r['score']:>6}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch ARC Prize's leaderboard data: {exc}", file=sys.stderr)
        raise SystemExit(1)
