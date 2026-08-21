#!/usr/bin/env python3
"""
Fetch Berkeley Function-Calling Leaderboard V4 (Overall Accuracy %) scores.

https://gorilla.cs.berkeley.edu/leaderboard.html renders its table client-side
from a CSV the page fetches by dataset name (``./data_overall.csv`` for the
"overall" view the page loads by default), so the CSV is read directly -- no
headless browser, no HTML parsing.

Only the V4 series is published: the CSV carries V4's own columns (Web Search
Acc, Memory Acc, Format Sensitivity) alongside the V1-V3 categories, and
"Overall Acc" is documented on the page as the unweighted average of all
sub-categories. V4 numbers are therefore not comparable with a V1-V3 "BFCL"
score, which is why huggingface-benchmark-name-mapping.json maps only the
v4-labelled aliases onto ``bfcl_v4`` and leaves a bare "BFCL" unmapped.

A model is listed once per calling mode -- "(FC)" for native function calling,
"(Prompt)" for the prompted walk-around, sometimes with a thinking modifier --
and the modes score differently. The reported ``model`` is a normalized base
name with every parenthetical stripped, so one name -> slug mapping covers all
of a model's modes and the usual best-reported-run rule picks between them;
``raw`` and ``mode`` keep the original label.

Rows carry an Organization and a License, the latter reported as
``open_weights`` ("Proprietary" being the only closed value).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request

from _openness import license_open


# Human-facing page; the CSV below is what it hydrates its table from.
LEADERBOARD_URL = "https://gorilla.cs.berkeley.edu/leaderboard.html"
# The page builds this path as `./data_${datasetName}.csv`; "overall" is the
# dataset it loads on DOMContentLoaded and the only one carrying every column.
CSV_URL = "https://gorilla.cs.berkeley.edu/data_overall.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Columns that must be present, or the CSV is not the V4 overall view.
REQUIRED_COLUMNS = ("Model", "Overall Acc")
# V4's own categories. Their absence means the page moved back to a V3 CSV (or
# the dataset was renamed), which would silently mix two score series.
V4_COLUMNS = ("Web Search Acc", "Memory Acc")

_PERCENT_RE = re.compile(r"(-?[0-9]+(?:\.[0-9]+)?)\s*%?")


def fetch_csv(url: str = CSV_URL, retries: int = 3, delay: float = 2.0) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8-sig", errors="replace")
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


def parse_percent(value: str) -> float | None:
    """Read a "77.47%" cell. Returns None for "N/A", "-" and empty cells."""
    if not isinstance(value, str):
        return None
    match = _PERCENT_RE.fullmatch(value.strip())
    if not match:
        return None
    return float(match.group(1))


def split_mode(raw: str) -> tuple[str, str | None]:
    """Split a leaderboard label into (label_without_mode, mode).

    "Claude-Opus-4-5-20251101 (FC)"      -> ("Claude-Opus-4-5-20251101", "FC")
    "GLM-4.6 (FC thinking)"              -> ("GLM-4.6", "FC thinking")
    "BitAgent-Bounty-8B"                 -> ("BitAgent-Bounty-8B", None)
    """
    match = re.search(r"\(([^)]*)\)", raw)
    mode = match.group(1).strip() if match else None
    without = re.sub(r"\s*\([^)]*\)\s*", " ", raw)
    return re.sub(r"\s+", " ", without).strip(), mode


def normalize_model(raw: str) -> str:
    """Normalize a leaderboard label to a base name for mapping.

    Every parenthetical is a calling mode or a thinking modifier, never part of
    the model, so all of them are stripped, separators unified and the result
    lowercased: "Qwen3-235B-A22B-Instruct-2507 (Prompt)" ->
    "qwen3 235b a22b instruct 2507".
    """
    without, _ = split_mode(raw)
    without = without.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", without).strip().lower()


def parse_rows(csv_text: str) -> list[dict]:
    """Rows of the overall CSV as dicts, header-keyed.

    Raises when the header is not the V4 overall one rather than reading a
    different score series into the same column.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    fieldnames = [name.strip() for name in (reader.fieldnames or [])]
    missing = [name for name in (*REQUIRED_COLUMNS, *V4_COLUMNS) if name not in fieldnames]
    if missing:
        raise ValueError(
            f"{CSV_URL} is missing the column(s) {', '.join(missing)} — the leaderboard "
            "layout changed; refusing to guess which BFCL series this table belongs to."
        )
    return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]


def get_scores() -> list[dict]:
    """Return one dict per leaderboard row.

    Keys: model (normalized base), raw (the leaderboard label), mode, score
    (Overall Acc %), organization, license, open_weights, rank (rank within the
    leaderboard, 1 = best).
    """
    print(f"Fetching {CSV_URL} ...", file=sys.stderr)
    rows = parse_rows(fetch_csv())
    print(f"  parsed {len(rows)} leaderboard rows", file=sys.stderr)

    kept: list[dict] = []
    for row in rows:
        raw = row.get("Model", "")
        if not raw:
            continue
        score = parse_percent(row.get("Overall Acc", ""))
        if score is None:
            continue
        _, mode = split_mode(raw)
        kept.append(
            {
                "model": normalize_model(raw),
                "raw": raw,
                "mode": mode,
                "score": round(score, 2),
                "organization": row.get("Organization") or None,
                "license": row.get("License") or None,
                "open_weights": license_open(row.get("License")),
            }
        )

    kept.sort(key=lambda r: -r["score"])
    for i, entry in enumerate(kept, 1):
        entry["rank"] = i
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Berkeley Function-Calling Leaderboard V4 scores."
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
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
        for name in sorted({entry["model"] for entry in scores}):
            print(name)
    else:
        width = max([len("MODEL")] + [len(e["raw"]) for e in scores])
        fmt = f"{{:<{width}}}  {{:>7}}  {{:<12}}  {{:<20}}"
        print(fmt.format("MODEL", "OVERALL", "WEIGHTS", "ORGANIZATION"))
        for entry in scores:
            weights = {True: "open", False: "proprietary", None: "unknown"}[
                entry["open_weights"]
            ]
            print(
                fmt.format(
                    entry["raw"],
                    str(entry["score"]),
                    weights,
                    entry["organization"] or "",
                )
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch the BFCL leaderboard: {exc}", file=sys.stderr)
        raise SystemExit(1)
