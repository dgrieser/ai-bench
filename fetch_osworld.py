#!/usr/bin/env python3
"""Fetch OSWorld-Verified and OSWorld 2.0 scores from their official boards.

XLANG Lab publishes the two generations of OSWorld on two sites, and this
script reads both. Every row it reports names the llm.json column it feeds
(``benchmark``), so update.py never has to guess which board a label came from.

**OSWorld-Verified** (``osworld_verified``) is read from the results workbook
the original site links (os-world.github.io, which now redirects to
osworld-v1.xlang.ai since 2.0 got a site of its own). Only the Foundation E2E
GUI setup is read -- no extra accessibility tree, no coding-based actions, no
multiple rollouts, 100 steps -- and a model's runs under it are averaged.

**OSWorld 2.0** is 108 long-horizon workflows on a separate site,
osworld-v2.xlang.ai, whose leaderboard hydrates from one JSON file. That file
carries every run the board can show, and the board's own filters say which of
them are one measurement:

  * *Release.* The task set has been re-released twice -- 2026.06.24 (the
    paper's), 2026.08.08, and the bug-fix 2.1 the maintainers now recommend --
    each with its own task files, assets and mocked websites, and the scores do
    not carry over: Claude Opus 5 at max effort reads 31.4 on 2026.08.08 and
    44.3 on 2.1. So each release llm.json tracks has its own column, the same
    rule _revisions.py applies to every other re-released board, and a release
    in RELEASES below is the only way a row reaches one. 2026.08.08 has no
    column: 2.1 superseded it five weeks later, and the one model it scores
    that no tracked release does is GPT-5.6 Sol, a closed one. A release this
    script does not know is skipped and reported on stderr, never folded into
    a neighbour.
  * *Step budget.* Rows are published at 150, 300 and 500 steps; the paper's
    primary metric is at 500, and that is the only budget read.
  * *Dataset scope.* The "offline set" is the 82 tasks runnable without
    internet access, a subset scored separately; only the full set is read.
  * *Metric.* Binary accuracy -- the share of tasks fully completed -- is the
    board's default sort and the paper's primary metric. The partial score (the
    share of scoring checkpoints passed) rides along on every row but is never
    the reported score.

What is left per model is one row per (reasoning effort, tool setting) run, and
update.py keeps the best of them per column, which is the ranking the board
itself shows.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from _fetch_checks import check_percentages, require_columns, require_rows

VERIFIED_KEY = "osworld_verified"

# The workbook's own host. os-world.github.io still serves it, but only as a
# 301 to plain-http osworld-v1.xlang.ai, so the download goes straight to the
# https URL instead.
OSWORLD_XLSX_URL = "https://osworld-v1.xlang.ai/static/data/osworld_verified_results.xlsx"
# Human-facing site the workbook belongs to; stored as the per-score source
# URL because the .xlsx identifies a download, not a page a reader can open.
# Kept at the address every stored osworld_verified score already cites, which
# redirects to the v1 site.
OSWORLD_SITE_URL = "https://os-world.github.io"
SHEET_NAME = "Eval Results"

# Foundation E2E GUI: no extra a11y tree, no extra coding actions, no multiple rollout, 100 steps
FOUNDATION_MAX_STEPS = 100
FOUNDATION_A11Y = "No"
FOUNDATION_CODING = "No"
FOUNDATION_ROLLOUT = "No"

# Every column parse_rows() reads. col() answers None for a missing one, so a
# renamed header used to drop every row -- a missing "Max steps" fails the
# Foundation filter on all of them -- and the fetcher reported an empty board.
REQUIRED_COLUMNS = (
    "Model",
    "Institution",
    "Approach type",
    "Max steps",
    "Additional a11y tree used",
    "Additional coding-based action",
    "Multiple rollout",
    "Date",
    "Success rate",
)


# OSWorld 2.0's own site, which is what a score is credited to, and the file its
# leaderboard renders from.
OSWORLD_V2_SITE_URL = "https://osworld-v2.xlang.ai"
OSWORLD_V2_JSON_URL = f"{OSWORLD_V2_SITE_URL}/static/data/leaderboard/official-results.json"
# What the payload has to call itself before any row of it is read: a file that
# starts describing another benchmark is not OSWorld 2.0 any more.
V2_BENCHMARK_VERSION = "OSWorld 2.0"
# The full task set every release tracked here scores. A board that starts
# dividing by something else has changed what binary accuracy means.
V2_DATASET_SIZE = 108
V2_STEP_BUDGET = 500
V2_DATASET_SCOPE = "full"
V2_METRIC = "binaryAccuracy"

# Release label, as the payload spells it -> the llm.json column it feeds.
# 2026.08.08 is deliberately absent (see the module docstring).
RELEASES: dict[str, str] = {
    "v2026.06.24": "osworld_2_0",
    "v2.1": "osworld_2_1",
}

# The fields a 2.0 row has to carry to be read at all. Without one of them the
# row cannot be placed on the board's own filters, so it is a layout change,
# not a row to skip.
V2_REQUIRED_FIELDS = ("model", "stepBudget", V2_METRIC)

# Every column this script can report a row for.
KEYS = (VERIFIED_KEY, *RELEASES.values())


def source_url(key: str) -> str:
    """The page a score in column ``key`` is credited to."""
    return OSWORLD_SITE_URL if key == VERIFIED_KEY else OSWORLD_V2_SITE_URL


def _excel_serial_to_iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, (int, float)):
        try:
            # Excel epoch: 1899-12-30 (accounts for the Lotus 1-2-3 leap-year bug)
            d = (datetime(1899, 12, 30) + timedelta(days=int(value))).date()
            return d.isoformat()
        except (ValueError, OverflowError):
            return None
    return None


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()



def parse_rows(data: bytes) -> list[dict[str, Any]]:
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError("openpyxl is required: pip install openpyxl") from exc

    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
    if SHEET_NAME not in wb.sheetnames:
        raise RuntimeError(f"Sheet {SHEET_NAME!r} not found; available: {wb.sheetnames}")

    ws = wb[SHEET_NAME]
    raw_rows = list(ws.iter_rows(min_row=1, values_only=True))
    require_rows(raw_rows, OSWORLD_XLSX_URL, f"rows in sheet {SHEET_NAME!r}")

    headers = [str(h).strip() if h is not None else None for h in raw_rows[0]]
    require_columns(headers, REQUIRED_COLUMNS, f"{OSWORLD_XLSX_URL} sheet {SHEET_NAME!r}")

    def col(row: tuple[Any, ...], name: str) -> Any:
        try:
            return row[headers.index(name)]
        except ValueError:
            return None

    result = []
    for row in raw_rows[1:]:
        model = col(row, "Model")
        if not isinstance(model, str) or not model.strip():
            continue
        result.append(
            {
                "model": model.strip(),
                "institution": col(row, "Institution"),
                "approach_type": col(row, "Approach type"),
                "max_steps": col(row, "Max steps"),
                "a11y": col(row, "Additional a11y tree used"),
                "coding": col(row, "Additional coding-based action"),
                "rollout": col(row, "Multiple rollout"),
                "date": col(row, "Date"),
                "score": col(row, "Success rate"),
            }
        )
    return result


def is_foundation_e2e(row: dict[str, Any]) -> bool:
    return (
        row.get("max_steps") == FOUNDATION_MAX_STEPS
        and row.get("a11y") == FOUNDATION_A11Y
        and row.get("coding") == FOUNDATION_CODING
        and row.get("rollout") == FOUNDATION_ROLLOUT
    )


def aggregate(rows: list[dict[str, Any]], foundation_only: bool) -> list[dict[str, Any]]:
    filtered = [r for r in rows if not foundation_only or is_foundation_e2e(r)]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        if isinstance(row["score"], (int, float)):
            groups[row["model"]].append(row)

    aggregated = []
    for model, model_rows in groups.items():
        scores = [r["score"] for r in model_rows]
        avg = round(sum(scores) / len(scores), 1)
        latest = max(model_rows, key=lambda r: _excel_serial_to_iso(r["date"]) or "")
        aggregated.append(
            {
                "benchmark": VERIFIED_KEY,
                "model": model,
                "score": avg,
                "date": _excel_serial_to_iso(latest["date"]),
                "approach_type": latest["approach_type"],
                "runs": len(scores),
            }
        )

    aggregated.sort(key=lambda r: r["score"], reverse=True)
    return aggregated


def get_verified_scores(foundation_only: bool = True) -> list[dict[str, Any]]:
    data = fetch_bytes(OSWORLD_XLSX_URL)
    rows = parse_rows(data)
    scores = aggregate(rows, foundation_only=foundation_only)
    # A setup cell spelled differently ("no" for "No", "100 steps" for 100)
    # drops every row from the Foundation filter without touching the header.
    require_rows(scores, OSWORLD_XLSX_URL, "Foundation E2E GUI rows" if foundation_only else "rows")
    check_percentages(scores, OSWORLD_XLSX_URL, "score")
    return scores


def check_v2_payload(payload: Any) -> list[dict[str, Any]]:
    """The payload's result rows, once it is the board this script expects.

    Raises on anything that would change what a row means without changing its
    shape: another benchmark's file, a different task count, or a metric that
    is no longer called what it was.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError(f"{OSWORLD_V2_JSON_URL} returned no results list")
    version = payload.get("benchmarkVersion")
    if version != V2_BENCHMARK_VERSION:
        raise ValueError(
            f"{OSWORLD_V2_JSON_URL} describes {version!r}, not {V2_BENCHMARK_VERSION!r} "
            "-- refusing to read another benchmark's scores into the OSWorld 2.0 columns."
        )
    size = payload.get("datasetSize")
    if size != V2_DATASET_SIZE:
        raise ValueError(
            f"{OSWORLD_V2_JSON_URL} scores {size!r} tasks, not {V2_DATASET_SIZE} "
            "-- the task set changed; refusing to report it as the same column."
        )
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or V2_METRIC not in metrics:
        raise ValueError(
            f"{OSWORLD_V2_JSON_URL} no longer declares the {V2_METRIC!r} metric "
            f"(metrics: {metrics!r}) -- the layout changed; refusing to guess."
        )
    results = payload["results"]
    require_rows(results, OSWORLD_V2_JSON_URL, "OSWorld 2.0 results")
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            raise ValueError(f"{OSWORLD_V2_JSON_URL}: result {index} is not an object")
        require_columns(row.keys(), V2_REQUIRED_FIELDS, f"{OSWORLD_V2_JSON_URL} result {index}")
    return results


def _v2_release(row: dict[str, Any], payload: dict[str, Any]) -> str | None:
    """The release a row was run on, resolved the way the board resolves it.

    The paper's rows name none and inherit the payload's default, so a row is
    only as dated as the file says it is -- the same fallback chain the site's
    leaderboard.js walks.
    """
    for value in (
        row.get("releaseVersion"),
        row.get("taskVersion"),
        payload.get("defaultResultReleaseVersion"),
        payload.get("taskVersion"),
    ):
        if isinstance(value, str) and value:
            return value
    return None


def _v2_in_scope(row: dict[str, Any], payload: dict[str, Any], scope: str) -> bool:
    """Whether the board lists this row under ``scope``, as leaderboard.js decides."""
    row_scope = (
        row.get("datasetScope") or row.get("scope")
        or payload.get("defaultResultDatasetScope") or "full"
    )
    available = row.get("availableScopes")
    return row_scope == scope or (isinstance(available, list) and scope in available)


def parse_v2(payload: Any) -> list[dict[str, Any]]:
    """One entry per official full-set run at the 500-step budget, per tracked release."""
    results = check_v2_payload(payload)
    rows: list[dict[str, Any]] = []
    unknown: dict[str, int] = defaultdict(int)
    for result in results:
        if result.get("official") is not True:
            continue
        if result.get("stepBudget") != V2_STEP_BUDGET:
            continue
        if not _v2_in_scope(result, payload, V2_DATASET_SCOPE):
            continue
        model = result.get("model")
        if not isinstance(model, str) or not model.strip():
            continue
        release = _v2_release(result, payload)
        key = RELEASES.get(release or "")
        if key is None:
            unknown[release or "<none>"] += 1
            continue
        score = result.get(V2_METRIC)
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        partial = result.get("partialScore")
        rows.append(
            {
                "benchmark": key,
                "model": model.strip(),
                "score": score,
                "partial_score": partial if isinstance(partial, (int, float)) else None,
                "release": release,
                "reasoning": result.get("reasoning"),
                "tool_setting": result.get("toolSetting"),
                "step_budget": result.get("stepBudget"),
            }
        )
    for release, count in sorted(unknown.items()):
        # Not an error: a release is only filed once llm.json has a column
        # for it. Reported so a new one is noticed rather than lost.
        print(
            f"fetch_osworld.py: skipped {count} OSWorld 2.0 row(s) on release "
            f"{release!r}, which has no llm.json column",
            file=sys.stderr,
        )
    require_rows(rows, OSWORLD_V2_JSON_URL, "official full-set 500-step OSWorld 2.0 rows")
    check_percentages(rows, OSWORLD_V2_JSON_URL, "score")
    rows.sort(key=lambda r: (KEYS.index(r["benchmark"]), -r["score"]))
    return rows


def get_v2_scores() -> list[dict[str, Any]]:
    data = fetch_bytes(OSWORLD_V2_JSON_URL)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{OSWORLD_V2_JSON_URL} returned no JSON: {exc}") from exc
    return parse_v2(payload)


def get_scores(foundation_only: bool = True, boards: str = "all") -> list[dict[str, Any]]:
    """Rows for the requested boards: "verified", "v2", or "all" (both)."""
    scores: list[dict[str, Any]] = []
    if boards in ("all", "verified"):
        scores.extend(get_verified_scores(foundation_only=foundation_only))
    if boards in ("all", "v2"):
        scores.extend(get_v2_scores())
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch OSWorld-Verified and OSWorld 2.0 leaderboard scores."
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--board",
        choices=["all", "verified", "v2"],
        default="all",
        help="Which board to read: OSWorld-Verified, OSWorld 2.0, or both (default).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--foundation-only",
        action="store_true",
        default=True,
        help="Only include Foundation E2E GUI entries from OSWorld-Verified (default).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        dest="all_entries",
        help="Include all OSWorld-Verified entries regardless of setup.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    foundation_only = not args.all_entries
    scores = get_scores(foundation_only=foundation_only, boards=args.board)

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        # One name per line even where a model is on several boards or runs:
        # the mapping is by label, not by row.
        for name in dict.fromkeys(entry["model"] for entry in scores):
            print(name)
    else:
        key_width = max(len("COLUMN"), max((len(e["benchmark"]) for e in scores), default=0))
        model_width = max(len("MODEL"), max((len(e["model"]) for e in scores), default=0))
        fmt = f"{{:<{key_width}}}  {{:<{model_width}}}  {{:>7}}  {{}}"
        print(fmt.format("COLUMN", "MODEL", "SCORE", "DETAILS"))
        for entry in scores:
            if entry["benchmark"] == VERIFIED_KEY:
                details = f"{entry.get('date') or ''}  {entry.get('approach_type') or ''}"
            else:
                details = (
                    f"{entry.get('release')}  {entry.get('reasoning') or ''}  "
                    f"{entry.get('tool_setting') or ''}  partial {entry.get('partial_score')}"
                )
            print(fmt.format(entry["benchmark"], entry["model"], str(entry["score"]), details))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
