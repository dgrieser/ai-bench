#!/usr/bin/env python3
"""
Fetch SEC-bench Pro scores (snapshot 260505) from https://sec-bench.github.io

SEC-bench Pro republishes itself as dated snapshots, each a different task set:

  * ``260505``  183 tasks, V8 and SpiderMonkey.
  * ``260617``  344 tasks, the same two plus 137 Linux kernel tasks -- the
                site's current default.

The column is **260505**, because that is the set the vendors report on --
OpenAI's system cards say so in as many words, and the DeepSeek and Xiaomi
cards quote numbers on the same scale -- and the maintainers still publish it
at ``/260505/``. Their board on the newer snapshot scores the same model up to
twelve points higher (GPT-5.5: 46.4 on 260505, 58.4 on 260617), so a row from
one would not rank against a row from the other.

The site is a static build whose pages inline their data; the build's own
source, ``data/results.json`` in the site repository, carries every snapshot
in one file, and that is what this reads. Only the ``overall`` board of the
260505 snapshot is taken, and from it only ``score_modes.headline`` -- solved
over all instances, timeouts counted as failures. The ``completed`` mode drops
the timeouts from the denominator, which moves a model that timed out on most
of its tasks (Opus 4.6 there: 114 of 183) from 25.7 to 60.9; the per-target
boards are V8 or SpiderMonkey alone.

A snapshot the file no longer carries, or one whose overall board states a
task count other than 183, stops the read rather than filing a different task
set under this column.
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

from _fetch_checks import check_percentages, require_rows


KEY = "sec_bench_pro"
SNAPSHOT = "260505"
INSTANCES = 183
SITE_URL = "https://sec-bench.github.io"
# The snapshot's own page: what a score is credited to and what a reader opens.
URL = f"{SITE_URL}/{SNAPSHOT}/"
# What this script requests: the site build's data, every snapshot in one file.
DATA_URL = "https://raw.githubusercontent.com/SEC-bench/sec-bench.github.io/main/data/results.json"
OVERALL_BOARD = "overall"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_json(url: str = DATA_URL, retries: int = 3, delay: float = 2.0) -> Any:
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
    return re.sub(r"\s+", " ", raw).strip()


def overall_board(payload: Any) -> dict:
    """The 260505 snapshot's overall board, checked to be the 183-task set."""
    snapshots = payload.get("snapshots") if isinstance(payload, dict) else None
    snapshot = (snapshots or {}).get(SNAPSHOT)
    if not isinstance(snapshot, dict):
        raise ValueError(
            f"{DATA_URL} no longer carries snapshot {SNAPSHOT} -- refusing to read "
            "another snapshot's task set into this column."
        )
    for board in snapshot.get("leaderboards") or []:
        if isinstance(board, dict) and board.get("name") == OVERALL_BOARD:
            if board.get("instances") != INSTANCES:
                raise ValueError(
                    f"snapshot {SNAPSHOT}'s overall board states {board.get('instances')!r} "
                    f"tasks, not {INSTANCES} -- the task set changed under the same name."
                )
            return board
    raise ValueError(f"snapshot {SNAPSHOT} has no {OVERALL_BOARD!r} board in {DATA_URL}")


def parse_rows(payload: Any) -> list[dict]:
    rows: list[dict] = []
    for row in overall_board(payload).get("results") or []:
        if not isinstance(row, dict) or not isinstance(row.get("model"), str):
            continue
        headline = (row.get("score_modes") or {}).get("headline") or {}
        score = headline.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        rows.append(
            {
                "model": normalize_model(row["model"]),
                "raw": row["model"],
                "agent": row.get("agent"),
                "effort": row.get("effort"),
                "backend": row.get("backend"),
                "solved": headline.get("solved"),
                "total": headline.get("total"),
                "timeouts": row.get("timeouts"),
                "date": row.get("date"),
                "score": round(float(score), 2),
            }
        )
    return rows


def get_scores(fetch=fetch_json) -> list[dict]:
    """One dict per row of the 260505 overall board.

    Keys: model, raw, agent, effort, backend, solved, total, timeouts, date,
    score (headline %, timeouts counted as failures), rank.
    """
    print(f"Fetching {DATA_URL} (snapshot {SNAPSHOT}) ...", file=sys.stderr)
    rows = parse_rows(fetch(DATA_URL))
    require_rows(rows, f"{DATA_URL}#{SNAPSHOT}")
    check_percentages(rows, URL)
    print(f"  parsed {len(rows)} leaderboard rows", file=sys.stderr)
    rows.sort(key=lambda r: -r["score"])
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Fetch SEC-bench Pro (snapshot {SNAPSHOT}) scores from sec-bench.github.io."
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
        width = max((len(e["model"]) for e in scores), default=5)
        print(f"{'MODEL':<{width}}  {'SCORE':>6}  AGENT")
        for e in scores:
            print(f"{e['model']:<{width}}  {e['score']:>6}  {e.get('agent') or ''}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch the SEC-bench Pro leaderboard: {exc}", file=sys.stderr)
        raise SystemExit(1)
