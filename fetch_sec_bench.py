#!/usr/bin/env python3
"""
Fetch SEC-bench Pro scores (snapshots 260505 and 260617) from https://sec-bench.github.io

SEC-bench Pro republishes itself as dated snapshots, each a different task set:

  * ``260505``  183 tasks, V8 and SpiderMonkey.
  * ``260617``  344 tasks, the same two plus 137 Linux kernel tasks -- the
                site's current default.

Each snapshot is its own column. ``sec_bench_pro`` is **260505**, because that
is the set the vendors report on --
OpenAI's system cards say so in as many words, and the DeepSeek and Xiaomi
cards quote numbers on the same scale -- and the maintainers still publish it
at ``/260505/``. Their board on the newer snapshot scores the same model up to
twelve points higher (GPT-5.5: 46.4 on 260505, 58.4 on 260617), so a row from
one would not rank against a row from the other. ``sec_bench_pro_260617`` is
the newer set, credited to the site root, which is where it is published.

The site is a static build whose pages inline their data; the build's own
source, ``data/results.json`` in the site repository, carries every snapshot
in one file, and that is what this reads. Only the ``overall`` board of the
260505 snapshot is taken, and from it only ``score_modes.headline`` -- solved
over all instances, timeouts counted as failures. The ``completed`` mode drops
the timeouts from the denominator, which moves a model that timed out on most
of its tasks (Opus 4.6 there: 114 of 183) from 25.7 to 60.9; the per-target
boards are V8 or SpiderMonkey alone. The same holds for each snapshot read.

A tracked snapshot the file no longer carries, or one whose overall board
states a task count other than its own (183, 344), stops the read rather than
filing a different task set under its column; a snapshot the file carries
beyond the tracked ones is reported and left unread.
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
from _fetch_checks import check_percentages, require_rows


KEY = "sec_bench_pro"
SNAPSHOT = "260505"
INSTANCES = 183
SITE_URL = "https://sec-bench.github.io"
# The snapshot's own page: what a score is credited to and what a reader opens.
URL = f"{SITE_URL}/{SNAPSHOT}/"

# Every snapshot llm.json carries a column for: snapshot -> (column, the
# overall board's task count, the page a score is credited to). 260505 is the
# set vendors report on; 260617 adds the Linux kernel (344 tasks) and is what
# the site root shows, having no page of its own. A snapshot the data file
# carries beyond these is reported, not read.
SNAPSHOTS: dict[str, tuple[str, int, str]] = {
    SNAPSHOT: (KEY, INSTANCES, URL),
    "260617": ("sec_bench_pro_260617", 344, f"{SITE_URL}/"),
}
KEYS = tuple(key for key, _instances, _url in SNAPSHOTS.values())


def source_url(key: str) -> str:
    """The page a score in column ``key`` is credited to."""
    for column, _instances, url in SNAPSHOTS.values():
        if column == key:
            return url
    raise KeyError(key)
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


def overall_board(payload: Any, snapshot_id: str = SNAPSHOT) -> dict:
    """A snapshot's overall board, checked to be the task set its column holds."""
    instances = SNAPSHOTS[snapshot_id][1]
    snapshots = payload.get("snapshots") if isinstance(payload, dict) else None
    snapshot = (snapshots or {}).get(snapshot_id)
    if not isinstance(snapshot, dict):
        raise ValueError(
            f"{DATA_URL} no longer carries snapshot {snapshot_id} -- refusing to read "
            "another snapshot's task set into this column."
        )
    for board in snapshot.get("leaderboards") or []:
        if isinstance(board, dict) and board.get("name") == OVERALL_BOARD:
            if board.get("instances") != instances:
                raise ValueError(
                    f"snapshot {snapshot_id}'s overall board states {board.get('instances')!r} "
                    f"tasks, not {instances} -- the task set changed under the same name."
                )
            return board
    raise ValueError(f"snapshot {snapshot_id} has no {OVERALL_BOARD!r} board in {DATA_URL}")


def unread_snapshots(payload: Any) -> list[str]:
    """Snapshots the data file carries that llm.json has no column for."""
    snapshots = payload.get("snapshots") if isinstance(payload, dict) else None
    return sorted(s for s in (snapshots or {}) if s not in SNAPSHOTS)


def parse_rows(payload: Any, snapshot_id: str = SNAPSHOT) -> list[dict]:
    rows: list[dict] = []
    key = SNAPSHOTS[snapshot_id][0]
    for row in overall_board(payload, snapshot_id).get("results") or []:
        if not isinstance(row, dict) or not isinstance(row.get("model"), str):
            continue
        headline = (row.get("score_modes") or {}).get("headline") or {}
        score = headline.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        rows.append(
            {
                "key": key,
                "snapshot": snapshot_id,
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


def get_scores(fetch=fetch_json, snapshots: tuple[str, ...] | None = None) -> list[dict]:
    """One dict per row of each tracked snapshot's overall board.

    Keys: key (the llm.json column), snapshot, model, raw, agent, effort,
    backend, solved, total, timeouts, date, score (headline %, timeouts counted
    as failures), rank -- within its own snapshot, since two snapshots are two
    task sets.
    """
    print(f"Fetching {DATA_URL} ...", file=sys.stderr)
    payload = fetch(DATA_URL)
    for snapshot_id in unread_snapshots(payload):
        message = f"SEC-bench Pro snapshot {snapshot_id} has no llm.json column; not read"
        print(f"  warning: {message}", file=sys.stderr)
        _fetch_warnings.record("fetch_sec_bench.py", message, source="new-version:sec_bench")
    rows: list[dict] = []
    for snapshot_id in snapshots or tuple(SNAPSHOTS):
        board = parse_rows(payload, snapshot_id)
        require_rows(board, f"{DATA_URL}#{snapshot_id}")
        check_percentages(board, SNAPSHOTS[snapshot_id][2])
        board.sort(key=lambda r: -r["score"])
        for i, row in enumerate(board, 1):
            row["rank"] = i
        print(f"  snapshot {snapshot_id}: {len(board)} leaderboard rows", file=sys.stderr)
        rows.extend(board)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch SEC-bench Pro scores, one board per tracked snapshot, from sec-bench.github.io."
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
        print(f"{'SNAPSHOT':<8}  {'MODEL':<{width}}  {'SCORE':>6}  AGENT")
        for e in scores:
            print(f"{e['snapshot']:<8}  {e['model']:<{width}}  {e['score']:>6}  {e.get('agent') or ''}")
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
