#!/usr/bin/env python3
"""
Fetch Terminal-Bench 4.0, 2.1 and 2.0 scores from https://www.tbench.ai/

tbench.ai renders only the current board on the server; every other version
is a client-side switch (``/leaderboard/terminal-bench/2.0`` redirects to
``/?version=2.0``, whose server payload is the same 4.0 board). What the
version picker calls instead is the Harbor Hub's public ``leaderboard-read``
function, one POST per board, naming the dataset package and the leaderboard:

    {"package": "terminal-bench/terminal-bench-2", "name": "2-0"}

and the answer is the board as plain JSON, {"leaderboard": {...}, "rows": [...]}.
This script asks for each board by that pair and checks the descriptor that
comes back names the same pair, so a board can never be read into another
revision's column -- the silent version bump the homepage payload invited,
where a 5.0 release would have taken 4.0's slot, cannot happen to a request
that names the board it wants.

One row is one (agent, model, reasoning effort) run, e.g. Claude Code / Fable
5.1 / max, so the reported ``model`` is the row's model label alone and the
agent and effort are kept beside it for transparency; update.py folds the
variants onto one llm.json slug per column, best run first.

The 2.0 board predates the others' schema and takes submissions: of its rows
about half are ``"verified": false`` self-reports, and a few run several models
at once (``model_display`` "Multiple"). Only verified single-model rows are
read -- an unverified harness claim is not the benchmark's own measurement, and
an ensemble has no one model to credit. The 4.0 and 2.1 boards carry no such
flag; every row there is a run with a Harbor Hub job behind it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

from _fetch_checks import check_percentages, require_rows


# What this script requests: the function tbench.ai's own version picker calls.
URL = "https://ofhuhcpkvzjlejydnvyd.supabase.co/functions/v1/leaderboard-read"
# What a reader opens, and the pages a score is credited to. Each board has its
# own, so the precedence prefix is the versioned leaderboard rather than the
# whole tbench.ai host.
PAGE_URL = "https://www.tbench.ai/leaderboard/terminal-bench/{version}"

# llm.json column -> the board that feeds it.
BOARDS: dict[str, dict[str, str]] = {
    "terminal_bench_4_0": {
        "version": "4.0",
        "package": "terminal-bench/terminal-bench",
        "name": "4-0-0",
    },
    "terminal_bench_2_1": {
        "version": "2.1",
        "package": "terminal-bench/terminal-bench-2-1",
        "name": "main",
    },
    "terminal_bench_2_0": {
        "version": "2.0",
        "package": "terminal-bench/terminal-bench-2",
        "name": "2-0",
    },
}

# The current board's page, which fill_source_urls.py lists as what URL covers.
LEADERBOARD_URL = PAGE_URL.format(version=BOARDS["terminal_bench_4_0"]["version"])


def board_url(key: str) -> str:
    """The leaderboard page a score in column ``key`` is credited to."""
    return PAGE_URL.format(version=BOARDS[key]["version"])


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/json",
    "Origin": "https://www.tbench.ai",
}


def fetch_board(package: str, name: str, retries: int = 3, delay: float = 2.0) -> dict:
    body = json.dumps({"package": package, "name": name}).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers=HEADERS, method="POST")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            break
        except (urllib.error.URLError, OSError) as exc:
            if attempt == retries:
                raise
            wait = delay * attempt
            print(
                f"  attempt {attempt}/{retries} failed ({exc}); retrying in {wait:.0f}s ...",
                file=sys.stderr,
            )
            time.sleep(wait)
    try:
        board = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{URL} returned no JSON for {package} {name}: {exc}")
    if not isinstance(board, dict) or not isinstance(board.get("rows"), list):
        raise ValueError(f"{URL} returned no leaderboard rows for {package} {name}")
    return board


def check_version(board: dict, package: str, name: str) -> dict:
    """The leaderboard descriptor, once it is the board that was asked for.

    Raises rather than reading whatever came back: a board filed under the
    wrong revision would look like every model moving at once.
    """
    leaderboard = board.get("leaderboard")
    if not isinstance(leaderboard, dict):
        raise ValueError(f"No leaderboard descriptor in {URL}'s answer for {package} {name}")
    got_name = leaderboard.get("name")
    got_package = leaderboard.get("package")
    if got_name != name or (got_package is not None and got_package != package):
        raise ValueError(
            f"{URL} answered {package} {name} with leaderboard {got_name!r} of "
            f"package {got_package!r} -- refusing to read another board's scores "
            "into this column."
        )
    return leaderboard


def _label(value: object) -> str | None:
    """Label of a {"url": ..., "label": ...} metadata field, or a bare string.

    The 2.0 board spells model_display as a plain string; the later boards
    wrap every display field in a link.
    """
    if isinstance(value, dict):
        value = value.get("label")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def read_rows(board: dict, key: str) -> list[dict]:
    """One entry per readable run on the board, filed under column ``key``."""
    source = board_url(key)
    results: list[dict] = []
    for row in board["rows"]:
        if not isinstance(row, dict):
            continue
        metadata = row.get("metadata")
        metrics = row.get("metrics")
        if not isinstance(metadata, dict) or not isinstance(metrics, dict):
            continue
        # Present on the 2.0 board only; absent means a run the board made.
        if metadata.get("verified") is False:
            continue
        model_names = metadata.get("model_names")
        if isinstance(model_names, list) and len(model_names) > 1:
            continue
        model = _label(metadata.get("model_display"))
        accuracy = metrics.get("accuracy")
        if model is None or not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
            continue
        results.append(
            {
                "benchmark": key,
                "model": model,
                "raw": model,
                "agent": _label(metadata.get("agent_display")),
                "effort": metadata.get("reasoning_effort") or None,
                "org": _label(metadata.get("model_org")),
                "score": round(float(accuracy), 2),
                "ci95": metrics.get("accuracy_ci95_half_width"),
                "n_trials": metrics.get("n_trials"),
                "date": metadata.get("date") or None,
            }
        )

    # A renamed "accuracy" or "model_display" drops every row one at a time.
    require_rows(results, source)
    check_percentages(results, source)

    results.sort(key=lambda r: -r["score"])
    for i, entry in enumerate(results, 1):
        entry["rank"] = i
    return results


def get_scores(keys: list[str] | None = None) -> list[dict]:
    """Return a list of dicts: benchmark, model, raw, agent, effort, org, score,
    ci95, n_trials, date, rank.

    ``score`` is the resolution rate the leaderboard prints, as a percentage.
    One entry per (agent, model, effort) row; rank is within its own board,
    1 = best.
    """
    results: list[dict] = []
    for key in keys or list(BOARDS):
        spec = BOARDS[key]
        print(f"Fetching {spec['package']} {spec['name']} from {URL} ...", file=sys.stderr)
        board = fetch_board(spec["package"], spec["name"])
        leaderboard = check_version(board, spec["package"], spec["name"])
        rows = read_rows(board, key)
        print(
            f"  reading {leaderboard.get('title') or spec['name']}:"
            f" {len(rows)} of {len(board['rows'])} rows",
            file=sys.stderr,
        )
        results.extend(rows)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Terminal-Bench leaderboard scores.")
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--board",
        choices=list(BOARDS),
        action="append",
        help="Read only this column's board (repeatable; default: all).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = get_scores(args.board)

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for name in sorted({entry["model"] for entry in scores}):
            print(name)
    else:
        widths = [
            max([len("BOARD")] + [len(e["benchmark"]) for e in scores]),
            max([len("MODEL")] + [len(e["model"]) for e in scores]),
            6,
            max([len("AGENT")] + [len(e["agent"] or "") for e in scores]),
            max([len("EFFORT")] + [len(e["effort"] or "") for e in scores]),
        ]
        fmt = (
            f"{{:<{widths[0]}}}  {{:<{widths[1]}}}  {{:>{widths[2]}}}  "
            f"{{:<{widths[3]}}}  {{:<{widths[4]}}}"
        )
        print(fmt.format("BOARD", "MODEL", "SCORE", "AGENT", "EFFORT"))
        for entry in scores:
            print(
                fmt.format(
                    entry["benchmark"], entry["model"], str(entry["score"]),
                    entry["agent"] or "", entry["effort"] or "",
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
        print(f"error: could not fetch the Terminal-Bench leaderboard: {exc}", file=sys.stderr)
        raise SystemExit(1)
