#!/usr/bin/env python3
"""
Fetch Terminal-Bench 4.0 scores from https://www.tbench.ai/

The site is a Next.js app whose leaderboard is server-rendered: requesting it
with an "RSC: 1" header returns the flight payload, and the react-query
dehydrated state in it carries the whole board as one plain JSON object,

    "data":{"leaderboard":{...},"rows":[...]}

which json.JSONDecoder can decode in place -- no headless browser, mirroring
fetch_aa_coding_agents.py and fetch_frontierswe.py.

The homepage always embeds the *current* leaderboard and nothing else: the
version picker is client-side, so ``?version=2.1`` returns the same 4.0 payload.
That makes a silent version bump the failure mode to avoid -- reading TB 5.0
numbers into the 4.0 column would look like every model suddenly moving -- so
this script asserts the leaderboard's own name and package and refuses to guess
when either changes.

One row is one (agent, model, reasoning effort) run, e.g. Claude Code / Fable
5.1 / max, so the reported ``model`` is the row's model label alone and the
agent and effort are kept beside it for transparency; update.py folds the
variants onto one llm.json slug, best run first.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request


# What this script requests: the app embeds the board in its own payload.
URL = "https://www.tbench.ai/"
# What a reader opens, and the page a score is credited to. Kept distinct from
# URL so the precedence prefix is the versioned leaderboard rather than the
# whole tbench.ai host, which also serves the 2.0 and 2.1 boards.
LEADERBOARD_URL = "https://www.tbench.ai/leaderboard/terminal-bench/4.0"

# The board this script is allowed to read, as the payload names it.
LEADERBOARD_NAME = "4-0-0"
PACKAGE = "terminal-bench/terminal-bench"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    # Ask for the React Server Component payload instead of the HTML shell.
    "RSC": "1",
}

# Where the dehydrated react-query cache holds the board.
_DATA_MARKER = '"data":{"leaderboard":'
_DATA_PREFIX = '"data":'
# Fallback for an HTML response (no RSC payload): the flight chunks the shell
# embeds, same shape fetch_frontierswe.py reads.
_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,(".*?")\]\)', re.DOTALL)


def fetch_payload(url: str = URL, retries: int = 3, delay: float = 2.0) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
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


def _decoded_chunks(payload: str) -> str:
    """Flight chunks of an HTML response, concatenated.

    Only reached when the server answers the RSC request with the HTML shell;
    the chunks carry the same payload, escaped inside script tags.
    """
    decoded = ""
    for chunk in _PUSH_RE.findall(payload):
        try:
            decoded += json.loads(chunk)
        except json.JSONDecodeError:
            continue
    return decoded


def extract_board(payload: str) -> dict:
    """The {"leaderboard": ..., "rows": [...]} object out of the flight payload."""
    for text in (payload, _decoded_chunks(payload)):
        index = text.find(_DATA_MARKER)
        if index == -1:
            continue
        try:
            board, _ = json.JSONDecoder().raw_decode(text, index + len(_DATA_PREFIX))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not decode the leaderboard payload on {URL}: {exc}")
        if isinstance(board, dict) and isinstance(board.get("rows"), list):
            return board
    raise ValueError(f"Could not find the leaderboard payload on {URL}")


def check_version(board: dict) -> dict:
    """The leaderboard descriptor, once it is the one this script reads.

    Raises rather than falling back to whatever board the homepage happens to
    serve: the version picker is client-side, so a new release silently takes
    this slot, and reading it as 4.0 would look like every model moving at once.
    """
    leaderboard = board.get("leaderboard")
    if not isinstance(leaderboard, dict):
        raise ValueError(f"No leaderboard descriptor in the payload on {URL}")
    name = leaderboard.get("name")
    package = leaderboard.get("package")
    if name != LEADERBOARD_NAME or (package is not None and package != PACKAGE):
        raise ValueError(
            f"{URL} now serves leaderboard {name!r} of package {package!r}, not "
            f"{LEADERBOARD_NAME!r} of {PACKAGE!r} -- refusing to read another "
            "version's scores into the Terminal-Bench 4.0 column."
        )
    return leaderboard


def _label(value: object) -> str | None:
    """Label of a {"url": ..., "label": ...} metadata field."""
    if isinstance(value, dict):
        label = value.get("label")
        if isinstance(label, str) and label.strip():
            return label.strip()
    return None


def get_scores() -> list[dict]:
    """Return a list of dicts: model, raw, agent, effort, org, score, ci95,
    n_trials, date, rank.

    ``score`` is the resolution rate the leaderboard prints, as a percentage.
    One entry per (agent, model, effort) row; rank is within this list, 1 = best.
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    board = extract_board(fetch_payload())
    leaderboard = check_version(board)
    print(
        f"  reading {leaderboard.get('title') or LEADERBOARD_NAME}"
        f" ({len(board['rows'])} rows)",
        file=sys.stderr,
    )

    results: list[dict] = []
    for row in board["rows"]:
        if not isinstance(row, dict):
            continue
        metadata = row.get("metadata")
        metrics = row.get("metrics")
        if not isinstance(metadata, dict) or not isinstance(metrics, dict):
            continue
        model = _label(metadata.get("model_display"))
        accuracy = metrics.get("accuracy")
        if model is None or not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
            continue
        results.append(
            {
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

    results.sort(key=lambda r: -r["score"])
    for i, entry in enumerate(results, 1):
        entry["rank"] = i
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Terminal-Bench 4.0 leaderboard scores.")
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
        widths = [
            max([len("MODEL")] + [len(e["model"]) for e in scores]),
            6,
            max([len("AGENT")] + [len(e["agent"] or "") for e in scores]),
            max([len("EFFORT")] + [len(e["effort"] or "") for e in scores]),
        ]
        fmt = f"{{:<{widths[0]}}}  {{:>{widths[1]}}}  {{:<{widths[2]}}}  {{:<{widths[3]}}}"
        print(fmt.format("MODEL", "SCORE", "AGENT", "EFFORT"))
        for entry in scores:
            print(
                fmt.format(
                    entry["model"], str(entry["score"]), entry["agent"] or "", entry["effort"] or ""
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
