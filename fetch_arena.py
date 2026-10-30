#!/usr/bin/env python3
"""
Fetch Arena (formerly LMArena) leaderboard scores, https://arena.ai/leaderboard

Arena ranks models by anonymous pairwise votes from people using it on their
own prompts, so it measures something no other column here does: which answer
a person prefers. arena.ai refuses scripted requests, but Arena publishes every
board as a CC BY 4.0 dataset on Hugging Face, ``lmarena-ai/leaderboard-dataset``,
one subset per arena with a ``latest`` split holding the current board. That is
what this script reads, one parquet file per subset.

Five columns, from four subsets:

  * ``arena_text`` -- the Text arena's ``overall`` category on the
    style-controlled board, the one arena.ai shows by default since May 2025:
    a Bradley-Terry rating with the effect of answer length and markdown
    regressed out.
  * ``arena_text_non_english`` -- the same board's ``non_english`` category,
    the votes cast on prompts in any language but English.
  * ``arena_webdev`` -- the Code Arena (``webdev``), where the two models each
    build a web app from the user's request and the user votes on the result.
  * ``arena_agent`` and ``arena_agent_tool_hallucination`` -- Agent Arena, which
    scores live agent sessions rather than votes. Its scores are IPS estimates
    (τ̂), signed effects around the field's average, stored here x100: the
    aggregate, and the per-signal score for tool calls the model hallucinated,
    where a higher score means fewer of them.

Rating boards keep a row only with ``MIN_VOTES`` battles behind it; below that
the confidence interval is wider than the gaps the column is read for.

Row labels carry the reasoning effort in a parenthetical -- "GLM-5.3 (Max)",
"Claude Opus 5.5 (High)" -- so the reported ``model`` is the label with run
settings stripped, lower-cased, and ``raw`` keeps the original. A
parenthetical naming anything else ("(0528)", "(NVFP4)") stays in the key, the
rule fetch_zerobench.py follows. Every configuration of one model maps to its
slug and update.py keeps the best, as on every board.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

from _fetch_checks import require_columns, require_rows
import _source_pages

DATASET = "lmarena-ai/leaderboard-dataset"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET}"
TREE_URL = f"https://huggingface.co/api/datasets/{DATASET}/tree/main/{{subset}}"
FILE_URL = f"{DATASET_URL}/resolve/main/{{path}}"
LEADERBOARD_URL = "https://arena.ai/leaderboard"

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ai-bench-fetcher/1.0"}

# Fewest battles a rating-board row needs to be read.
MIN_VOTES = 500


@dataclass(frozen=True)
class Board:
    """One column: the dataset subset and category it reads, and how."""

    subset: str
    category: str
    page: str
    # "rating" (Bradley-Terry, stored as published) or "ips" (stored x100).
    kind: str


BOARDS: dict[str, Board] = {
    "arena_text": Board("text_style_control", "overall", "text", "rating"),
    "arena_text_non_english": Board("text_style_control", "non_english", "text", "rating"),
    "arena_webdev": Board("webdev", "overall", "webdev", "rating"),
    "arena_agent": Board("agent", "overall", "agent", "ips"),
    "arena_agent_tool_hallucination": Board("agent_tool_hallucination", "overall", "agent", "ips"),
}

# The columns each kind of subset has to carry.
RATING_COLUMNS = ("model_name", "rating", "vote_count", "category", "leaderboard_publish_date")
IPS_COLUMNS = ("model_name", "score", "category", "leaderboard_publish_date")

# Bounds a value of each kind must lie in: Arena ratings sit around 1000-1600,
# and an IPS effect is a fraction of a session's outcome, so beyond +-1 is a
# file on some other scale.
RATING_BOUNDS = (0.0, 3000.0)
IPS_BOUNDS = (-1.0, 1.0)

_MODIFIER_TOKENS = frozenset(
    {
        "xhigh", "x-high", "high", "medium", "low", "max", "min", "minimal", "default",
        "effort", "reasoning", "non-reasoning", "thinking", "non-thinking", "extended",
    }
)


def board_url(key: str) -> str:
    """The arena.ai page a column's scores are credited to."""
    return f"{LEADERBOARD_URL}/{BOARDS[key].page}"


def fetch_bytes(url: str, retries: int = 3, delay: float = 2.0) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
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


def latest_path(subset: str, fetch_json=None) -> str:
    """The path of a subset's ``latest`` parquet file, looked up rather than
    assumed: the shard name is the dataset tooling's, not a promise."""
    fetch_json = fetch_json or (lambda url: json.loads(fetch_bytes(url)))
    listing = fetch_json(TREE_URL.format(subset=subset))
    paths = sorted(
        entry["path"]
        for entry in listing
        if isinstance(entry, dict)
        and str(entry.get("path", "")).startswith(f"{subset}/latest-")
        and str(entry.get("path", "")).endswith(".parquet")
    )
    if len(paths) != 1:
        raise ValueError(
            f"{DATASET}: expected one latest parquet file under {subset}/, found {paths} -- "
            "the dataset layout changed; refusing to guess."
        )
    return paths[0]


def read_parquet(data: bytes) -> list[dict[str, Any]]:
    """The rows of one parquet file, read from memory on this thread only.

    Handed a Python file object (io.BytesIO), pyarrow wraps it in a
    PyReadableFile that an Arrow worker thread may be the last to release --
    and at interpreter exit that thread waits for a GIL the finalizing main
    thread never gives back, so the process hangs after it has printed its
    output (17 of 20 runs did, and in CI it held update_arena_mapping.py until
    the job was cancelled). A native buffer and a single-threaded read leave
    no Python object behind on an Arrow thread.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pq.read_table(pa.BufferReader(data), use_threads=False).to_pylist()


def split_effort(raw: str) -> tuple[str, str | None]:
    """("GLM-5.3 (Max)") -> ("GLM-5.3", "Max"); no parenthetical -> (raw, None)."""
    match = re.search(r"\(([^)]*)\)\s*$", raw)
    if not match:
        return raw.strip(), None
    return raw[: match.start()].strip(), match.group(1).strip()


def is_run_setting(parenthetical: str | None) -> bool:
    if not parenthetical:
        return False
    tokens = [t for t in re.split(r"[\s,/]+", parenthetical.strip().lower()) if t]
    return bool(tokens) and all(t in _MODIFIER_TOKENS for t in tokens)


def normalize_model(raw: str) -> str:
    """The mapping key for a label: run settings dropped, case and spacing folded.

    "GLM-5.3 (Max)" -> "glm-5.3"; "DeepSeek-R1 (0528)" -> "deepseek-r1 (0528)".
    """
    base, setting = split_effort(raw)
    name = base if is_run_setting(setting) or setting is None else f"{base} ({setting})"
    return re.sub(r"\s+", " ", name).strip().lower()


def check_bounds(values: Iterable[float], source: str, bounds: tuple[float, float]) -> None:
    low, high = bounds
    outside = [v for v in values if not low <= v <= high]
    if outside:
        raise ValueError(
            f"{source}: values outside {low:g}-{high:g} ({outside[:3]}) -- the scale "
            "changed; refusing to store them."
        )


def parse_board(key: str, rows: list[dict[str, Any]], source: str) -> list[dict]:
    """One column's rows out of its subset's table."""
    board = BOARDS[key]
    if rows:
        require_columns(rows[0].keys(), RATING_COLUMNS if board.kind == "rating" else IPS_COLUMNS, source)
    picked = [r for r in rows if r.get("category") == board.category]
    require_rows(picked, source, what=f"{board.category!r} rows")
    out: list[dict] = []
    values: list[float] = []
    for row in picked:
        raw = str(row.get("model_name") or "").strip()
        value = row.get("rating" if board.kind == "rating" else "score")
        if not raw or not isinstance(value, (int, float)):
            continue
        values.append(float(value))
        if board.kind == "rating" and (row.get("vote_count") or 0) < MIN_VOTES:
            continue
        out.append(
            {
                "key": key,
                "model": normalize_model(raw),
                "raw": raw,
                "license": row.get("license"),
                "score": round(float(value), 2) if board.kind == "rating" else round(float(value) * 100, 2),
                "votes": row.get("vote_count") if board.kind == "rating" else row.get("observation_count"),
                "published": row.get("leaderboard_publish_date"),
                "source": board_url(key),
            }
        )
    check_bounds(values, source, RATING_BOUNDS if board.kind == "rating" else IPS_BOUNDS)
    require_rows(out, source)
    out.sort(key=lambda r: -r["score"])
    return out


def get_scores(keys: list[str] | None = None, fetch=None, fetch_json=None) -> list[dict]:
    """Rows for every column (or the ones named): key, model, raw, license,
    score, votes, published, source. Each subset is downloaded once."""
    fetch = fetch or fetch_bytes
    keys = keys or list(BOARDS)
    tables: dict[str, tuple[list[dict[str, Any]], str]] = {}
    results: list[dict] = []
    for key in keys:
        subset = BOARDS[key].subset
        if subset not in tables:
            path = latest_path(subset, fetch_json)
            url = FILE_URL.format(path=path)
            print(f"Fetching {url} ...", file=sys.stderr)
            tables[subset] = (read_parquet(fetch(url)), url)
        rows, url = tables[subset]
        results.extend(parse_board(key, rows, url))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Arena leaderboard scores.")
    parser.add_argument(
        "--board", choices=["all", *BOARDS], default="all", help="Column to read (default: all)."
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "names", "pages"],
        default="table",
        help="Output format (default: table).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = get_scores(None if args.board == "all" else [args.board])
    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for name in sorted({r["model"] for r in scores}):
            print(name)
    elif args.format == "pages":
        # A name is asked about once; its link is the first board listing it.
        _source_pages.emit(sorted(scores, key=lambda r: r["model"]), lambda r: r["source"])
    else:
        width = max([len("MODEL"), *(len(r["raw"]) for r in scores)])
        fmt = f"{{:<30}}  {{:<{width}}}  {{:>8}}"
        print(fmt.format("COLUMN", "MODEL", "SCORE"))
        for r in scores:
            print(fmt.format(r["key"], r["raw"], r["score"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch the Arena leaderboard dataset: {exc}", file=sys.stderr)
        raise SystemExit(1)
