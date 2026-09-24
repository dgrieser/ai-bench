#!/usr/bin/env python3
"""
Fetch SWE Atlas (Task Resolve Rate %) scores from Scale Labs leaderboards.

SWE Atlas (Scale AI / Scale Labs) is a coding-agent benchmark with three
independent sub-tracks, each on its own leaderboard page:
  - QnA          Codebase Q&A          -> swe_atlas_qna
  - refactoring  Refactoring           -> swe_atlas_rf
  - tw           Test Writing          -> swe_atlas_tw

The pages are a Next.js App Router app (no classic __NEXT_DATA__ tag). The
leaderboard rows live in the React Server Component flight payload emitted as
self.__next_f.push([1,"..."]) chunks. This script concatenates the decoded
flight chunks and extracts the row objects (model, score, ...) without a
headless browser, mirroring fetch_frontierswe.py.

Leaderboard model names embed the harness (e.g. "Opus 4.8 (Claude Code)",
"GLM 5.2 (Mini-SWE-Agent)", "GPT 5.5 (Codex) xHigh") and vary in punctuation
across tracks ("Opus 4.8" vs "Opus-4.8", "Mini-SWE-Agent" vs "Mini-SWE"). The
reported `model` is therefore a normalized base name (harness and reasoning
modifiers stripped) so a single name->slug mapping covers every track/harness
variant; `raw` and `harness` are kept for transparency.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from _fetch_checks import check_percentages
from _scale_labs import extract_board_rows, fetch_board_rows


BASE_URL = "https://labs.scale.com/leaderboard/sweatlas-{track}"

# Track slug on the Scale leaderboard -> llm.json benchmark key.
TRACKS: dict[str, str] = {
    "qna": "swe_atlas_qna",
    "refactoring": "swe_atlas_rf",
    "tw": "swe_atlas_tw",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Reasoning-effort modifiers that trail a model name (case-insensitive). Only
# a word standing on its own is one: Scale writes the effort after a space
# ("Opus 5 (Claude Code) xHigh", "Gpt 5.4 xHigh (Mini-SWE-Agent)"), while a
# hyphen joins a word that is part of the name, so "Qwen3.8-Max" stays its own
# model instead of collapsing onto the base "qwen3.8" (issue #233).
_EFFORT_RE = re.compile(r"(?<![\w-])(?:xhigh|x-high|high|medium|low|max)(?![\w-])", re.IGNORECASE)


def fetch_html(url: str, retries: int = 3, delay: float = 2.0) -> str:
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


def extract_rows(html: str, track: str) -> list[dict]:
    """Rows of one track's board; raises if the page is another board or empty."""
    return extract_board_rows(html, f"sweatlas-{track}", BASE_URL.format(track=track))


def split_harness(raw: str) -> tuple[str, str | None]:
    """Split a leaderboard label into (display_without_harness, harness).

    "Opus 4.8 (Claude Code)"            -> ("Opus 4.8", "Claude Code")
    "Gpt 5.4 xHigh (Mini-SWE-Agent)"    -> ("Gpt 5.4 xHigh", "Mini-SWE-Agent")
    "Muse Spark"                        -> ("Muse Spark", None)
    """
    match = re.search(r"\(([^)]*)\)", raw)
    harness = match.group(1).strip() if match else None
    without = re.sub(r"\s*\([^)]*\)\s*", " ", raw).strip()
    return without, harness


def normalize_model(raw: str) -> str:
    """Normalize a leaderboard label to a base model name for mapping.

    Strips the harness parenthetical and reasoning-effort modifiers, unifies
    separators, and lowercases. e.g. "GPT-5.5 (Codex) xHigh" -> "gpt 5.5".
    """
    without, _ = split_harness(raw)
    without = _EFFORT_RE.sub(" ", without)
    without = without.replace("-", " ").replace("_", " ")
    without = re.sub(r"\s+", " ", without).strip().lower()
    return without


def get_scores(tracks: list[str]) -> list[dict]:
    """Return a flat list of row dicts across the requested tracks.

    Keys: track, key, model (normalized base), raw, harness, score, ci,
    company, rank (rank within the track, 1 = best).
    """
    def fetch_track(track: str) -> list[dict]:
        url = BASE_URL.format(track=track)
        print(f"Fetching {url} ...", file=sys.stderr)
        return fetch_board_rows(fetch_html, url, f"sweatlas-{track}")

    # One page per track on one host; fetched together, parsed in order.
    with ThreadPoolExecutor(max_workers=len(tracks) or 1) as pool:
        pages = list(pool.map(fetch_track, tracks))

    results: list[dict] = []
    for track, rows in zip(tracks, pages):
        key = TRACKS[track]
        print(f"  parsed {len(rows)} rows for {track}", file=sys.stderr)
        track_rows: list[dict] = []
        for obj in rows:
            raw = obj["model"]
            without, harness = split_harness(raw)
            track_rows.append(
                {
                    "track": track,
                    "key": key,
                    "model": normalize_model(raw),
                    "raw": raw,
                    "harness": harness,
                    "score": round(float(obj["score"]), 2),
                    "ci": obj.get("confidenceInterval_upper"),
                    "company": obj.get("company"),
                }
            )
        check_percentages(track_rows, BASE_URL.format(track=track))
        track_rows.sort(key=lambda r: -r["score"])
        for i, r in enumerate(track_rows, 1):
            r["rank"] = i
        results.extend(track_rows)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch SWE Atlas leaderboard scores.")
    parser.add_argument(
        "--track",
        choices=[*TRACKS.keys(), "all"],
        default="all",
        help="Which sub-track leaderboard to fetch (default: all).",
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
    tracks = list(TRACKS.keys()) if args.track == "all" else [args.track]
    scores = get_scores(tracks)

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for name in sorted({entry["model"] for entry in scores}):
            print(name)
    else:
        col_widths = [
            max(len("MODEL"), max((len(e["model"]) for e in scores), default=0)),
            6,
            5,
            max(len("RAW"), max((len(e["raw"]) for e in scores), default=0)),
        ]
        fmt = f"{{:<{col_widths[0]}}}  {{:>{col_widths[1]}}}  {{:<{col_widths[2]}}}  {{:<{col_widths[3]}}}"
        print(fmt.format("MODEL", "SCORE", "TRACK", "RAW"))
        for entry in scores:
            print(fmt.format(entry["model"], str(entry["score"]), entry["track"], entry["raw"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch SWE Atlas leaderboard: {exc}", file=sys.stderr)
        raise SystemExit(1)
