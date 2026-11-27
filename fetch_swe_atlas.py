#!/usr/bin/env python3
"""
Fetch SWE Atlas (Task Resolve Rate %) scores from Scale Labs leaderboards.

SWE Atlas (Scale AI / Scale Labs) is a coding-agent benchmark with three
sub-tracks -- Codebase QnA, Refactoring, Test Writing -- over 284 tasks, and
it has published two revisions of itself that are not comparable:

  * v1.0 -- one board per track (labs.scale.com/leaderboard/sweatlas-{qna,
    refactoring,tw}), judged by Claude Opus 4.5. Scale marks these boards
    "Legacy" since v1.1 (deprecated in the site's own board list); they are
    frozen (newest row 2026-09-09) and read into swe_atlas_{qna,rf,tw}_1_0.
  * v1.1 (2026-11-06) -- one page, labs.scale.com/leaderboard/swe-atlas, with
    an Overall table and the three tracks as variant tables. Same 284 tasks,
    but every model re-run with a new judge (GPT-6.1 Sol, majority of three),
    refined criteria and restricted network access, so Scale itself says the
    scores are not comparable with v1.0 (Fable 5.1 QnA: 59.95 on v1.0, 79.03
    on v1.1). Read into swe_atlas_{qna,rf,tw}_1_1 and swe_atlas_1_1 (Overall).

The current page's revision is read off the page itself: the site's board
list names the table behind it ("airtableTableName":"sweatlas_1.1 Overall").
A page naming no revision, or one llm.json has no column for (see
_revisions.KNOWN_REVISIONS), is fetched but not filed: its rows carry no key
and are reported on stderr, so a v1.2 is noticed rather than written into
1.1's columns.

The pages are a Next.js App Router app (no classic __NEXT_DATA__ tag). The
leaderboard rows live in the React Server Component flight payload emitted as
self.__next_f.push([1,"..."]) chunks; _scale_labs reads them without a
headless browser, mirroring fetch_frontierswe.py.

Leaderboard model names embed the harness (e.g. "Opus 4.8 (Claude Code)",
"GLM 5.2 (Mini-SWE-Agent)", "Claude Fable 5.2 (Claude Code) max") and vary in
punctuation across boards ("Opus 4.8" vs "Opus-4.8", "Mini-SWE-Agent" vs
"Mini-SWE"). The reported `model` is therefore a normalized base name (harness
and reasoning modifiers stripped) so a single name->slug mapping covers every
board/harness variant; `raw` and `harness` are kept for transparency.
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

import _fetch_warnings
from _fetch_checks import check_percentages
from _revisions import KNOWN_REVISIONS, known_revision_key, revision_key, revision_label
from _scale_labs import NoRowsError, decode_flight, extract_board_rows, fetch_board_rows
import _source_pages


# v1.0: one legacy board per track.
LEGACY_URL = "https://labs.scale.com/leaderboard/sweatlas-{track}"
LEGACY_REVISION = "1.0"

# Legacy track slug on the Scale leaderboard -> llm.json benchmark base key
# (the column is the base plus the revision, see _revisions.revision_key).
LEGACY_TRACKS: dict[str, str] = {
    "qna": "swe_atlas_qna",
    "refactoring": "swe_atlas_rf",
    "tw": "swe_atlas_tw",
}

# v1.1 onwards: one page, a variant table per track plus Overall.
CURRENT_URL = "https://labs.scale.com/leaderboard/swe-atlas"
CURRENT_SLUG = "swe-atlas"
CURRENT_BOARD = "current"
CURRENT_VARIANTS: dict[str, str] = {
    "overall": "swe_atlas",
    "qna": "swe_atlas_qna",
    "refactoring": "swe_atlas_rf",
    "testwriting": "swe_atlas_tw",
}

# Every page this fetcher reads, in the order rows are reported: the three
# legacy track boards, then the current page.
BOARDS: list[str] = [*LEGACY_TRACKS, CURRENT_BOARD]

# The site's board list names the table behind each board; the current page's
# is "sweatlas_1.1 Overall", which is where its revision is read.
_CURRENT_REVISION_RE = re.compile(
    r'"id":"' + re.escape(CURRENT_SLUG) + r'","airtableTableName":"sweatlas_(\d+(?:\.\d+)*)'
)


def board_url(board: str) -> str:
    return CURRENT_URL if board == CURRENT_BOARD else LEGACY_URL.format(track=board)


def _column_urls() -> dict[str, str]:
    urls = {
        revision_key(base, LEGACY_REVISION): LEGACY_URL.format(track=track)
        for track, base in LEGACY_TRACKS.items()
    }
    for base in CURRENT_VARIANTS.values():
        for label in KNOWN_REVISIONS.get(base, ()):
            if label != LEGACY_REVISION:
                urls[revision_key(base, label)] = CURRENT_URL
    return urls


# llm.json column -> the page its values are read from: 1.0's on each legacy
# track board, every later revision's on the current page.
COLUMN_URLS: dict[str, str] = _column_urls()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Seconds before a board that failed is read once more. See get_scores().
SECOND_PASS_DELAY = 60.0

# Reasoning-effort modifiers (case-insensitive). Only a word standing on its
# own is one: Scale writes the effort after a space ("Opus 5 (Claude Code)
# xHigh", "Gpt 5.4 xHigh (Mini-SWE-Agent)"), while a hyphen joins a word that
# is part of the name, so "Qwen3.8-Max" stays its own model instead of
# collapsing onto the base "qwen3.8" (issue #233). v1.1 writes the name with a
# space ("Qwen 3.8 Max (mini-swe-agent) max"), so where an effort trails the
# harness, that trailing one is the effort and the name is left whole.
_EFFORT_RE = re.compile(r"(?<![\w-])(?:xhigh|x-high|high|medium|low|max)(?![\w-])", re.IGNORECASE)


def fetch_html(url: str, retries: int = 3, delay: float = 2.0) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            # A 404 or 403 says the same thing two seconds later; only a
            # rate limit or a server/network error is worth asking again.
            if attempt == retries or _is_final(exc):
                raise
            wait = delay * attempt
            print(
                f"  attempt {attempt}/{retries} failed ({exc}); retrying in {wait:.0f}s ...",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def _is_final(exc: Exception) -> bool:
    return isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500 and exc.code != 429


def extract_rows(html: str, track: str) -> list[dict]:
    """Rows of one legacy track's board; raises if the page is another board or empty."""
    return extract_board_rows(html, f"sweatlas-{track}", LEGACY_URL.format(track=track))


def current_revision(html: str) -> str | None:
    """The revision the current page names for itself ("1.1"), or None."""
    match = _CURRENT_REVISION_RE.search(decode_flight(html))
    return revision_label(match.group(1)) if match else None


def read_current(html: str) -> tuple[str | None, dict[str, list[dict]]]:
    """(revision, variant -> rows) off the current page.

    Every variant in CURRENT_VARIANTS has to be there: a page that lost one is
    refused (NoRowsError for a page with no tables at all, ValueError for one
    naming other variants) rather than read in part.
    """
    tables = {
        variant: extract_board_rows(html, CURRENT_SLUG, CURRENT_URL, variant)
        for variant in CURRENT_VARIANTS
    }
    return current_revision(html), tables


def fetch_current(attempts: int = 3, delay: float = 10.0) -> tuple[str | None, dict[str, list[dict]]]:
    """read_current over a fresh fetch, re-fetching a page with no rows (see
    _scale_labs.fetch_board_rows, which does the same for one table)."""
    for attempt in range(1, attempts + 1):
        try:
            return read_current(fetch_html(CURRENT_URL))
        except NoRowsError as exc:
            if attempt == attempts:
                raise
            wait = delay * attempt
            print(
                f"  {CURRENT_URL}: no rows on attempt {attempt}/{attempts} ({exc}); "
                f"re-fetching in {wait:.0f}s ...",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


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
    An effort written after the harness is the only one stripped, so a name
    that itself ends in Max keeps it: "Qwen 3.8 Max (mini-swe-agent) max" ->
    "qwen 3.8 max".
    """
    without, _ = split_harness(raw)
    trailing = re.search(r"\)([^()]*)$", raw)
    if trailing and _EFFORT_RE.search(trailing.group(1)):
        name = re.sub(r"\s*\([^)]*\)\s*", " ", raw[: trailing.start() + 1])
        without = f"{name} {_EFFORT_RE.sub(' ', trailing.group(1))}"
    else:
        without = _EFFORT_RE.sub(" ", without)
    without = without.replace("-", " ").replace("_", " ")
    without = re.sub(r"\s+", " ", without).strip().lower()
    return without


def _row(board: str, track: str, base: str, revision: str | None, obj: dict) -> dict:
    raw = obj["model"]
    _, harness = split_harness(raw)
    return {
        "board": board,
        "track": track,
        "base": base,
        "revision": revision,
        "key": known_revision_key(base, revision),
        "url": board_url(board),
        "model": normalize_model(raw),
        "raw": raw,
        "harness": harness,
        "score": round(float(obj["score"]), 2),
        "ci": obj.get("confidenceInterval_upper"),
        "company": obj.get("company"),
    }


def _ranked(rows: list[dict], source: str) -> list[dict]:
    check_percentages(rows, source)
    rows.sort(key=lambda r: -r["score"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def get_scores(boards: list[str]) -> list[dict]:
    """Return a flat list of row dicts across the requested boards.

    Keys: board (legacy track slug, or "current"), track (the board's track,
    or the current page's variant: overall/qna/refactoring/testwriting), base
    (llm.json base key), revision ("1.0" on the legacy boards, the page's own
    on the current one), key (the llm.json column, or None when llm.json has
    no column for that revision), url, model (normalized base), raw, harness,
    score, ci, company, rank (rank within the track, 1 = best).
    """
    def fetch_board(board: str):
        url = board_url(board)
        print(f"Fetching {url} ...", file=sys.stderr)
        if board == CURRENT_BOARD:
            return fetch_current()
        return fetch_board_rows(fetch_html, url, f"sweatlas-{board}")

    def try_board(board: str):
        try:
            return fetch_board(board)
        except (urllib.error.URLError, OSError, NoRowsError) as exc:
            return exc

    # One page per board on one host; fetched together, parsed in order.
    with ThreadPoolExecutor(max_workers=len(boards) or 1) as pool:
        pages = list(pool.map(try_board, boards))

    # Scale's board pages fail for minutes at a time and then serve again: a
    # 404 for a board its own index still lists, or the page with no rows
    # (42 runs to 2026-10-05: 8 such failures, every one of them across all
    # three tracks and none lasting to the next run). fetch_html gives up on a
    # 404 at once and fetch_board_rows re-reads an empty page for half a
    # minute, so a failed board gets one more pass a minute later. Only the
    # failures that would be skipped are retried; a page that is another
    # board still fails the source on the first read.
    retry = [i for i, page in enumerate(pages) if isinstance(page, Exception)]
    if retry:
        names = ", ".join(boards[i] for i in retry)
        print(
            f"  SWE Atlas board(s) {names} failed; one more pass in {SECOND_PASS_DELAY:.0f}s ...",
            file=sys.stderr,
        )
        time.sleep(SECOND_PASS_DELAY)
        for i in retry:
            pages[i] = try_board(boards[i])

    # A board whose page is still unreachable or empty after that (sweatlas-tw
    # has 404'd for hours at a time while the other two served) costs that board
    # only: its columns stay where the last good run left them, since scores
    # are applied, never cleared. It is still a warning on the run. A page that
    # is another board, or reads wrong, still fails the whole source, and so
    # does every board failing at once.
    failed = [(b, p) for b, p in zip(boards, pages) if isinstance(p, Exception)]
    if len(failed) == len(boards):
        raise failed[0][1]
    for board, exc in failed:
        url = board_url(board)
        print(f"warning: SWE Atlas board {board} skipped this run: {url}: {exc}", file=sys.stderr)
        _fetch_warnings.record(
            "fetch_swe_atlas.py", f"{url}: {exc}", source=f"swe_atlas ({board})"
        )

    results: list[dict] = []
    for board, page in zip(boards, pages):
        if isinstance(page, Exception):
            continue
        if board != CURRENT_BOARD:
            base = LEGACY_TRACKS[board]
            print(f"  parsed {len(page)} rows for {board} (v{LEGACY_REVISION})", file=sys.stderr)
            rows = [_row(board, board, base, LEGACY_REVISION, obj) for obj in page]
            results.extend(_ranked(rows, board_url(board)))
            continue
        revision, tables = page
        unfiled = [
            base for base in CURRENT_VARIANTS.values()
            if known_revision_key(base, revision) is None
        ]
        if unfiled:
            # Fetched, not filed: update.py writes a row only under its key.
            what = f"v{revision}" if revision else "no revision"
            message = (
                f"{CURRENT_URL}: the page names {what}; llm.json has no column for "
                f"{', '.join(unfiled)} at that revision, so those rows are not filed"
            )
            print(f"warning: {message}", file=sys.stderr)
            _fetch_warnings.record("fetch_swe_atlas.py", message, source="swe_atlas (current)")
        for variant, objs in tables.items():
            base = CURRENT_VARIANTS[variant]
            print(f"  parsed {len(objs)} rows for {variant} (v{revision})", file=sys.stderr)
            rows = [_row(board, variant, base, revision, obj) for obj in objs]
            results.extend(_ranked(rows, f"{CURRENT_URL} ({variant})"))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch SWE Atlas leaderboard scores.")
    parser.add_argument(
        "--track",
        choices=[*BOARDS, "all"],
        default="all",
        help="Which board to fetch: a legacy v1.0 track board, 'current' for the "
        "v1.1+ page with every track, or all (default).",
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
    boards = list(BOARDS) if args.track == "all" else [args.track]
    scores = get_scores(boards)

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "pages":
        # Which board each name was read from, for the admin page's link.
        _source_pages.emit(scores, lambda e: e["url"])
    elif args.format == "names":
        for name in sorted({entry["model"] for entry in scores}):
            print(name)
    else:
        col_widths = [
            max(len("MODEL"), max((len(e["model"]) for e in scores), default=0)),
            6,
            max(len("COLUMN"), max((len(str(e["key"])) for e in scores), default=0)),
            max(len("RAW"), max((len(e["raw"]) for e in scores), default=0)),
        ]
        fmt = f"{{:<{col_widths[0]}}}  {{:>{col_widths[1]}}}  {{:<{col_widths[2]}}}  {{:<{col_widths[3]}}}"
        print(fmt.format("MODEL", "SCORE", "COLUMN", "RAW"))
        for entry in scores:
            print(fmt.format(entry["model"], str(entry["score"]), str(entry["key"]), entry["raw"]))
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
