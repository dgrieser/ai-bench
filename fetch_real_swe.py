#!/usr/bin/env python3
"""
Fetch Real-SWE resolution rates from https://realswe.withspecific.com/

Real-SWE (Specific Labs, September 2026) runs frontier coding agents against
tasks lifted from *private* production codebases licensed from real companies:
the code, the tickets and the reference solutions are nowhere on the public
internet, which is the property the benchmark is built around. Each task ships
in Harbor format with a verifier injected at grading time, and every model is
run eight independent times per task, so the published number is a pass@1
averaged over eight rollouts rather than a single attempt.

The board is a model *and harness* pairing -- Fable 5.1 under Claude Code,
GPT-6 Astra under Codex CLI, GLM 5.3 under Claude Code -- because that is how
the benchmark is run, so every row reports the `harness` it was measured with
beside the score.

There is no API: the site is a Next.js app whose RSC payload carries the
leaderboard as rendered markup rather than as data (there is no `entries`
object to read, the way fetch_frontierswe.py reads one), so the table is parsed
out of the served HTML. The page renders the same eight rows twice -- an <ol>
for phones and a <table> for everything else -- and this script reads the table
inside the `leaderboard` container only, so no model is counted twice.

The 95% confidence interval the page says it shows exists only as the geometry
of the whisker drawn over each bar, so it is reported as `ci_low`/`ci_high`
when it can be read and left None when it cannot. It is never the score: the
score is the percentage the board prints.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request


URL = "https://realswe.withspecific.com/"

# The container the leaderboard is rendered into, and the figure inside it. Both
# are read: the id survives a restyle, and the label is what says the figure is
# the model board rather than one of the two analysis tables further down the
# page.
LEADERBOARD_ID = "leaderboard"
LEADERBOARD_LABEL = "Model leaderboard"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.DOTALL | re.IGNORECASE)
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
# "38.8%" -- the number the board prints, which is the score.
_PERCENT_RE = re.compile(r"(-?[0-9]+(?:\.[0-9]+)?)\s*%")
# "=5" marks a tie on this board; "5" a clear position.
_RANK_RE = re.compile(r"^(=)?\s*([0-9]+)$")
# The whisker drawn over the bar: the 95% interval, as CSS rather than as text.
_WHISKER_RE = re.compile(
    r'data-confidence-whisker="true"[^>]*style="([^"]*)"', re.IGNORECASE
)
_LEFT_RE = re.compile(r"left:\s*(-?[0-9.]+)%")
_WIDTH_RE = re.compile(r"width:\s*(-?[0-9.]+)%")


def fetch_html(url: str = URL, retries: int = 3, delay: float = 2.0) -> str:
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


def _text(markup: str) -> str:
    text = _TAG_RE.sub(" ", html.unescape(markup))
    return re.sub(r"\s+", " ", text).strip()


def select_leaderboard_table(page_html: str) -> str:
    """Body of the model leaderboard table.

    The page carries three tables -- the board, the per-task pass matrix and the
    effort table -- and only the first is a list of models with one score each.
    Picking it by position would silently start reading the task matrix the day
    a section moves, so the container is located first and the table taken from
    inside it; if the container is gone this raises instead of guessing.
    """
    anchor = page_html.find(f'id="{LEADERBOARD_ID}"')
    if anchor == -1:
        anchor = page_html.find(f'aria-label="{LEADERBOARD_LABEL}"')
    if anchor == -1:
        raise ValueError(
            f"No {LEADERBOARD_ID!r} container and no {LEADERBOARD_LABEL!r} figure on "
            f"{URL} -- the page layout changed; refusing to guess which table is "
            "the leaderboard."
        )
    match = _TABLE_RE.search(page_html, anchor)
    if match is None:
        raise ValueError(f"No <table> after the leaderboard container on {URL}")
    return match.group(1)


def _confidence_interval(cell_markup: str) -> tuple[float | None, float | None]:
    """The 95% interval drawn over one bar, as (low, high) percentage points.

    The bar and the whisker are both laid out against a full-width 0-100 scale
    -- the top model's 38.75% bar is 38.75% wide -- so the whisker's left edge
    and width read straight off as the interval's ends. Decoration this is read
    from, so anything unexpected yields (None, None) rather than an error: a
    missing interval must never cost the score beside it.
    """
    whisker = _WHISKER_RE.search(cell_markup)
    if whisker is None:
        return (None, None)
    style = whisker.group(1)
    left, width = _LEFT_RE.search(style), _WIDTH_RE.search(style)
    if left is None or width is None:
        return (None, None)
    low = float(left.group(1))
    high = low + float(width.group(1))
    return (round(low, 1), round(high, 1))


def parse_rows(table_body: str) -> list[dict]:
    """Leaderboard rows as dicts, keyed by the table's own header labels."""
    header: list[str] = []
    rows: list[dict] = []
    for tr in _TR_RE.findall(table_body):
        cells = _CELL_RE.findall(tr)
        if not cells:
            continue
        if not header:
            header = [_text(c).lower().strip("# ").strip() or "rank" for c in cells]
            continue
        values = {header[i]: cells[i] for i in range(min(len(header), len(cells)))}
        rows.append(values)
    if not header:
        raise ValueError(f"The leaderboard table on {URL} has no header row")
    for required in ("model", "resolution rate"):
        if required not in header:
            raise ValueError(
                f"The leaderboard table on {URL} has no {required!r} column; "
                f"it now reads: {', '.join(header)}"
            )
    return rows


def _rank(raw: str) -> tuple[int | None, bool]:
    """(position, tied) from the board's own rank cell, which prints '=5' for a tie."""
    match = _RANK_RE.match(_text(raw))
    if match is None:
        return (None, False)
    return (int(match.group(2)), bool(match.group(1)))


def get_scores() -> list[dict]:
    """Return a list of dicts: model, harness, score, rank, tied, ci_low, ci_high.

    `score` is the resolution rate the board prints -- pass@1 over eight
    independent runs per task, as a percentage. `rank` and `tied` are the
    board's own position for the row rather than a re-derived one, so two models
    published level stay level here.
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    rows = parse_rows(select_leaderboard_table(fetch_html()))

    scores: list[dict] = []
    for row in rows:
        model = _text(row.get("model", ""))
        if not model:
            continue
        cell = row.get("resolution rate", "")
        percent = _PERCENT_RE.search(_text(cell))
        if percent is None:
            continue
        rank, tied = _rank(row.get("rank", ""))
        ci_low, ci_high = _confidence_interval(cell)
        scores.append(
            {
                "model": model,
                "harness": _text(row.get("harness", "")) or None,
                "score": round(float(percent.group(1)), 2),
                "rank": rank,
                "tied": tied,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    print(f"  parsed {len(scores)} Real-SWE row(s)", file=sys.stderr)
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Real-SWE leaderboard scores.")
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
        model_width = max([len("MODEL"), *(len(e["model"]) for e in scores)])
        harness_width = max(
            [len("HARNESS"), *(len(e.get("harness") or "") for e in scores)]
        )
        fmt = f"{{:>4}}  {{:<{model_width}}}  {{:>6}}  {{:<{harness_width}}}  {{:<13}}"
        print(fmt.format("#", "MODEL", "SCORE", "HARNESS", "95% CI"))
        for entry in scores:
            rank = entry["rank"]
            position = "" if rank is None else f"{'=' if entry['tied'] else ''}{rank}"
            low, high = entry["ci_low"], entry["ci_high"]
            interval = "" if low is None or high is None else f"{low:.1f}-{high:.1f}"
            print(
                fmt.format(
                    position,
                    entry["model"],
                    f"{entry['score']:.1f}",
                    entry.get("harness") or "",
                    interval,
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
        print(f"error: could not fetch the Real-SWE leaderboard: {exc}", file=sys.stderr)
        raise SystemExit(1)
