#!/usr/bin/env python3
"""
Fetch ProgramBench almost-resolved rates from https://programbench.com

ProgramBench (Meta AI Research and collaborators, 2026) asks whether a model can
rebuild a program *from scratch*: the agent is given a reference executable and
its documentation, nothing else, and has to architect and implement a codebase
whose behaviour matches. The 200 tasks run from compact CLI tools up to FFmpeg,
SQLite and the PHP interpreter, and grading is by hidden behavioural tests
against the rebuilt binary. Every published run is mini-SWE-agent with the
internet switched off -- the authors' FAQ explains that allowing it mostly
produced solutions that had found the original source.

There are two pages and they are the same runs, not two boards:

  * ``/``          the top ten rows, which is the page a reader lands on.
  * ``/extended/`` every row of the same evaluation, plus cost and call counts.
                   "Extended" here means the full table, NOT a larger task set
                   -- the header states "200 tasks" on both -- so this is not
                   the Main/Extended split FrontierCode publishes and there is
                   nothing to keep apart. This script reads /extended/ because
                   it is the complete list.

**The score is Almost Resolved, and that is a deliberate choice this file has to
keep making.** The board prints three quantities and the authors rule on them in
the FAQ. Resolved -- every behavioural test passing -- is "the primary metric
that should be reported", and it is what ``resolved`` carries here. Almost
Resolved, at least 95% of the tests passing, is published "as an additional
point of reference while the scores of our primary metric are low", with the
caveat that even one failed test out of the 15k some tasks carry "can indicate
severe issues with a program". An average test pass rate is the one they rule
out entirely -- it "would be extremely misleading", because every task carries
trivial tests (does the binary exist, does ``--help`` work) that a program doing
nothing useful still passes -- and this scraper never reports it.

``score`` is Almost Resolved because Resolved is, today, almost entirely floor:
the best published run resolves 4.5% of the 200 tasks and most rows resolve
none, which ranks the top few models and calls everything below them equal.
Almost separates the same field over a real range. It is the more permissive of
the two readings the authors publish, and it is read as what it is -- how close
a rebuild came, not whether it was correct. Resolved rides along on every row so
the strict number is never more than a field away, and the column it feeds says
in its own name which of the two it stores. See README, "Why ProgramBench enters
at 0.35".

Rows name an effort where the run used one -- "Claude Opus 5 (xhigh)", "GPT 5.5
(high)" beside a bare "GPT 5.5" -- and the label is reported as published. The
suffix is not stripped: two efforts of one model are two runs, and the ingest's
own "best reported run wins" rule is what collapses them onto a slug.
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


SITE_URL = "https://programbench.com"
# The complete table. The site root shows the top ten of these same runs, so
# reading the root as well would double every row it carries.
LEADERBOARD_URL = f"{SITE_URL}/extended/"

# The class the leaderboard table is rendered with, on both pages. Picking the
# table by position would silently start reading something else the day a
# section moves.
LEADERBOARD_CLASS = "lb-table"

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
# Every header cell carries a help popover whose text would otherwise be read as
# part of the column name ("Resolved The number of fully solved instances ..."),
# and a second, abbreviated copy of the label for narrow screens ("Resolved"
# beside "Res."). Both are stripped before a header is read, so the column is
# named what the table calls it.
_POPOVER_RE = re.compile(
    r'<span class="info-popover">.*?</span>', re.DOTALL | re.IGNORECASE
)
_ICON_RE = re.compile(r'<span class="material-icons[^"]*">.*?</span>', re.DOTALL)
_MOBILE_LABEL_RE = re.compile(
    r'<span class="th-mobile">.*?</span>', re.DOTALL | re.IGNORECASE
)
# The board prints "4.5%" on /extended/ and "4.5<span class="pct">%</span>" on
# the root, so the percent sign is read after the markup is stripped.
_PERCENT_RE = re.compile(r"(-?[0-9]+(?:\.[0-9]+)?)\s*%")
_RANK_RE = re.compile(r"^([0-9]+)$")
# "Claude Opus 5 (xhigh)" -> the effort half, kept beside the published label
# rather than removed from it.
_EFFORT_RE = re.compile(r"\(([^)]+)\)\s*$")
# "Evaluated with mini-SWE-agent - 200 tasks - Updated Sep. 9, 2026"
_TASK_COUNT_RE = re.compile(r"([0-9]+)\s+tasks", re.IGNORECASE)


def fetch_html(url: str = LEADERBOARD_URL, retries: int = 3, delay: float = 2.0) -> str:
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
    stripped = _MOBILE_LABEL_RE.sub(" ", _POPOVER_RE.sub(" ", _ICON_RE.sub(" ", markup)))
    text = _TAG_RE.sub(" ", html.unescape(stripped))
    return re.sub(r"\s+", " ", text).strip()


def select_leaderboard_table(page_html: str) -> str:
    """Body of the leaderboard table.

    Raises rather than falling back to the first table on the page: a layout
    change that moved the board would otherwise be read as every model's score
    disappearing at once.
    """
    for match in _TABLE_RE.finditer(page_html):
        opening = page_html[match.start() : match.start(1)]
        if LEADERBOARD_CLASS in opening:
            return match.group(1)
    raise ValueError(
        f"No <table class={LEADERBOARD_CLASS!r}> on {LEADERBOARD_URL} -- the page "
        "layout changed; refusing to guess which table is the leaderboard."
    )


def task_count(page_html: str) -> int | None:
    """Tasks the published runs cover, from the board's own subtitle.

    Reported rather than used: the score is already a percentage, so this is a
    reader's sanity check on how coarse it is -- at 200 tasks one task is half a
    point, which is why the board's values step in halves.
    """
    match = _TASK_COUNT_RE.search(_text(page_html[: page_html.find("<table")]))
    return int(match.group(1)) if match else None


def parse_rows(table_body: str) -> list[dict]:
    """Leaderboard rows as dicts, keyed by the table's own header labels."""
    header: list[str] = []
    rows: list[dict] = []
    for tr in _TR_RE.findall(table_body):
        cells = _CELL_RE.findall(tr)
        if not cells:
            continue
        if not header:
            # The logo column has an empty header, and naming it after a
            # neighbour would let it overwrite that neighbour's cell; it is
            # given a positional name instead.
            header = [
                _text(c).lower().strip("# ").strip() or f"column{i}"
                for i, c in enumerate(cells)
            ]
            continue
        rows.append({header[i]: cells[i] for i in range(min(len(header), len(cells)))})
    if not header:
        raise ValueError(f"The leaderboard table on {LEADERBOARD_URL} has no header row")
    for required in ("model", "resolved", "almost"):
        if required not in header:
            raise ValueError(
                f"The leaderboard table on {LEADERBOARD_URL} has no {required!r} "
                f"column; it now reads: {', '.join(header)}"
            )
    return rows


def _percent(cell_markup: str) -> float | None:
    """A percentage cell, or None. '0%' is a real score here, not a missing one."""
    match = _PERCENT_RE.search(_text(cell_markup))
    return round(float(match.group(1)), 2) if match else None


def _effort(model: str) -> str | None:
    """The effort a row names, if it names one: 'Claude Opus 5 (xhigh)' -> 'xhigh'."""
    match = _EFFORT_RE.search(model)
    return match.group(1).strip() if match else None


def get_scores() -> list[dict]:
    """Return a list of score dicts for the ProgramBench leaderboard.

    Keys: model (the published label, effort suffix included), effort, agent,
    score (Almost Resolved %), resolved (the strict Resolved %, reported but not
    scored), cost_per_task, calls_per_task, tasks, rank (the board's own
    position).
    """
    print(f"Fetching {LEADERBOARD_URL} ...", file=sys.stderr)
    page = fetch_html()
    rows = parse_rows(select_leaderboard_table(page))
    tasks = task_count(page)

    scores: list[dict] = []
    for row in rows:
        model = _text(row.get("model", ""))
        if not model:
            continue
        almost = _percent(row.get("almost", ""))
        if almost is None:
            continue
        rank = _RANK_RE.match(_text(row.get("rank", "")))
        cost = _text(row.get("cost", "")).lstrip("$")
        calls = _text(row.get("calls", ""))
        scores.append(
            {
                "model": model,
                "effort": _effort(model),
                "agent": _text(row.get("agent", "")) or None,
                "score": almost,
                "resolved": _percent(row.get("resolved", "")),
                "cost_per_task": float(cost) if re.fullmatch(r"[0-9.]+", cost) else None,
                "calls_per_task": int(calls) if calls.isdigit() else None,
                "tasks": tasks,
                "rank": int(rank.group(1)) if rank else None,
            }
        )

    print(
        f"  parsed {len(scores)} ProgramBench row(s)"
        + (f" over {tasks} tasks" if tasks else ""),
        file=sys.stderr,
    )
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch ProgramBench leaderboard scores (Almost Resolved %)."
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
        width = max([len("MODEL"), *(len(e["model"]) for e in scores)])
        agent_width = max([len("AGENT"), *(len(e.get("agent") or "") for e in scores)])
        fmt = f"{{:>4}}  {{:<{width}}}  {{:>7}}  {{:>9}}  {{:<{agent_width}}}"
        print(fmt.format("#", "MODEL", "ALMOST", "RESOLVED", "AGENT"))
        for entry in scores:
            resolved = entry["resolved"]
            print(
                fmt.format(
                    "" if entry["rank"] is None else str(entry["rank"]),
                    entry["model"],
                    f"{entry['score']:.1f}%",
                    "" if resolved is None else f"{resolved:.1f}%",
                    entry.get("agent") or "",
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
        print(
            f"error: could not fetch the ProgramBench leaderboard: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
