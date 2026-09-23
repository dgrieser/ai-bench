#!/usr/bin/env python3
"""
Fetch ZeroBench (pass@1 %) scores from https://zerobench.github.io

ZeroBench is 100 hand-written visual reasoning questions that no model has yet
come close to solving, so the column is the rare unsaturated one -- and the
rare one where a few points of methodology is most of the score. The page
carries three tables and only the first is the maintainers' own run:

  * ``sortableTable``          The official board: the ZeroBench team's own
                               evaluation of each model. This is the one we read.
  * ``sortableTableProvider``  Externally reported numbers, lifted from model
                               cards, technical reports and release posts. Same
                               questions, somebody else's harness, judge and
                               tool budget -- and many rows are explicitly
                               "w/ tools", which on a near-zero benchmark
                               roughly doubles the number.
  * ``sortableTableRelease``   The models evaluated at ZeroBench's release,
                               under the single-greedy-sampling protocol the
                               board has since moved off. Its models all appear
                               on the official board too.

Reading the provider table would be the same mistake ``update.py`` already
makes nowhere else: the Hugging Face ingest is fill-only precisely because
model cards report the flattered variant, and this table *is* those cards. The
table is therefore selected by id and everything else on the page ignored.

The column is **pass@1 on the 100 main questions**. The 334 subquestions are a
different, three-to-four-times-easier question set; the official board does not
print them, which is a further reason to read this table rather than the other
two, both of which do.

Row labels carry the reasoning effort in a parenthetical -- "Claude Opus 5
(max)", "GPT-5.2 (medium reasoning)" -- so the reported ``model`` is a
normalized base name and ``raw`` keeps the original label. A parenthetical that
names the model rather than the run ("gpt-5.6 (sol)") is kept in the key, the
same rule fetch_mcp_atlas.py follows and for the same reason.

The board marks how each pass@1 was computed: ``●`` a mean over 5 samplings,
``○`` a single greedy sampling. Both are pass@1 over the same 100 questions, so
both are kept; ``sampling`` reports which, since the mean is the protocol the
board has used since the release cohort.
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

from _fetch_checks import check_percentages


URL = "https://zerobench.github.io"

# Id of the maintainers' own leaderboard. The two other tables on the page are
# externally reported numbers and the release cohort; see the module docstring.
OFFICIAL_TABLE_ID = "sortableTable"

# Columns the official table is expected to open with. Checked rather than
# assumed: the provider table starts "Model | pass@1 | pass@5" too, so a page
# that renamed its ids would otherwise be read straight into the column.
EXPECTED_HEADER = ("model", "pass@1", "pass@5", "pass^5")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.DOTALL | re.IGNORECASE)
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
_SCRIPT_RE = re.compile(r"<script\b.*?</script>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
# "17.2●" / "9.4●" / "0.0○" / "-"
_SCORE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*([●○])?")

# Reasoning-effort modifiers that trail a model name outside a parenthetical.
_EFFORT_RE = re.compile(r"\b(?:xhigh|x-high|high|medium|low|max)\b", re.IGNORECASE)
# Every token a parenthetical may contain and still be a run setting rather
# than a model name, the same list fetch_mcp_atlas.py applies to Scale's
# labels. Anything else in one names the model and is kept in the key.
_MODIFIER_TOKENS = frozenset(
    {
        "xhigh",
        "x-high",
        "high",
        "medium",
        "low",
        "max",
        "min",
        "default",
        "effort",
        "reasoning",
        "non-reasoning",
        "thinking",
        "non-thinking",
        "extended",
    }
)


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


def split_effort(raw: str) -> tuple[str, str | None]:
    """Split a leaderboard label into (label_without_parenthetical, modifier).

    "Claude Opus 5 (max)"       -> ("Claude Opus 5", "max")
    "gpt-5.6 (sol)"             -> ("gpt-5.6", "sol")
    "Claude Opus 4.8"           -> ("Claude Opus 4.8", None)
    """
    match = re.search(r"\(([^)]*)\)", raw)
    effort = match.group(1).strip() if match else None
    without = re.sub(r"\s*\([^)]*\)\s*", " ", raw).strip()
    return without, effort


def is_run_setting(parenthetical: str | None) -> bool:
    """Whether a parenthetical describes how a model was run, not which one."""
    if not parenthetical:
        return False
    tokens = [token for token in re.split(r"[\s,/]+", parenthetical.strip().lower()) if token]
    return bool(tokens) and all(token in _MODIFIER_TOKENS for token in tokens)


def normalize_model(raw: str) -> str:
    """Normalize a leaderboard label to a base model name for mapping.

    "Claude Opus 5 (max)" -> "claude opus 5", "GPT-5.2 (medium reasoning)" ->
    "gpt 5.2", "gpt-5.6 (sol)" -> "gpt 5.6 sol".
    """
    without, effort = split_effort(raw)
    if not is_run_setting(effort):
        without = f"{without} {effort}" if effort is not None else without
    without = _EFFORT_RE.sub(" ", without)
    # Commas and slashes only ever separate tokens inside a parenthetical
    # ("sol, max"), so they go the way of the other separators rather than
    # trailing the key once the effort beside them is stripped.
    without = re.sub(r"[-_,/]", " ", without)
    return re.sub(r"\s+", " ", without).strip().lower()


def select_official_table(page_html: str) -> str:
    """Body of the maintainers' own leaderboard.

    Raises if the id is gone rather than falling back to another table: the two
    it would fall back to are externally reported numbers and a retired
    protocol, and either would look like a routine refresh.
    """
    stripped = _SCRIPT_RE.sub(" ", page_html)
    for match in _TABLE_RE.finditer(stripped):
        opening = stripped[match.start() : match.start(1)]
        if re.search(rf'id\s*=\s*["\']{re.escape(OFFICIAL_TABLE_ID)}["\']', opening):
            return match.group(1)
    raise ValueError(
        f"No <table id={OFFICIAL_TABLE_ID!r}> on {URL} — the page layout changed; "
        "refusing to guess which of its tables is the maintainers' own run."
    )


def parse_rows(table_body: str) -> list[dict]:
    """Rows of the official table as dicts, header-keyed and cleaned.

    The table opens with a grouping row ("Main questions (100)") above the
    column names, so the header is the first row whose cells match the expected
    columns rather than simply the first row.
    """
    header: list[str] = []
    rows: list[dict] = []
    for tr in _TR_RE.findall(table_body):
        cells = [_text(cell) for cell in _CELL_RE.findall(tr)]
        if not cells:
            continue
        if not header:
            candidate = [cell.lower() for cell in cells]
            if tuple(candidate[: len(EXPECTED_HEADER)]) == EXPECTED_HEADER:
                header = candidate
            continue
        values = {header[i]: cells[i] for i in range(min(len(header), len(cells)))}
        raw = values.get("model")
        if not raw:
            continue
        rows.append({"raw": raw, "pass_1": values.get("pass@1", "")})

    if not header:
        raise ValueError(
            f"No header row {EXPECTED_HEADER} in the {OFFICIAL_TABLE_ID!r} table on {URL} — "
            "the columns changed; refusing to read a column whose metric is unknown."
        )
    return rows


def get_scores() -> list[dict]:
    """Return one dict per official-board row.

    Keys: model (normalized base), raw, effort, score (pass@1 % on the 100 main
    questions), sampling ("mean" over 5 samplings or "greedy"), rank (rank
    within the leaderboard, 1 = best).
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    rows = parse_rows(select_official_table(fetch_html()))

    kept: list[dict] = []
    for row in rows:
        match = _SCORE_RE.search(row["pass_1"])
        if not match:
            continue
        _, effort = split_effort(row["raw"])
        kept.append(
            {
                "model": normalize_model(row["raw"]),
                "raw": row["raw"],
                "effort": effort,
                "score": round(float(match.group(1)), 2),
                "sampling": {"●": "mean", "○": "greedy"}.get(match.group(2) or ""),
            }
        )

    if not kept:
        raise ValueError(
            f"No scored rows in the {OFFICIAL_TABLE_ID!r} table on {URL} — the page "
            "layout changed; refusing to report an empty leaderboard."
        )
    print(f"  parsed {len(kept)} leaderboard rows", file=sys.stderr)
    check_percentages(kept, URL)

    kept.sort(key=lambda r: -r["score"])
    for i, entry in enumerate(kept, 1):
        entry["rank"] = i
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch ZeroBench leaderboard scores from zerobench.github.io."
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
        col_widths = [
            max(len("MODEL"), max((len(e["model"]) for e in scores), default=0)),
            6,
            max(len("RAW"), max((len(e["raw"]) for e in scores), default=0)),
        ]
        fmt = f"{{:<{col_widths[0]}}}  {{:>{col_widths[1]}}}  {{:<{col_widths[2]}}}  {{}}"
        print(fmt.format("MODEL", "PASS@1", "RAW", "SAMPLING"))
        for entry in scores:
            print(
                fmt.format(
                    entry["model"],
                    str(entry["score"]),
                    entry["raw"],
                    entry["sampling"] or "",
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
        print(f"error: could not fetch the ZeroBench leaderboard: {exc}", file=sys.stderr)
        raise SystemExit(1)
