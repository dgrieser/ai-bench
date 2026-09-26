#!/usr/bin/env python3
"""
Fetch HLE-Diamond scores from https://lastexam.ai/blog/hle-diamond

HLE-Diamond is the Center for AI Safety and Scale AI's refined subset of
Humanity's Last Exam: 1,000 questions, 500 reasoning and 500 knowledge, kept
after a year of cleaning so that each one is answerable closed-book. It is a
different question set from the 2,500-question HLE (and from the 2,158-question
text-only subset Artificial Analysis runs), so it gets a column of its own,
``hle_diamond``, next to ``hle`` rather than in it.

The maintainers publish their own runs on the announcement page, and that is
the only first-party board -- lastexam.ai's /leaderboard is a sign-in
dashboard. The post can show three views of its results, and only one is the
column:

  * **highest effort, no tools** -- "Models are evaluated using highest
    reasoning effort available": *max* for GPT, Claude and Muse, *xhigh* for
    Grok, *high* for Gemini, Kimi and GLM. This is the column, because every
    other column in the table carries a model at its highest setting too. It is
    the post's own results table, ``headers: ["Model", "Accuracy"]``, which the
    server renders into the page's React flight data, and that is what this
    reads.
  * **reasoning high** -- the chart's default toggle: every model at *high*,
    up to six points lower (DeepSeek V4 Pro 19.1 -> 13.4, GPT-6 Astra 66.2 ->
    60.6). It lives only in the page's JS chunk and is not read.
  * **with tools** (web search + code execution) -- worth +19 to +31 points on
    the eight models the page runs both ways. A different measurement, so
    never read, the same line the ``hle`` column draws.

The post's second table splits each score into its reasoning and knowledge
halves. Its rows are carried beside the score only where its overall agrees
with the headline table: DeepSeek V4 Pro's row there is still its reasoning-high
run (13.4 against 19.1), so a split is decoration, never the score.

A row the chart marks ``textOnly`` (GLM 5.3 today, "* Text-only subset") was
run on the questions without images only, because the model takes no images.
It is kept -- for a text-only model that subset is the whole of the benchmark
it can sit -- and reported with ``text_only: true`` so the difference is on
record. That flag lives in the chart's model table in the post's JS chunk, so
the chunk is read for it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from _fetch_checks import check_percentages, require_rows


KEY = "hle_diamond"
SITE_URL = "https://lastexam.ai"
# The announcement post: what a score is credited to and what a reader opens.
URL = f"{SITE_URL}/blog/hle-diamond"
DATASET_URL = "https://huggingface.co/datasets/cais/hle-diamond"
# The headline results table and its reasoning/knowledge split, by header.
RESULTS_HEADERS = ["Model", "Accuracy"]
SPLIT_HEADERS = ["Model", "Reasoning", "Knowledge", "HLE-Diamond"]
# The post's own client chunk; its hash changes with every build, so it is
# found in the page rather than pinned.
_CHUNK_RE = re.compile(r'/_next/static/chunks/app/blog/hle-diamond/page-[0-9a-f]+\.js')
_FLIGHT_RE = re.compile(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)', re.DOTALL)
_TABLE_RE = re.compile(
    r'\{\s*"headers"\s*:\s*(\[[^\]]*\])\s*,\s*"rows"\s*:\s*(\[\[.*?\]\])\s*\}', re.DOTALL
)
# JSON.parse('...') with a single-quoted JS string literal inside.
_JSON_PARSE_RE = re.compile(r"JSON\.parse\('((?:[^'\\]|\\.)*)'\)")
_JS_ESCAPE_RE = re.compile(r"\\(u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|.)", re.DOTALL)
_JS_SIMPLE_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}
_PERCENT_RE = re.compile(r"^\s*(\d{1,3}(?:\.\d+)?)\s*%\s*$")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_text(url: str, retries: int = 3, delay: float = 2.0) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
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


def normalize_model(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip()


def flight_data(html: str) -> str:
    """The page's React flight chunks, decoded and joined."""
    return "".join(json.loads(f'"{chunk}"') for chunk in _FLIGHT_RE.findall(html))


def tables(html: str) -> list[tuple[list, list]]:
    """Every ``{"headers": [...], "rows": [[...]]}`` table the post renders."""
    out: list[tuple[list, list]] = []
    for match in _TABLE_RE.finditer(flight_data(html)):
        try:
            out.append((json.loads(match.group(1)), json.loads(match.group(2))))
        except ValueError:
            continue
    return out


def _one_table(found: list[tuple[list, list]], headers: list[str], required: bool) -> list | None:
    rows = [r for h, r in found if h == headers]
    if len(rows) > 1 or (required and not rows):
        raise ValueError(
            f"expected one {headers} table in {URL}, found {len(rows)} -- "
            "the post's layout changed; refusing to guess."
        )
    return rows[0] if rows else None


def percent(cell: Any) -> float | None:
    match = _PERCENT_RE.match(cell) if isinstance(cell, str) else None
    return float(match.group(1)) if match else None


def chunk_url(html: str) -> str:
    """The post's client chunk, which carries the chart's model table."""
    match = _CHUNK_RE.search(html)
    if not match:
        raise ValueError(f"{URL} no longer links its page chunk -- the layout changed.")
    return urllib.parse.urljoin(SITE_URL, match.group(0))


def _unescape_js(literal: str) -> str:
    def repl(match: re.Match) -> str:
        esc = match.group(1)
        if esc[0] in "ux" and len(esc) > 1:
            return chr(int(esc[1:], 16))
        return _JS_SIMPLE_ESCAPES.get(esc, esc)

    return _JS_ESCAPE_RE.sub(repl, literal)


def json_literals(js: str) -> list[Any]:
    """Every JSON.parse('...') literal in the chunk that decodes."""
    out: list[Any] = []
    for match in _JSON_PARSE_RE.finditer(js):
        try:
            out.append(json.loads(_unescape_js(match.group(1))))
        except ValueError:
            continue
    return out


def model_table(js: str) -> dict:
    """The chart's per-model display table, which is where textOnly is set."""
    found = [
        v for v in json_literals(js)
        if isinstance(v, dict) and v and all(isinstance(m, dict) and "logo" in m for m in v.values())
    ]
    if len(found) != 1:
        raise ValueError(
            f"expected one model table in {URL}'s chunk, found {len(found)} -- "
            "the chart's data changed; refusing to guess."
        )
    return found[0]


def parse_rows(html: str, js: str) -> list[dict]:
    found = tables(html)
    results = _one_table(found, RESULTS_HEADERS, required=True)
    split_rows = _one_table(found, SPLIT_HEADERS, required=False) or []
    models = model_table(js)

    split: dict[str, tuple[float | None, ...]] = {}
    for row in split_rows:
        if isinstance(row, list) and len(row) == 4 and isinstance(row[0], str):
            split[row[0]] = tuple(percent(cell) for cell in row[1:])

    rows: list[dict] = []
    for row in results:
        if not (isinstance(row, list) and len(row) == 2 and isinstance(row[0], str)):
            continue
        raw = row[0]
        score = percent(row[1])
        if score is None:
            continue
        text_only = bool((models.get(raw) or {}).get("textOnly"))
        reasoning, knowledge, overall = split.get(raw, (None, None, None))
        if overall != score:
            reasoning = knowledge = None
        rows.append(
            {
                "model": normalize_model(raw),
                "raw": raw,
                "reasoning": reasoning,
                "knowledge": knowledge,
                "text_only": text_only,
                "score": score,
            }
        )
    return rows


def get_scores(fetch=fetch_text) -> list[dict]:
    """One dict per model in the post's results table (highest effort, no tools).

    Keys: model, raw, reasoning, knowledge (None where the split table does not
    agree with the headline), text_only (run on the text-only subset), score
    (overall %), rank.
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    html = fetch(URL)
    js_url = chunk_url(html)
    print(f"Fetching {js_url} ...", file=sys.stderr)
    rows = parse_rows(html, fetch(js_url))
    require_rows(rows, URL)
    check_percentages(rows, URL)
    print(f"  parsed {len(rows)} leaderboard rows", file=sys.stderr)
    rows.sort(key=lambda r: -r["score"])
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch HLE-Diamond scores (highest reasoning effort, no tools) from lastexam.ai."
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
        def cell(value: float | None) -> str:
            return "" if value is None else str(value)

        width = max((len(e["model"]) for e in scores), default=5)
        print(f"{'MODEL':<{width}}  {'SCORE':>6}  {'REAS':>6}  {'KNOW':>6}  SUBSET")
        for e in scores:
            print(
                f"{e['model']:<{width}}  {e['score']:>6}  "
                f"{cell(e['reasoning']):>6}  {cell(e['knowledge']):>6}  "
                f"{'text-only' if e['text_only'] else ''}"
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch the HLE-Diamond results: {exc}", file=sys.stderr)
        raise SystemExit(1)
