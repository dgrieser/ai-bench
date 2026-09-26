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
dashboard. The page is a Next.js build whose results chart is a client
component, and the chart's data is compiled into that page's own JS chunk as
``JSON.parse('...')`` literals. This reads them from there, because the
server-rendered table only carries one of the two views the chart can show:

  * **reasoning high** -- the chart's default: every model run at reasoning
    effort *high*, no tools. This is the column. The setting is the same for
    every row, and it is the view the page's own with-tools comparison is
    drawn against ("Without tools" there is these numbers).
  * **reasoning max** -- "the highest reasoning effort available", which is
    *max* for GPT, Claude and Muse, *xhigh* for Grok and still *high* for
    Gemini, Kimi and GLM. Up to six points higher (DeepSeek V4 Pro 13.4 -> 19.1,
    GPT-6 Astra 60.6 -> 66.2), and a different setting per vendor, so it is not
    read.
  * **with tools** (web search + code execution) -- worth +19 to +31 points on
    the eight models the page runs both ways. A different measurement, so
    never read, the same line the ``hle`` column draws.

Each score triple is ``[overall, reasoning, knowledge]``; the overall is the
mean of the two 500-question halves, and a triple where it is not stops the
read -- that is how a reordering of the literal would show. A row the page
marks ``textOnly`` (GLM 5.3 today, "* Text-only subset") was run on the
questions without images only, a different question set again, and is dropped.
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
# The post's own client chunk; its hash changes with every build, so it is
# found in the page rather than pinned.
_CHUNK_RE = re.compile(r'/_next/static/chunks/app/blog/hle-diamond/page-[0-9a-f]+\.js')
# JSON.parse('...') with a single-quoted JS string literal inside.
_JSON_PARSE_RE = re.compile(r"JSON\.parse\('((?:[^'\\]|\\.)*)'\)")
_JS_ESCAPE_RE = re.compile(r"\\(u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|.)", re.DOTALL)
_JS_SIMPLE_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}
# overall is the mean of the reasoning and knowledge halves; the page rounds
# each of the three to one decimal on its own.
MEAN_TOLERANCE = 0.1

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


def chunk_url(html: str) -> str:
    """The post's client chunk, which carries the chart's data."""
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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_score_table(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(v, list) and len(v) == 3 and all(_is_number(x) for x in v)
            for v in value.values()
        )
    )


def _is_model_table(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(isinstance(v, dict) and "logo" in v for v in value.values())
    )


def score_and_model_tables(literals: list[Any]) -> tuple[dict, dict]:
    """The reasoning-high score table and the per-model display table.

    Exactly one of each, or the read stops: two candidate score tables would
    mean the chunk now carries a second view, and guessing which is which is
    how a max-effort number would land.
    """
    scores = [v for v in literals if _is_score_table(v)]
    models = [v for v in literals if _is_model_table(v)]
    if len(scores) != 1:
        raise ValueError(
            f"expected one [overall, reasoning, knowledge] score table in {URL}'s chunk, "
            f"found {len(scores)} -- the chart's data changed; refusing to guess."
        )
    if len(models) != 1:
        raise ValueError(
            f"expected one model table in {URL}'s chunk, found {len(models)} -- "
            "the chart's data changed; refusing to guess."
        )
    return scores[0], models[0]


def parse_rows(js: str) -> list[dict]:
    scores, models = score_and_model_tables(json_literals(js))
    rows: list[dict] = []
    for raw, triple in scores.items():
        if not isinstance(raw, str):
            continue
        # First, because the halves of the text-only subset are not 500 each,
        # so its overall is not their mean.
        if (models.get(raw) or {}).get("textOnly"):
            print(f"  dropped {raw}: run on the text-only subset", file=sys.stderr)
            continue
        overall, reasoning, knowledge = (float(x) for x in triple)
        if abs(overall - (reasoning + knowledge) / 2) > MEAN_TOLERANCE:
            raise ValueError(
                f"{raw!r}: overall {overall} is not the mean of reasoning {reasoning} "
                f"and knowledge {knowledge} -- the triple's order changed."
            )
        rows.append(
            {
                "model": normalize_model(raw),
                "raw": raw,
                "reasoning": reasoning,
                "knowledge": knowledge,
                "score": overall,
            }
        )
    return rows


def get_scores(fetch=fetch_text) -> list[dict]:
    """One dict per model on the reasoning-high, no-tools chart.

    Keys: model, raw, reasoning, knowledge, score (overall %), rank.
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    js_url = chunk_url(fetch(URL))
    print(f"Fetching {js_url} ...", file=sys.stderr)
    rows = parse_rows(fetch(js_url))
    require_rows(rows, URL)
    check_percentages(rows, URL)
    print(f"  parsed {len(rows)} leaderboard rows", file=sys.stderr)
    rows.sort(key=lambda r: -r["score"])
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch HLE-Diamond scores (reasoning high, no tools) from lastexam.ai."
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
        print(f"{'MODEL':<{width}}  {'SCORE':>6}  {'REAS':>6}  {'KNOW':>6}")
        for e in scores:
            print(
                f"{e['model']:<{width}}  {e['score']:>6}  {e['reasoning']:>6}  {e['knowledge']:>6}"
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
