#!/usr/bin/env python3
"""
Fetch benchmark scores from evals.report (inductive.ml's benchmark aggregator).

evals.report keeps per-benchmark score tables at
https://evals.report/benchmarks/<slug>?tab=scores with columns
Model | Lab | Score | Source model | Status | Date. The pages are fully
server-side rendered, so plain HTML parsing suffices (no flight payload).

Rows carry a provenance Status (Official = published by the benchmark/lab,
Verified = independently verified, Unverified = unconfirmed reports). Only
Official/Verified rows are reported by default; pass --include-unverified to
keep the rest.

Open-weights models are labelled with an " Open" suffix on the model name
("Kimi K2.6 Open"); the suffix is stripped for the reported `model` so a
single name->slug mapping works, `raw` keeps the original label, and
`open_weights` reports the label as a boolean.

Some rows are harness+model composites ("Terminus 2 + GLM 5.1", Lab = "Agent
systems") whose label never carries the " Open" suffix even when the model is
open, so those report open_weights=None instead of False.

BENCHMARKS maps the evals.report benchmark slug to our llm.json benchmark
key; add entries there to scrape more benchmarks.
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


BASE_URL = "https://evals.report/benchmarks/{slug}?tab=scores"

# evals.report benchmark slug -> llm.json benchmark key.
BENCHMARKS: dict[str, str] = {
    # evals.report publishes no revision label, so a table only feeds a
    # versioned column when its numbers identify one. FrontierCode's does:
    # every row it carries matches Cognition's 1.1 block (its leader is
    # Claude Fable 5 at 53.5, the 1.1 number), so it fills gaps in
    # frontiercode_1_1 and nothing else.
    "frontiercode": "frontiercode_1_1",
    # "swe-marathon" is deliberately absent. Its table is a mix: seven rows are
    # the v1.0 archive verbatim (Grok 4.5 29.0, Opus 4.8 26.0, GLM-5.2 13.0
    # ...) sitting beside a Kimi K3 at 42.0 that is on neither published board,
    # with no column saying which revision any row was measured under. That is
    # the mix swe_marathon_1_0/_1_1 exist to separate, so it is not ingested --
    # the rule the README's version traps apply to a bare "BFCL".
    # The only source we scrape for SWE-bench Multimodal: the benchmark's own
    # leaderboard lists no open-weight system, and AA does not run it at all.
    "swe-bench-multimodal": "swe_bench_multimodal",
    # evals.report's "bfcl" table is the V4 series: its leader matches the
    # Overall Acc column of BFCL's own data_overall.csv (see fetch_bfcl.py).
    "bfcl": "bfcl_v4",
    "mcp-atlas": "mcp_atlas",
    "ifbench": "ifbench",
    # Both also arrive as Hugging Face card self-reports, which the HF ingest
    # writes into nulls only; this source runs after it and does overwrite, so
    # an Official or Verified run displaces a self-report wherever one exists.
    "swe-bench-multilingual": "swe_bench_multilingual",
    "mmlu-pro": "mmlu_pro",
    # The two vision columns. MathVista's own leaderboard stopped at the 2024
    # field and ZeroBench's official board is almost all closed-weight, so
    # neither first-party page is scraped: evals.report leads both and does
    # overwrite, with the Hugging Face cards filling gaps ahead of it -- the
    # same relation swe-bench-multimodal stands in. Its "mathvista" table is
    # the 1,000-example testmini split, which is what mathvista_mini tracks;
    # every open-weight row in its "zerobench" table is pass@1, the metric the
    # column carries, though the table also holds closed-model pass@5 rows.
    "zerobench": "zerobench",
    "mathvista": "mathvista_mini",
}

TRUSTED_STATUSES = {"official", "verified"}

# Lab value evals.report uses for harness+model composite rows.
AGENT_SYSTEMS_LAB = "agent systems"

_OPEN_SUFFIX_RE = re.compile(r"\s+Open$")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_SCORE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%?")


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


def _cell_text(cell_html: str) -> str:
    text = _TAG_RE.sub(" ", html.unescape(cell_html))
    return re.sub(r"\s+", " ", text).strip()


def extract_rows(page_html: str) -> list[dict]:
    """Extract score-table rows as raw column dicts (header-name keyed)."""
    start = page_html.find("score-table")
    if start == -1:
        return []
    end = page_html.find("</table>", start)
    segment = page_html[start : end if end != -1 else len(page_html)]

    header: list[str] = []
    rows: list[dict] = []
    for tr in _TR_RE.findall(segment):
        cells = [_cell_text(c) for c in _CELL_RE.findall(tr)]
        if not cells:
            continue
        if not header:
            # First row is the header; normalize "Score ↓" -> "score".
            header = [re.sub(r"[^a-z ]", "", c.lower()).strip() for c in cells]
            continue
        row = {header[i]: cells[i] for i in range(min(len(header), len(cells)))}
        if row.get("model"):
            rows.append(row)
    return rows


def _open_weights(raw: str, lab: str | None) -> bool | None:
    """Read the " Open" label, or None when the row is a harness composite."""
    if _OPEN_SUFFIX_RE.search(raw):
        return True
    if " + " in raw or (lab or "").strip().lower() == AGENT_SYSTEMS_LAB:
        return None
    return False


def get_scores(slugs: list[str], include_unverified: bool) -> list[dict]:
    """Return a flat list of score dicts across the requested benchmarks.

    Keys: benchmark, key, model (raw minus " Open"), raw, lab, open_weights,
    score, status, date, rank (rank within the benchmark, 1 = best).
    """
    results: list[dict] = []
    for slug in slugs:
        key = BENCHMARKS[slug]
        url = BASE_URL.format(slug=slug)
        print(f"Fetching {url} ...", file=sys.stderr)
        raw_rows = extract_rows(fetch_html(url))
        kept: list[dict] = []
        dropped = 0
        for row in raw_rows:
            status = row.get("status", "")
            if not include_unverified and status.lower() not in TRUSTED_STATUSES:
                dropped += 1
                continue
            match = _SCORE_RE.match(row.get("score", ""))
            if not match:
                continue
            raw = row["model"]
            model = _OPEN_SUFFIX_RE.sub("", raw)
            kept.append(
                {
                    "benchmark": slug,
                    "key": key,
                    "model": model,
                    "raw": raw,
                    "lab": row.get("lab"),
                    "open_weights": _open_weights(raw, row.get("lab")),
                    "score": round(float(match.group(1)), 2),
                    "status": status,
                    "date": row.get("date") or None,
                }
            )
        print(
            f"  parsed {len(kept)} rows for {slug}"
            + (f" ({dropped} untrusted dropped)" if dropped else ""),
            file=sys.stderr,
        )
        kept.sort(key=lambda r: -r["score"])
        for i, r in enumerate(kept, 1):
            r["rank"] = i
        results.extend(kept)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch evals.report benchmark scores.")
    parser.add_argument(
        "--benchmark",
        choices=[*BENCHMARKS.keys(), "all"],
        default="all",
        help="Which benchmark to fetch (default: all).",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="Also keep rows whose Status is not Official/Verified.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    slugs = list(BENCHMARKS.keys()) if args.benchmark == "all" else [args.benchmark]
    scores = get_scores(slugs, args.include_unverified)

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for name in sorted({entry["model"] for entry in scores}):
            print(name)
    else:
        col_widths = [
            max(len("MODEL"), max((len(e["model"]) for e in scores), default=0)),
            6,
            max(len("BENCHMARK"), max((len(e["benchmark"]) for e in scores), default=0)),
            max(len("STATUS"), max((len(e["status"]) for e in scores), default=0)),
        ]
        fmt = f"{{:<{col_widths[0]}}}  {{:>{col_widths[1]}}}  {{:<{col_widths[2]}}}  {{:<{col_widths[3]}}}"
        print(fmt.format("MODEL", "SCORE", "BENCHMARK", "STATUS"))
        for entry in scores:
            print(fmt.format(entry["model"], str(entry["score"]), entry["benchmark"], entry["status"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch evals.report: {exc}", file=sys.stderr)
        raise SystemExit(1)
