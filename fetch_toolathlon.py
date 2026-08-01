#!/usr/bin/env python3
"""
Fetch Toolathlon-Verified (Pass@1 %) scores from https://toolathlon.xyz

The leaderboard page carries two tables and they are NOT interchangeable:

  * ``leaderboard-current-table``  Toolathlon-Verified, the official series since
                                   2026-06-30. This is the one we read.
  * ``leaderboard-history-table``  An archived snapshot from before Verified,
                                   kept for historical context. Verified revised
                                   evaluation logic on 76 of the 108 tasks, so the
                                   site states outright that the two series are
                                   not directly comparable.

Mixing them is the failure mode to avoid, and it is a real one: llm-stats.com
and some model cards publish a single "Toolathlon" number that is old-series for
some models and Verified for others. This script therefore selects the table by
class and ignores everything else on the page.

Rows carry a model type ("Open-Weights" / "Open-Source" / "Proprietary"),
reported as ``open_weights``, and a "✓ Evaluated by us" badge, reported as
``verified``. Self-reported rows are dropped unless --include-self-reported is
passed, so a lab-submitted number never silently lands next to the maintainers'
own runs; the count dropped is always logged.
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

from _openness import source_type_open


URL = "https://toolathlon.xyz/docs/leaderboard"

# Class marking the Toolathlon-Verified table. The page is authored as MDX, so
# the attribute is `class` in the rendered HTML and `className` in the .md
# mirror; the class token itself is the stable part.
CURRENT_TABLE_CLASS = "leaderboard-current-table"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.DOTALL | re.IGNORECASE)
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
# The vendor logo sits inside the model cell and carries a <title> with the org
# name, which would otherwise be read as part of the model name.
_SVG_RE = re.compile(r"<svg\b.*?</svg>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
# "76.5 ± 1.9" / "55.6" / "—"
_SCORE_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")
_VERIFIED_RE = re.compile(r"verified-badge", re.IGNORECASE)


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


def _model_name(cell_markup: str) -> str:
    """Model label with the org logo and the verified tick removed."""
    return _text(_SVG_RE.sub(" ", cell_markup)).rstrip("✓").strip()


def select_current_table(page_html: str) -> str:
    """Body of the Toolathlon-Verified table.

    Raises if the class is gone rather than falling back to another table: a
    silent switch to the archived series would look like every model suddenly
    regressing.
    """
    for match in _TABLE_RE.finditer(page_html):
        opening = page_html[match.start() : match.start(1)]
        if CURRENT_TABLE_CLASS in opening:
            return match.group(1)
    raise ValueError(
        f"No <table> with class {CURRENT_TABLE_CLASS!r} on {URL} — the page layout "
        "changed; refusing to guess which series a table belongs to."
    )


def parse_rows(table_body: str) -> list[dict]:
    """Rows of the Verified table as dicts, header-keyed and cleaned."""
    header: list[str] = []
    rows: list[dict] = []
    for tr in _TR_RE.findall(table_body):
        cells = _CELL_RE.findall(tr)
        if not cells:
            continue
        if not header:
            header = [_text(c).lower().lstrip("# ").strip() for c in cells]
            continue
        values = {header[i]: cells[i] for i in range(min(len(header), len(cells)))}
        raw = values.get("model")
        if raw is None:
            continue
        rows.append(
            {
                "raw": _model_name(raw),
                "model_type": _text(values.get("type", "")),
                "agent": _text(values.get("agent", "")),
                "date": _text(values.get("date", "")) or None,
                "pass_1": _text(values.get("pass@1", "")),
                "pass_3": _text(values.get("pass@3", "")),
                "verified": bool(_VERIFIED_RE.search(raw)),
            }
        )
    return rows


def get_scores(include_self_reported: bool = False) -> list[dict]:
    """Return a list of score dicts for the Toolathlon-Verified leaderboard.

    Keys: model, raw, open_weights, verified, agent, score (Pass@1 %), date,
    rank (rank within the leaderboard, 1 = best).
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    rows = parse_rows(select_current_table(fetch_html()))

    kept: list[dict] = []
    dropped = 0
    for row in rows:
        if not include_self_reported and not row["verified"]:
            dropped += 1
            continue
        match = _SCORE_RE.search(row["pass_1"])
        if not match:
            continue
        kept.append(
            {
                "model": row["raw"],
                "raw": row["raw"],
                "open_weights": source_type_open(row["model_type"]),
                "verified": row["verified"],
                "agent": row["agent"] or None,
                "score": round(float(match.group(1)), 2),
                "date": row["date"],
            }
        )
    print(
        f"  parsed {len(kept)} Toolathlon-Verified rows"
        + (f" ({dropped} self-reported dropped)" if dropped else ""),
        file=sys.stderr,
    )

    kept.sort(key=lambda r: -r["score"])
    for i, entry in enumerate(kept, 1):
        entry["rank"] = i
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Toolathlon-Verified leaderboard scores."
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--include-self-reported",
        action="store_true",
        help='Also keep rows without the "Evaluated by us" badge.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = get_scores(args.include_self_reported)

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for name in sorted({entry["model"] for entry in scores}):
            print(name)
    else:
        width = max([len("MODEL")] + [len(e["model"]) for e in scores])
        fmt = f"{{:<{width}}}  {{:>6}}  {{:<12}}  {{:<8}}"
        print(fmt.format("MODEL", "PASS@1", "WEIGHTS", "VERIFIED"))
        for entry in scores:
            weights = {True: "open", False: "proprietary", None: "unknown"}[
                entry["open_weights"]
            ]
            print(
                fmt.format(
                    entry["model"],
                    str(entry["score"]),
                    weights,
                    "yes" if entry["verified"] else "self",
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
        print(f"error: could not fetch the Toolathlon leaderboard: {exc}", file=sys.stderr)
        raise SystemExit(1)
