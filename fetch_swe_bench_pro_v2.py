#!/usr/bin/env python3
"""
Fetch SWE-Bench Pro V2 (Pass@1 %) scores from the Scale Labs leaderboard.

SWE-Bench Pro V2 (Scale AI, co-developed with Reflection; released 2026-09-22)
is a refreshed public split of SWE-Bench Pro, not a new run of it: 642 tasks
across the same 11 repositories, after 89 of the 731 v1 public instances were
dropped as invalid, with every instruction, verifier and image rebuilt, and a
network-locked protocol in which the sandbox reaches the model endpoint and
nothing else. Its scores sit on another scale from v1's -- the board's top runs
are at 95-99 where Scale's v1 public board tops out at 61.5 -- so it is its own
column, never `swe_bench_pro`. See https://labs.scale.com/blog/swe-bench-pro-v2.

https://labs.scale.com/leaderboard/swe_bench_pro_public_v2 is Scale's own runs
and publishes two tables on one page, as variants of one board:
  - full  SWE-Bench Pro V2 Full, all 642 tasks  -> swe_bench_pro_2_0
  - hard  SWE-Bench Pro V2 HARD, a 51-task subset of the hardest
          -> swe_bench_pro_hard_2_0
The columns are spelled the way _revisions.py spells a revision -- V2 is 2.0,
as FrontierSWE's V2 is frontierswe_2_0 -- with Hard as a subset base of its
own, as FrontierCode's Extended is. `swe_bench_pro` itself stays the unversioned
column of lab self-reports of the original benchmark.

The page is the same Next.js App Router app as the SWE Atlas boards, and its
labels have the same shape -- "Opus 5 (Claude Code) xhigh", "GPT-5.6-sol
(Codex) xhigh" -- so the label normalizer is fetch_swe_atlas.py's: harness
parenthetical and reasoning effort stripped, separators unified, lowercased.
The two tables spell some names differently ("GPT-6-Astra" in Full, "GPT-6
Astra " in Hard); both normalize to the one key.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error

from _fetch_checks import check_percentages
from _scale_labs import extract_board_rows, fetch_board_rows
from fetch_swe_atlas import fetch_html, normalize_model, split_harness
import _source_pages


URL = "https://labs.scale.com/leaderboard/swe_bench_pro_public_v2"

# The route slug the page's payload has to carry to be this board.
BOARD_SLUG = "swe_bench_pro_public_v2"

# Variant key on the board -> llm.json benchmark key.
VARIANTS: dict[str, str] = {
    "full": "swe_bench_pro_2_0",
    "hard": "swe_bench_pro_hard_2_0",
}


def extract_rows(html: str, variant: str) -> list[dict]:
    """Rows of one variant of the board.

    Raises when the page is another board, names no such variant, or carries
    more than one table under it -- see _scale_labs.
    """
    return extract_board_rows(html, BOARD_SLUG, URL, variant)


def get_scores(variants: list[str], include_deprecated: bool = False) -> list[dict]:
    """Return a flat list of row dicts across the requested variants.

    Keys: variant, key, model (normalized base), raw, harness, score, ci,
    company, rank (rank within the variant, 1 = best).
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    # One page carries every variant: the first is read with fetch_board_rows'
    # re-fetch of an empty page, the rest from that same copy, so the columns
    # always come from one snapshot of the board.
    pages: list[str] = []

    def fetch(url: str) -> str:
        pages.append(fetch_html(url))
        return pages[-1]

    first = fetch_board_rows(fetch, URL, BOARD_SLUG, variant=variants[0])

    results: list[dict] = []
    for variant in variants:
        rows = first if variant == variants[0] else extract_rows(pages[-1], variant)
        key = VARIANTS[variant]
        kept: list[dict] = []
        dropped = 0
        for obj in rows:
            if obj.get("deprecated") and not include_deprecated:
                dropped += 1
                continue
            raw = obj["model"]
            _, harness = split_harness(raw)
            kept.append(
                {
                    "variant": variant,
                    "key": key,
                    "model": normalize_model(raw),
                    "raw": raw.strip(),
                    "harness": harness,
                    "score": round(float(obj["score"]), 2),
                    "ci": obj.get("confidenceInterval_upper"),
                    "company": obj.get("company"),
                }
            )
        print(
            f"  parsed {len(kept)} rows for {variant}"
            + (f" ({dropped} deprecated dropped)" if dropped else ""),
            file=sys.stderr,
        )
        check_percentages(kept, f"{URL} ({variant})")
        kept.sort(key=lambda r: -r["score"])
        for i, entry in enumerate(kept, 1):
            entry["rank"] = i
        results.extend(kept)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch SWE-Bench Pro V2 leaderboard scores from Scale Labs."
    )
    parser.add_argument(
        "--variant",
        choices=[*VARIANTS.keys(), "all"],
        default="all",
        help="Which table of the board to fetch (default: all).",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "names", "pages"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--include-deprecated",
        action="store_true",
        help="Also keep rows Scale has flagged as deprecated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]
    scores = get_scores(variants, args.include_deprecated)

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "pages":
        # Both variants are one page, so every name links to it.
        _source_pages.emit(scores, lambda e: URL)
    elif args.format == "names":
        for name in sorted({entry["model"] for entry in scores}):
            print(name)
    else:
        col_widths = [
            max(len("MODEL"), max((len(e["model"]) for e in scores), default=0)),
            6,
            7,
            max(len("RAW"), max((len(e["raw"]) for e in scores), default=0)),
        ]
        fmt = f"{{:<{col_widths[0]}}}  {{:>{col_widths[1]}}}  {{:<{col_widths[2]}}}  {{:<{col_widths[3]}}}"
        print(fmt.format("MODEL", "PASS@1", "VARIANT", "RAW"))
        for entry in scores:
            print(fmt.format(entry["model"], str(entry["score"]), entry["variant"], entry["raw"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch the SWE-Bench Pro V2 leaderboard: {exc}", file=sys.stderr)
        raise SystemExit(1)
