#!/usr/bin/env python3
"""Fill llm.json with the source URLs of every page the scrapers in this repo read.

Two destinations:

  * ``doc["sources"]`` — backs the Sources panel in llm.html. Gets one entry per
    scraped page, including the ones that carry no benchmark of their own (the
    Spheron GPU recommender, the OSWorld results workbook).
  * ``doc["benchmarks"][key]["urls"]`` — gets only benchmark-specific pages. An
    aggregator feeding a dozen benchmarks (Artificial Analysis, llm-stats) stays
    in "sources" so its URL is not repeated on every benchmark entry.

URLs are read from the ``fetch_*.py`` constants instead of being duplicated
here, so a scraper repointed at a new leaderboard host surfaces as a missing URL
on the next run rather than going unnoticed.

Default is a dry-run; pass -w/--write to persist changes (same convention as
update.py and prune.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import artificialanalysis
import fetch_deepswe
import fetch_evals_report
import fetch_frontierswe
import fetch_huggingface
import fetch_llmstats
import fetch_osworld
import fetch_spheron
import fetch_swe_atlas
import fetch_swe_marathon
import fetch_swe_rebench

DEFAULT_LLM_JSON = Path(__file__).resolve().parent / "llm.json"
JSON_DUMP_KWARGS = {"indent": 2, "ensure_ascii": False}

# Terminal-Bench 2.1 numbers arrive through the Artificial Analysis API, but the
# leaderboard that publishes them is its own source; "sources" already lists the
# 2.0 page.
TBENCH_2_1_LEADERBOARD = "https://www.tbench.ai/leaderboard/terminal-bench/2.1"

# Scrapers whose data is already represented in "sources" by a human-facing
# equivalent: (what the scraper actually requests, the URL that covers it).
# Reported, never inserted — an API host next to its own leaderboard page would
# be a duplicate entry in the panel.
COVERED_BY = [
    (
        fetch_llmstats.URL,
        "https://llm-stats.com/leaderboards/open-llm-leaderboard",
    ),
    (
        artificialanalysis.API_URL,
        "https://artificialanalysis.ai/leaderboards/models?is_open_weights=open_source",
    ),
    (
        f"{fetch_huggingface.HF_BASE}/<org>/<repo> (per-model cards)",
        "models[].url — llm.html merges those into the Sources panel",
    ),
]


def canonical(url: str) -> str:
    """URL as stored: no query, no fragment, no trailing slash on the path.

    Scrapers request pages with tab/filter queries (``?tab=scores``) that select a
    view rather than identify the page, so they are dropped from the stored form.
    """
    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def match_key(url: str) -> str:
    """Comparison key: canonical form, lowercased host and scheme.

    Query-bearing URLs already in llm.json collapse onto their query-less form,
    which is what keeps a second copy of an existing page from being appended.
    """
    parts = urlsplit(canonical(url))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, "", ""))


def spheron_root() -> str:
    """The GPU recommender root, from the per-model URL template."""
    template = fetch_spheron.BASE_URL
    head, _, _ = template.partition("{model}")
    return head


def build_inventory() -> list[tuple[str, tuple[str, ...]]]:
    """(url, benchmark keys) for every page a scraper reads.

    An empty benchmark tuple means sources-only: the page either carries no
    benchmark score (Spheron) or is not a page worth linking from a benchmark
    entry (the OSWorld .xlsx download; benchmarks.osworld_verified already links
    the site).
    """
    items: list[tuple[str, tuple[str, ...]]] = [
        (fetch_deepswe.URL, ("deepswe",)),
        (fetch_frontierswe.URL, ("frontierswe",)),
        (fetch_swe_rebench.URL, ("swe_rebench",)),
        (fetch_swe_marathon.URL, ("swe_marathon",)),
        (TBENCH_2_1_LEADERBOARD, ("terminal_bench_2_1",)),
        (fetch_osworld.OSWORLD_XLSX_URL, ()),
        (spheron_root(), ()),
    ]
    for slug, key in fetch_evals_report.BENCHMARKS.items():
        items.append((fetch_evals_report.BASE_URL.format(slug=slug), (key,)))
    for track, key in fetch_swe_atlas.TRACKS.items():
        items.append((fetch_swe_atlas.BASE_URL.format(track=track), (key,)))
    return items


def benchmark_urls(entry: dict) -> list[str]:
    """Existing URLs of one benchmark entry, across both supported shapes."""
    urls = entry.get("urls")
    if isinstance(urls, list):
        return [u for u in urls if isinstance(u, str) and u.strip()]
    url = entry.get("url")
    if isinstance(url, str) and url.strip():
        return [url]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(DEFAULT_LLM_JSON),
        help='Path to JSON file to read/update (default: "./llm.json" next to this script)',
    )
    parser.add_argument(
        "--write",
        "-w",
        action="store_true",
        help="Write changes back to the input JSON file (default is dry-run).",
    )
    args = parser.parse_args()

    path = Path(args.json_file)
    doc = json.loads(path.read_text(encoding="utf-8"))

    sources = doc.get("sources")
    if sources is None:
        sources = []
    if not isinstance(sources, list):
        print(f'Error: "sources" in {path} is not a list', file=sys.stderr)
        return 1
    benchmarks = doc.get("benchmarks")
    if not isinstance(benchmarks, dict):
        print(f'Error: "benchmarks" in {path} is missing or not an object', file=sys.stderr)
        return 1

    inventory = build_inventory()

    unknown = sorted(
        {key for _, keys in inventory for key in keys if key not in benchmarks}
    )
    if unknown:
        print(
            "Error: scrapers reference benchmark keys absent from "
            f"{path}: {', '.join(unknown)}",
            file=sys.stderr,
        )
        return 1

    # Missing "sources" entries, deduped against each other as well as the file.
    have_sources = {match_key(u) for u in sources if isinstance(u, str) and u.strip()}
    new_sources: list[str] = []
    for url, _ in inventory:
        stored = canonical(url)
        if match_key(stored) in have_sources:
            continue
        have_sources.add(match_key(stored))
        new_sources.append(stored)

    # Missing benchmark URLs, per entry.
    new_benchmark_urls: dict[str, list[str]] = {}
    for url, keys in inventory:
        stored = canonical(url)
        for key in keys:
            existing = benchmark_urls(benchmarks[key])
            have = {match_key(u) for u in existing}
            have.update(match_key(u) for u in new_benchmark_urls.get(key, []))
            if match_key(stored) in have:
                continue
            new_benchmark_urls.setdefault(key, []).append(stored)

    uncovered = [
        (requested, covering)
        for requested, covering in COVERED_BY
        if covering.startswith("http") and match_key(covering) not in have_sources
    ]

    print(f"{len(inventory)} scraped page(s) in the inventory")
    if new_sources:
        print(f'\nsources — {len(new_sources)} missing:')
        for url in new_sources:
            print(f"  + {url}")
    else:
        print('\nsources — nothing missing')

    if new_benchmark_urls:
        total = sum(len(v) for v in new_benchmark_urls.values())
        print(f"\nbenchmarks[].urls — {total} missing:")
        for key in sorted(new_benchmark_urls):
            for url in new_benchmark_urls[key]:
                print(f"  + {key}: {url}")
    else:
        print("\nbenchmarks[].urls — nothing missing")

    print("\naggregators (sources-only, not inserted):")
    for requested, covering in COVERED_BY:
        print(f"  {requested}\n      covered by {covering}")
    if uncovered:
        print("\nWarning: aggregator page(s) no longer listed in sources:", file=sys.stderr)
        for requested, covering in uncovered:
            print(f"  {covering}  (source for {requested})", file=sys.stderr)

    if not new_sources and not new_benchmark_urls:
        print("\nNothing to do.")
        return 0

    if not args.write:
        print("\ndry-run only, pass --write to persist changes")
        return 0

    doc["sources"] = [*sources, *new_sources]
    for key, urls in new_benchmark_urls.items():
        entry = benchmarks[key]
        merged = [*benchmark_urls(entry), *urls]
        entry.pop("url", None)
        entry["urls"] = merged

    path.write_text(json.dumps(doc, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
