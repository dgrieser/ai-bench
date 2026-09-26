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

import _history
import artificialanalysis
import fetch_aa_coding_agents
import fetch_agents_last_exam
import fetch_bfcl
import fetch_cybergym
import fetch_datacurve
import fetch_deepswe
import fetch_epoch
import fetch_evals_report
import fetch_frontiercode
import fetch_frontierswe
import fetch_huggingface
import fetch_llmstats
import fetch_mcp_atlas
import fetch_openrouter
import fetch_osworld
import fetch_programbench
import fetch_real_swe
import fetch_sec_bench
import fetch_spheron
import fetch_swe_atlas
import fetch_swe_marathon
import fetch_tbench
import fetch_toolathlon
import fetch_vals

DEFAULT_LLM_JSON = Path(__file__).resolve().parent / "llm.json"
JSON_DUMP_KWARGS = {"indent": 2, "ensure_ascii": False}

# Scrapers whose data is already represented in "sources" by a human-facing
# equivalent: (what the scraper actually requests, the URL that covers it).
# Reported, never inserted — an API host next to its own leaderboard page would
# be a duplicate entry in the panel.
COVERED_BY = [
    (
        fetch_llmstats.URL,
        fetch_llmstats.LEADERBOARD_URL,
    ),
    (
        artificialanalysis.MODELS_URL,
        "https://artificialanalysis.ai/leaderboards/models?is_open_weights=open_source",
    ),
    (
        f"{fetch_huggingface.HF_BASE}/<org>/<repo> (per-model cards)",
        "models[].url — llm.html merges those into the Sources panel",
    ),
    (
        fetch_frontiercode.URL,
        fetch_frontiercode.LEADERBOARD_URL,
    ),
    (
        f"{fetch_datacurve.URL} (versioned artifact)",
        fetch_datacurve.SITE_URL,
    ),
    (
        fetch_tbench.URL,
        fetch_tbench.LEADERBOARD_URL,
    ),
    (
        fetch_agents_last_exam.URL,
        fetch_agents_last_exam.LEADERBOARD_URL,
    ),
    (
        fetch_osworld.OSWORLD_V2_JSON_URL,
        fetch_osworld.OSWORLD_V2_SITE_URL,
    ),
    (
        f"{fetch_openrouter.MODEL_PAGE_URL.format(model='<author>/<model>')} (per-model pages)",
        fetch_openrouter.MODELS_PAGE_URL,
    ),
    (
        fetch_openrouter.MODELS_API_URL,
        fetch_openrouter.MODELS_PAGE_URL,
    ),
    (
        f"{fetch_cybergym.DATA_URL} (one file per board)",
        fetch_cybergym.source_url(fetch_cybergym.CYBERGYM_KEY),
    ),
    (
        fetch_sec_bench.DATA_URL,
        fetch_sec_bench.URL,
    ),
    (
        f"{fetch_epoch.DATA_URL} (one archive, every benchmark)",
        fetch_epoch.HUB_URL,
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
        # Read off the scraper's own dataset map rather than repeated here: AA
        # re-cuts the index on new dataset revisions, and a list spelled out
        # again is a list that goes stale without failing.
        (
            fetch_aa_coding_agents.URL,
            tuple(sorted(set(fetch_aa_coding_agents.DATASETS.values()))),
        ),
        # Both DeepSWE readers can write either revision's column: benchlm
        # mirrors whichever artifact it names, and Datacurve's own board
        # publishes both.
        (fetch_deepswe.URL, ("deepswe_1_0", "deepswe_1_1")),
        (fetch_datacurve.SITE_URL, ("deepswe_1_0", "deepswe_1_1")),
        # FrontierSWE keeps its revisions on two pages rather than behind one
        # toggle, so each board covers its own column.
        (fetch_frontierswe.URL, ("frontierswe_2_0",)),
        (fetch_frontierswe.V1_URL, ("frontierswe_1_0",)),
        # One page carries both revisions and both task subsets: the board
        # toggles between Main and Extended the same way it toggles between 1.1
        # and 1.0.
        (
            fetch_frontiercode.LEADERBOARD_URL,
            ("frontiercode_1_0", "frontiercode_1_1", "frontiercode_extended_1_1"),
        ),
        (fetch_swe_marathon.URL, ("swe_marathon_1_0", "swe_marathon_1_1")),
        (fetch_real_swe.URL, ("real_swe",)),
        (fetch_programbench.LEADERBOARD_URL, ("programbench_almost",)),
        (fetch_toolathlon.URL, ("toolathlon",)),
        (fetch_mcp_atlas.URL, ("mcp_atlas",)),
        (fetch_bfcl.LEADERBOARD_URL, ("bfcl_v4",)),
        # One page per Terminal-Bench revision, each covering its own column.
        *((fetch_tbench.board_url(key), (key,)) for key in fetch_tbench.BOARDS),
        (fetch_agents_last_exam.LEADERBOARD_URL, ("agents_last_exam",)),
        # cybergym.io's two boards, one page each; the JSON files they render
        # from are not pages.
        *((fetch_cybergym.source_url(key), (key,)) for key in fetch_cybergym.KEYS),
        # The snapshot the column is pinned to, not the site root (the newer one).
        (fetch_sec_bench.URL, (fetch_sec_bench.KEY,)),
        (fetch_osworld.OSWORLD_XLSX_URL, ()),
        # Every tracked OSWorld 2.0 release comes off the one 2.0 site, which a
        # reader opens; the JSON file it renders from is not a page.
        (
            fetch_osworld.OSWORLD_V2_SITE_URL,
            tuple(key for key in fetch_osworld.KEYS if key != fetch_osworld.VERIFIED_KEY),
        ),
        (spheron_root(), ()),
        # OpenRouter's scores live on one page per model, each credited in
        # models[].scores_source; the catalogue is what the panel links. It
        # carries no score itself, so it stays off benchmarks.gpqa_diamond.
        (fetch_openrouter.MODELS_PAGE_URL, ()),
        # Epoch AI's hub: the index is what the panel links, and each page read
        # credits the scores it carries. A page with a fixed column covers it;
        # a mirror's page covers whichever revision it names on the day, which
        # a static list cannot know, so those stay sources-only.
        (fetch_epoch.HUB_URL, ()),
        *(
            (fetch_epoch.page_url(page), (bench.key,) if bench.key else ())
            for page, bench in fetch_epoch.BENCHMARKS.items()
        ),
    ]
    for slug, key in fetch_evals_report.BENCHMARKS.items():
        items.append((fetch_evals_report.BASE_URL.format(slug=slug), (key,)))
    for track, key in fetch_swe_atlas.TRACKS.items():
        items.append((fetch_swe_atlas.BASE_URL.format(track=track), (key,)))
    for slug, key in fetch_vals.BENCHMARKS.items():
        items.append((fetch_vals.benchmark_url(slug), (key,)))
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

    _history.sync(doc)
    path.write_text(json.dumps(doc, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
