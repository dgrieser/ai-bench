#!/usr/bin/env python3
"""
Fetch benchmark scores from Epoch AI's Benchmarking Hub, https://epoch.ai/benchmarks

Epoch publishes the whole hub as one download, ``benchmark_data.zip``: a CSV
per benchmark, one row per model configuration ("glm-5.2_max"), plus
``model_metadata.csv``, which names the model each configuration belongs to
(its ``model_group``, "GLM-5.2") and whether its weights are open. The data is
CC BY 4.0 -- free to reuse with credit, which llm.json's ``credits`` carries
and llm.html prints under the benchmarks (see EPOCH_CREDIT below).

The hub holds two kinds of file, and five of them are read:

  * Epoch's own runs, the same harness for every model: GPQA Diamond and
    SWE-bench Verified. A third-party measurement of a benchmark someone else
    owns -- the relation Vals AI stands in.
  * Mirrors of a benchmark's own leaderboard (the ``*_external.csv`` files),
    each row citing the board it came from: DeepSWE (Datacurve), FrontierSWE
    and FrontierCode (Cognition). Their numbers are the board's.

The three mirrors feed revision columns, and the rule for those is that a
source which does not say which revision it measured does not write to one
(_revisions.py). The CSVs name none, but Epoch's page for each benchmark does:
its title is "DeepSWE v1.1" or "FrontierSWE (v2)", and FrontierCode's, whose
title is bare, says it uses "only the current 1.1 revision". So each mirror's
revision is read off its page on every run, and a page that names none, or a
revision llm.json has no column for, drops that benchmark with a warning
instead of guessing -- the silent version bump a pinned column would invite,
where v1.2 numbers land in deepswe_1_1, cannot happen.

Deliberately not read, though the hub carries them:

  * Terminal-Bench 2.0 -- the file holds the board's unverified self-reports
    beside its verified runs with nothing to tell them apart; 64 of its rows are
    unverified submissions on tbench.ai, which terminal_bench_2_0 excludes.
  * OSWorld 2.0 -- one file mixes releases (MiniMax M3 and Kimi K2.6 at 4.6 are
    the 2026.06.24 release, Opus 5 at 31.4 the 2026.08.08 one), and neither
    file nor page says which row is which. Those are separate columns here.
  * OSWorld, HLE, CritPt, SciCode -- OSWorld's file is the pre-Verified series
    at mixed step budgets; HLE's is the full multimodal set rather than the
    text-only subset the column tracks; CritPt and SciCode mirror an older cut
    of Artificial Analysis, which artificialanalysis.py reads at the source.

Every row is one configuration; update.py folds a model's configurations onto
one llm.json slug per column, best run first, the rule every leaderboard ingest
applies.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass

from _fetch_checks import check_fractions, require_columns, require_rows
from _revisions import KNOWN_REVISIONS, known_revision_key, revision_key, subset_base

SITE_URL = "https://epoch.ai"
# The hub's index, which is what the Sources panel links.
HUB_URL = f"{SITE_URL}/benchmarks"
# What this script requests: every benchmark's CSV in one archive.
DATA_URL = f"{SITE_URL}/data/benchmark_data.zip"
PAGE_URL = HUB_URL + "/{page}"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

# The attribution the licence asks for, in the words Epoch's own README gives,
# as it is stored in llm.json's credits. test_epoch.py holds llm.json to it.
EPOCH_CREDIT = {
    "name": "Epoch AI",
    "work": "Capabilities & Benchmarking",
    "url": HUB_URL,
    "license": "CC BY 4.0",
    "license_url": LICENSE_URL,
    "citation": (
        "Epoch AI, ‘Capabilities & Benchmarking’. Published online at epoch.ai. "
        "Retrieved from ‘https://epoch.ai/benchmarks’ [online resource]."
    ),
}

METADATA_FILE = "model_metadata.csv"
VERSION_COLUMN = "Model version"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


@dataclass(frozen=True)
class Benchmark:
    """One hub file and the llm.json column(s) it feeds.

    ``key`` is the column for a benchmark with one; ``base`` (and ``subset``)
    name a revision-split one instead, whose column is resolved from the
    revision Epoch's page states.
    """

    file: str
    column: str
    key: str | None = None
    base: str | None = None
    subset: str | None = None


# Epoch's page slug (epoch.ai/benchmarks/<slug>) -> the file behind it.
BENCHMARKS: dict[str, Benchmark] = {
    # Epoch's own runs. "Best score (across scorers)" is the column the hub's
    # own benchmark_metadata.csv names as each file's score.
    "gpqa-diamond": Benchmark("gpqa_diamond.csv", "Best score (across scorers)", key="gpqa_diamond"),
    "swe-bench-verified": Benchmark(
        "swe_bench_verified.csv", "Best score (across scorers)", key="swe_bench_verified"
    ),
    # Mirrors of the benchmarks' own boards, routed by the revision on the page.
    "deepswe": Benchmark("deepswe_external.csv", "Pass@1", base="deepswe"),
    "frontierswe": Benchmark("frontierswe_external.csv", "Score", base="frontierswe"),
    # Epoch charts Cognition's Main board ("Our chart reports the Main score"),
    # the 100 hardest tasks, which is frontiercode_1_x rather than Extended.
    "frontiercode": Benchmark(
        "frontiercode_external.csv", "Main score", base="frontiercode", subset="main"
    ),
}


def page_url(page: str) -> str:
    """The hub page a score read from ``page``'s file is credited to."""
    return PAGE_URL.format(page=page)


def possible_keys() -> set[str]:
    """Every llm.json column a row of this source could be filed under."""
    keys: set[str] = set()
    for bench in BENCHMARKS.values():
        if bench.key:
            keys.add(bench.key)
            continue
        base = subset_base(bench.base, bench.subset)
        keys.update(revision_key(base, label) for label in KNOWN_REVISIONS.get(base, ()))
    return keys


def fetch_bytes(url: str, retries: int = 3, delay: float = 2.0) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
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


def fetch_archive() -> zipfile.ZipFile:
    print(f"Fetching {DATA_URL} ...", file=sys.stderr)
    return zipfile.ZipFile(io.BytesIO(fetch_bytes(DATA_URL)))


def read_csv(
    archive: zipfile.ZipFile, name: str, required: tuple[str, ...] = (VERSION_COLUMN,)
) -> list[dict[str, str]]:
    """One file of the archive as rows. Cells may hold newlines, so csv parses it."""
    try:
        raw = archive.read(name)
    except KeyError:
        raise ValueError(
            f"{DATA_URL} no longer carries {name} -- the archive changed; "
            "refusing to read its benchmark as empty."
        ) from None
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    rows = list(reader)
    require_columns(reader.fieldnames or [], required, f"{DATA_URL}:{name}")
    return rows


def open_weights(accessibility: str) -> bool | None:
    """Epoch's accessibility label as an open-weights verdict, None when unset."""
    label = accessibility.strip().lower()
    if not label:
        return None
    return label.startswith("open weights")


def load_models(archive: zipfile.ZipFile) -> dict[str, dict[str, object]]:
    """Model version -> {"model": its model group, "open_weights": verdict}."""
    rows = read_csv(archive, METADATA_FILE, ("model_version", "model_group", "accessibility"))
    models: dict[str, dict[str, object]] = {}
    for row in rows:
        version = (row.get("model_version") or "").strip()
        group = (row.get("model_group") or "").strip()
        if not version or not group:
            continue
        models[version] = {
            "model": group,
            "open_weights": open_weights(row.get("accessibility") or ""),
        }
    require_rows(list(models), f"{DATA_URL}:{METADATA_FILE}", what="model metadata rows")
    return models


_TITLE_RE = re.compile(r'<meta[^>]+property="og:title"[^>]+content="([^"]*)"', re.IGNORECASE)
_TITLE_TAG_RE = re.compile(r"<title>([^<]*)</title>", re.IGNORECASE)
_TITLE_REVISION_RE = re.compile(r"\bv(\d+(?:\.\d+)*)\b", re.IGNORECASE)
# The one wording a bare-titled page names its revision in, FrontierCode's
# "using only the current 1.1 revision".
_TEXT_REVISION_RE = re.compile(r"\bcurrent\s+v?(\d+(?:\.\d+)+)\s+revision\b", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def page_revision(page_html: str) -> str | None:
    """The revision a hub page says its data is, or None when it names none.

    The title is read first ("DeepSWE v1.1", "FrontierSWE (v2)"); a bare title
    falls back to the methodology's "the current X revision". Anything else --
    a year, a task count -- is not a revision and is never read as one.
    """
    match = _TITLE_RE.search(page_html) or _TITLE_TAG_RE.search(page_html)
    if match:
        found = _TITLE_REVISION_RE.search(html.unescape(match.group(1)))
        if found:
            return found.group(1)
    text = re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub(" ", page_html)))
    found = _TEXT_REVISION_RE.search(text)
    return found.group(1) if found else None


def resolve_key(page: str, bench: Benchmark, fetch_page=None) -> tuple[str | None, str | None]:
    """(column, revision) one benchmark's rows are filed under.

    A fixed-key benchmark needs no page. A revision-split one reads its page,
    and a page that names no revision, or one llm.json has no column for,
    resolves to no column -- which drops the benchmark rather than guessing.
    """
    if bench.key:
        return bench.key, None
    fetch_page = fetch_page or (lambda url: fetch_bytes(url).decode("utf-8", errors="replace"))
    url = page_url(page)
    print(f"Fetching {url} ...", file=sys.stderr)
    revision = page_revision(fetch_page(url))
    base = subset_base(bench.base, bench.subset)
    key = known_revision_key(base, revision) if base else None
    if key is None:
        print(
            f"  warning: {url} names "
            + (f"revision {revision!r}, which llm.json has no {base} column for"
               if revision else "no revision")
            + f"; {bench.file} is not read.",
            file=sys.stderr,
        )
    return key, revision


def read_benchmark(
    archive: zipfile.ZipFile,
    models: dict[str, dict[str, object]],
    page: str,
    key: str | None,
    revision: str | None,
) -> list[dict]:
    """One row per scored configuration in one benchmark's file."""
    bench = BENCHMARKS[page]
    source = f"{DATA_URL}:{bench.file}"
    rows = read_csv(archive, bench.file, (VERSION_COLUMN, bench.column))
    results: list[dict] = []
    fractions: list[float] = []
    for row in rows:
        version = (row.get(VERSION_COLUMN) or "").strip()
        raw = (row.get(bench.column) or "").strip()
        # A row with no model version is a duplicate the hub keeps for its
        # chart (FrontierCode carries five), with no model to credit it to.
        if not version or not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        model = models.get(version)
        fractions.append(value)
        results.append(
            {
                "benchmark": page,
                "key": key,
                "revision": revision,
                "model": model["model"] if model else version,
                "version": version,
                "open_weights": model["open_weights"] if model else None,
                "score": round(value * 100, 2),
                "source": page_url(page),
            }
        )
    require_rows(results, source)
    check_fractions(fractions, source, field=bench.column)
    results.sort(key=lambda r: -r["score"])
    return results


def get_scores(pages: list[str] | None = None, archive: zipfile.ZipFile | None = None,
               fetch_page=None) -> list[dict]:
    """Return a list of dicts: benchmark, key, revision, model, version,
    open_weights, score, source.

    ``model`` is Epoch's model group (what the mapping is keyed by), ``version``
    the configuration the row measured, ``score`` a percentage, ``source`` the
    hub page the score is credited to. A benchmark whose revision cannot be
    established contributes no rows.
    """
    archive = archive or fetch_archive()
    models = load_models(archive)
    results: list[dict] = []
    for page in pages or list(BENCHMARKS):
        key, revision = resolve_key(page, BENCHMARKS[page], fetch_page)
        if key is None:
            continue
        results.extend(read_benchmark(archive, models, page, key, revision))
    return results


def get_catalogue(archive: zipfile.ZipFile | None = None) -> list[dict]:
    """Every model group scored in a file this script reads, with Epoch's
    open-weights verdict (True when any of its configurations is open)."""
    archive = archive or fetch_archive()
    models = load_models(archive)
    verdicts: dict[str, set[bool]] = {}
    for bench in BENCHMARKS.values():
        for row in read_csv(archive, bench.file, (VERSION_COLUMN, bench.column)):
            version = (row.get(VERSION_COLUMN) or "").strip()
            if not version or not (row.get(bench.column) or "").strip():
                continue
            model = models.get(version) or {"model": version, "open_weights": None}
            seen = verdicts.setdefault(str(model["model"]), set())
            if model["open_weights"] is not None:
                seen.add(bool(model["open_weights"]))
    return [
        {"model": name, "open_weights": True if True in seen else False if seen else None}
        for name, seen in sorted(verdicts.items())
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch benchmark scores from Epoch AI's Benchmarking Hub."
    )
    parser.add_argument(
        "--benchmark",
        choices=["all", *BENCHMARKS],
        default="all",
        help="Hub benchmark to read (default: all).",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "names", "catalogue"],
        default="table",
        help="Output format (default: table). names lists the model groups the "
        "mapping is reviewed against; catalogue adds Epoch's open-weights verdict.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.format in ("names", "catalogue"):
        catalogue = get_catalogue()
        if args.format == "names":
            for entry in catalogue:
                print(entry["model"])
        else:
            print(json.dumps(catalogue, ensure_ascii=False))
        return 0

    pages = None if args.benchmark == "all" else [args.benchmark]
    scores = get_scores(pages)
    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
        return 0

    width = max([len("MODEL"), *(len(e["version"]) for e in scores)])
    fmt = f"{{:<22}}  {{:<{width}}}  {{:>6}}"
    print(fmt.format("COLUMN", "MODEL", "SCORE"))
    for entry in scores:
        print(fmt.format(entry["key"], entry["version"], entry["score"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
    except (urllib.error.URLError, OSError, zipfile.BadZipFile) as exc:
        print(f"error: could not fetch Epoch AI's hub: {exc}", file=sys.stderr)
        raise SystemExit(1)
