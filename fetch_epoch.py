#!/usr/bin/env python3
"""
Fetch benchmark scores from Epoch AI's Benchmarking Hub, https://epoch.ai/benchmarks

Epoch publishes the whole hub as one download, ``benchmark_data.zip``: a CSV
per benchmark, one row per model configuration ("glm-5.2_max"), plus
``model_metadata.csv``, which names the model each configuration belongs to
(its ``model_group``, "GLM-5.2") and whether its weights are open. The data is
CC BY 4.0 -- free to reuse with credit, which llm.json's ``credits`` carries
and llm.html prints under the benchmarks (see EPOCH_CREDIT below).

The hub holds two kinds of file, and eleven of them are read:

  * Epoch's own runs, the same harness for every model: GPQA Diamond and
    SWE-bench Verified.
  * Mirrors of someone else's board (the ``*_external.csv`` files), each row
    citing the board it came from: DeepSWE (Datacurve), FrontierSWE, FrontierCode
    (Cognition), Terminal-Bench 2.0, OSWorld-Verified, OSWorld 2.0, Humanity's
    Last Exam, and CritPt and SciCode as Artificial Analysis runs them.

Every one of them is a second source here, ranked just above the model cards
(_precedence.RANK_BENCHMARKING_HUB): whatever the board itself, AA, Vals or
any compilation reports takes the cell over, so the hub fills gaps rather than
deciding a column. That is also why the mirrors whose rows cannot all be
checked are read at all -- a row the board would not carry is overruled the
moment the board reports that model.

Three mirrors feed revision columns, and the rule for those is that a source
which does not say which revision it measured does not write to one
(_revisions.py). The CSVs name none, but Epoch's page for each benchmark does:
its title is "DeepSWE v1.1" or "FrontierSWE (v2)", and FrontierCode's, whose
title is bare, says it uses "only the current 1.1 revision". So each mirror's
revision is read off its page on every run, and a page that names none, or a
revision llm.json has no column for, drops that benchmark with a warning
instead of guessing -- the silent version bump a pinned column would invite,
where v1.2 numbers land in deepswe_1_1, cannot happen.

OSWorld 2.0 splits by release, and the hub keeps each model's newest runs in
one file with no release on the row or the page. The release is read off the
official board instead: a hub row is filed under the release of the one board
run with the same model, reasoning setting and binary accuracy, and a row that
matches no run, or runs on two releases, is dropped and counted.

What each of the others holds, and where it differs from its column:

  * Terminal-Bench 2.0 -- rows citing the 2.0 board only. The file carries the
    board's unverified submissions beside its verified runs with nothing to
    tell them apart; terminal_bench_2_0 is read from the verified runs, so an
    unverified number stands only until the board verifies that model.
  * OSWorld-Verified -- the board's rows at a step budget of 100 or less; a
    vendor announcement the file also carries is dropped.
  * Humanity's Last Exam -- the full set, about a tenth of it multimodal, where
    the hle column is AA's text-only run: a hub number stands in a cell AA has
    not measured.
  * CritPt, SciCode -- Artificial Analysis' own boards as Epoch last copied
    them; artificialanalysis.py reads the current ones and outranks the copy.

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
from typing import Callable

from _fetch_checks import check_fractions, check_percentages, require_columns, require_rows
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
    revision Epoch's page states; ``releases`` files each row under the OSWorld
    2.0 release the official board ran it on. ``percent`` marks a file already
    in percentages (the hub's own metadata gives it a 0.01 scale), and ``keep``
    is the row filter for a file that holds rows its column does not.
    """

    file: str
    column: str
    key: str | None = None
    base: str | None = None
    subset: str | None = None
    releases: bool = False
    percent: bool = False
    keep: Callable[[dict[str, str]], bool] | None = None


_STEPS_RE = re.compile(r"\((\d+)\s+steps?\)", re.IGNORECASE)


def _tbench_2_0_row(row: dict[str, str]) -> bool:
    """A Terminal-Bench row citing the 2.0 board, by link or by name."""
    source = f"{row.get('Source') or ''} {row.get('Source Link') or ''}".lower()
    return "terminal-bench/2.0" in source or "terminal-bench v2 leaderboard" in source


def _osworld_verified_row(row: dict[str, str]) -> bool:
    """An OSWorld-Verified board row at a budget the column takes (<= 100 steps)."""
    if not (row.get("Source link") or "").startswith("https://os-world.github.io"):
        return False
    steps = _STEPS_RE.search(row.get("Agent") or "")
    return steps is None or int(steps.group(1)) <= 100


def _exploitbench_plain_row(row: dict[str, str]) -> bool:
    """An ExploitBench run in the plain regime, not one AutoNudge kept prompting."""
    return (row.get("Autonudge") or "").strip().lower() not in {"true", "1", "yes"}


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
    # Epoch's page is "Terminal-Bench 2.0"; the rows are filtered to the ones
    # citing that board, so a later board filed in the same file stays out.
    "terminal-bench": Benchmark(
        "terminalbench_external.csv", "Accuracy mean", key="terminal_bench_2_0", keep=_tbench_2_0_row
    ),
    "os-world": Benchmark(
        "os_world_external.csv", "Score", key="osworld_verified", percent=True, keep=_osworld_verified_row
    ),
    "osworld-2": Benchmark("osworld_2_external.csv", "Binary accuracy", releases=True),
    "hle": Benchmark("hle_external.csv", "Accuracy", key="hle"),
    "critpt": Benchmark("critpt_external.csv", "Accuracy", key="critpt"),
    "scicode": Benchmark("scicode_external.csv", "Score", key="scicode"),
    # A mirror of exploitbench.ai, whose own board is inlined in a hashed
    # JavaScript chunk; each model is listed plain and under AutoNudge, where
    # the harness keeps prompting the agent to continue, and only plain runs
    # are the column.
    "exploitbench": Benchmark(
        "exploitbench_external.csv", "Mean capability", key="exploitbench", keep=_exploitbench_plain_row
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
        if bench.releases:
            keys.update(osworld_release_keys())
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

    A fixed-key benchmark needs no page, and neither does OSWorld 2.0, whose
    column is resolved per row (RELEASE_PER_ROW). A revision-split one reads
    its page, and a page that names no revision, or one llm.json has no column
    for, resolves to no column -- which drops the benchmark rather than
    guessing.
    """
    if bench.key:
        return bench.key, None
    if bench.releases:
        return RELEASE_PER_ROW, None
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


# resolve_key()'s answer for a benchmark whose rows each find their own column.
RELEASE_PER_ROW = "<per row>"

_TRAILING_PARENS_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _norm_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def osworld_release_keys() -> tuple[str, ...]:
    import fetch_osworld

    return tuple(fetch_osworld.RELEASES.values())


def osworld_board(fetch_json=None) -> list[tuple[str, str, str, float, str | None]]:
    """(model, reasoning, tool setting, binary accuracy %, column) for every official full-set
    run at the 500-step budget on OSWorld 2.0's own board.

    The column is None for a release llm.json has no column for, so a hub row
    matching only such a run is dropped rather than filed next door.
    """
    import fetch_osworld

    if fetch_json is None:
        print(f"Fetching {fetch_osworld.OSWORLD_V2_JSON_URL} ...", file=sys.stderr)
        payload = json.loads(fetch_bytes(fetch_osworld.OSWORLD_V2_JSON_URL))
    else:
        payload = fetch_json(fetch_osworld.OSWORLD_V2_JSON_URL)
    runs: list[tuple[str, str, str, float, str | None]] = []
    for row in fetch_osworld.check_v2_payload(payload):
        # The board's own percentage, on the full task set at the tracked
        # budget: the runs fetch_osworld.py files, and the only ones the hub's
        # binary accuracy can be a copy of.
        accuracy = row.get(fetch_osworld.V2_METRIC)
        if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
            continue
        if row.get("stepBudget") != fetch_osworld.V2_STEP_BUDGET:
            continue
        if not fetch_osworld._v2_in_scope(row, payload, fetch_osworld.V2_DATASET_SCOPE):
            continue
        release = fetch_osworld._v2_release(row, payload)
        runs.append(
            (
                _norm_label(str(row.get("model") or "")),
                str(row.get("reasoning") or "").strip().lower(),
                _norm_label(str(row.get("toolSetting") or "")),
                round(float(accuracy), 1),
                fetch_osworld.RELEASES.get(release or ""),
            )
        )
    return runs


def osworld_release(row: dict[str, str], value: float, board: list) -> str | None:
    """The column of the one board run a hub OSWorld 2.0 row copies, else None."""
    label = _norm_label(_TRAILING_PARENS_RE.sub("", row.get("Name") or ""))
    reasoning = (row.get("Reasoning") or "").strip().lower()
    tool = _norm_label(row.get("Tool setting") or "")
    accuracy = round(value * 100, 1)
    keys = {
        key
        for model, effort, setting, score, key in board
        if model == label
        and score == accuracy
        and (not effort or not reasoning or effort == reasoning)
        and (not setting or not tool or setting == tool)
    }
    return keys.pop() if len(keys) == 1 else None


def read_benchmark(
    archive: zipfile.ZipFile,
    models: dict[str, dict[str, object]],
    page: str,
    key: str | None,
    revision: str | None,
    board: list | None = None,
) -> list[dict]:
    """One row per scored configuration in one benchmark's file."""
    bench = BENCHMARKS[page]
    source = f"{DATA_URL}:{bench.file}"
    rows = read_csv(archive, bench.file, (VERSION_COLUMN, bench.column))
    results: list[dict] = []
    values: list[float] = []
    unplaced: list[str] = []
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
        # The scale is checked on every row of the file, kept or not, so a
        # file that changed scale fails whatever the filter lets through.
        values.append(value)
        if bench.keep is not None and not bench.keep(row):
            continue
        row_key = key
        if key == RELEASE_PER_ROW:
            row_key = osworld_release(row, value, board or [])
            if row_key is None:
                unplaced.append(version)
                continue
        model = models.get(version)
        results.append(
            {
                "benchmark": page,
                "key": row_key,
                "revision": revision,
                "model": model["model"] if model else version,
                "version": version,
                "open_weights": model["open_weights"] if model else None,
                "score": round(value if bench.percent else value * 100, 2),
                "source": page_url(page),
            }
        )
    if bench.percent:
        check_percentages([{"score": v} for v in values], source)
    else:
        check_fractions(values, source, field=bench.column)
    if unplaced:
        print(
            f"  {page}: {len(unplaced)} row(s) match no single tracked release on "
            f"the official board and are not read: {', '.join(sorted(set(unplaced)))}",
            file=sys.stderr,
        )
    require_rows(results, source)
    results.sort(key=lambda r: -r["score"])
    return results


def get_scores(pages: list[str] | None = None, archive: zipfile.ZipFile | None = None,
               fetch_page=None, fetch_json=None) -> list[dict]:
    """Return a list of dicts: benchmark, key, revision, model, version,
    open_weights, score, source.

    ``model`` is Epoch's model group (what the mapping is keyed by), ``version``
    the configuration the row measured, ``score`` a percentage, ``source`` the
    hub page the score is credited to. A benchmark whose revision cannot be
    established contributes no rows, and an OSWorld 2.0 row whose release
    cannot be established is left out.
    """
    archive = archive or fetch_archive()
    models = load_models(archive)
    results: list[dict] = []
    for page in pages or list(BENCHMARKS):
        key, revision = resolve_key(page, BENCHMARKS[page], fetch_page)
        if key is None:
            continue
        board = osworld_board(fetch_json) if key == RELEASE_PER_ROW else None
        results.extend(read_benchmark(archive, models, page, key, revision, board))
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
