"""Source precedence: whose number wins when two sources report one score.

Most benchmark columns in llm.json have more than one publisher -- 32 of them
carry values from two or more sources today -- so every refresh has to decide
which reading lands. That decision used to be implicit in the order
``update.py`` calls its ingests: the last writer won, which made a source's
authority a side effect of its position in ``main()``. Two things followed from
that, both visible in the file's history:

  * the result depended on *which* ingests ran. The AA-only pass in 2ab9ab0
    replaced four evals.report values with Artificial Analysis' own numbers
    (``ifbench`` on two models, ``mmlu_pro`` on two more), and the next full
    refresh, 0c24cc9, put evals.report back. A score that flips with the shape
    of the run is not a measurement of anything.
  * precedence could only be read by tracing call order through ``main()``,
    while the reasoning for it lived in prose in the README.

Rank is declared here instead, and ``update.apply_score()`` refuses a write
whose source is outranked by the one already attributed to the stored value.
The outcome no longer depends on which subset of the ingests runs, or in what
order -- the same property ``keep_best_row()`` gives row collisions inside a
single source.

The rungs, strongest first:

  0. ``RANK_AA_CODING_AGENTS`` -- Artificial Analysis' Coding Agent Index. It
     shares Terminal-Bench 4.0 (and could share DeepSWE) with AA's other
     surfaces, but runs each model under its vendor's own coding agent rather
     than AA's single harness, so the two AA readings of one column differ by
     design. Both used to sit on rung 1, where each refresh let the model pages
     overwrite the index and the index overwrite them back; the index is ranked
     above them instead, so its run is the one that lands wherever both report.
  1. ``RANK_AA`` -- Artificial Analysis' other evaluations: the API, model
     pages and official social posts. Nothing but AA overwrites them.
  2. ``RANK_BENCHMARK_SITE`` -- the leaderboard run by the team that owns the
     benchmark. First-party for the one column it publishes, and no two members
     of this rung publish the same column, so their relative order is
     unobservable and none is needed.
  3. ``RANK_THIRD_PARTY_RUN`` -- a third party that runs the models itself.
     Vals AI runs every model on its own harness and its own held-out sets, so
     its numbers are measurements rather than republished ones -- but of
     benchmarks it does not own, which is what keeps it off rung 2 next to the
     boards themselves. Its own benchmarks are the exception, and they are
     ranked as what they are: for a board Vals authored, runs and solely
     publishes -- ``vibe_code_bench_1_1`` today -- the Vals page *is* the
     benchmark's leaderboard, so it sits on rung 2 with the other first-party
     boards. ``fetch_vals.VALS_OWN_BENCHMARKS`` draws the line, and it moves
     the rank of one board, not of the source.
  4. ``RANK_CURATED`` -- a third party that compiles or vets other people's
     results rather than running them. evals.report keeps only Official and
     Verified rows (``fetch_evals_report.TRUSTED_STATUSES``), but either kind
     is a number someone else produced, under that lab's own prompts and
     settings; benchlm.ai republishes one revision of Datacurve's DeepSWE board
     without a status of its own. Both used to share rung 3 with Vals, and on
     ``mmlu_pro`` -- which evals.report and Vals both publish -- that made the
     stored value whichever of the two ran last, so ``--skip-vals`` left a
     different number than a full run. A uniform run outranks a compilation.
  5. ``RANK_AGGREGATE`` -- llm-stats, a cross-benchmark aggregate that
     republishes numbers nobody in the chain ran. Its ingest is fill-only, so
     in practice it reaches a column only where it is still null,
     custom-sourced, credited to the very page it is re-reading, or credited to
     one of rungs 6 and 7 below, the fetcher rungs that yield to it.
  6. ``RANK_ENDPOINT_RUN`` -- OpenRouter's own GPQA Diamond runs, one per
     provider endpoint serving a model, of which fetch_openrouter.py reports
     the median. A run rather than a republished number, but of whatever
     deployment each provider serves -- quantized, on its own inference stack,
     now and then plainly broken -- so the number is the model as the API
     market serves it rather than as its lab or a uniform harness measured it.
     It seeds a column every other fetcher may take over, llm-stats' fill-only
     ingest included (``yields_to_fill_only``), and only the hub and the model
     cards below leave it alone. It is not fill-only itself, so it replaces
     their numbers and a hand entry the way any better rung does.
  7. ``RANK_BENCHMARKING_HUB`` -- Epoch AI's Benchmarking Hub: Epoch's own
     GPQA Diamond and SWE-bench Verified runs, and its copies of nine other
     people's boards. A second source for every column it feeds, so it sits
     right above the model cards and yields to everything else, llm-stats'
     fill-only ingest included (``yields_to_fill_only``): it fills a cell
     nobody better reports and gives it up the moment someone does. That is
     what lets it carry files whose rows are not all what the column holds --
     Terminal-Bench 2.0's unverified submissions, HLE's full set against the
     column's text-only run -- since the board or AA takes the cell over as
     soon as it reports the model. Not fill-only itself, so it replaces a
     card's number and a hand entry.
  8. ``RANK_MODEL_CARD`` -- the Hugging Face model cards, a lab's own report
     under its own prompts and settings. Fill-only like llm-stats, and ranked
     under OpenRouter and the hub because a card is the flattered variant of a
     number they at least measured or copied from a board. It shared a rung
     with llm-stats before OpenRouter was read; both ingests are fill-only and
     neither replaces the other's value, so the split changes nothing between
     the two of them.
  9. ``RANK_HAND_ENTERED`` -- a value typed in through ``add.py`` or
     ``edit.py``. Its attribution is whatever page the entry cited, or null
     when a hand edit cleared it (``stamp_score_source`` takes None for exactly
     that), and either way it is the weakest rung: a hand entry seeds a column
     until a source measures it, and any scraper may overwrite it -- the
     fill-only aggregates of rungs 5 and 8 included, which otherwise never
     replace a stored value (``is_fetcher_source``).

Apart from the pages of one source, no two sources that can overwrite each
other share a rung: rung 2 holds that by construction, rungs 0-4 by the split
above, and rungs 5 to 8 by one each.

Two sources on the same rung may still overwrite each other, which is what lets
a source refresh its own value: rank blocks a write only when the stored value
came from a *strictly* better-ranked source.

Every URL here is read off a ``fetch_*.py`` constant rather than spelled out
again, the same rule ``fill_source_urls.py`` follows, so a scraper repointed at
a new host loses its rank instead of silently keeping one that no longer
matches where it reads.
"""

from __future__ import annotations

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
import fetch_swe_atlas
import fetch_swe_marathon
import fetch_tbench
import fetch_toolathlon
import fetch_vals
import fetch_zerobench
from _revisions import revision_key
from fill_source_urls import canonical

RANK_AA_CODING_AGENTS = 0
RANK_AA = 1
RANK_BENCHMARK_SITE = 2
RANK_THIRD_PARTY_RUN = 3
RANK_CURATED = 4
RANK_AGGREGATE = 5
RANK_ENDPOINT_RUN = 6
RANK_BENCHMARKING_HUB = 7
RANK_MODEL_CARD = 8
RANK_HAND_ENTERED = 9

# The fetcher rungs a fill-only ingest may still take a cell from.
YIELDING_RANKS = frozenset({RANK_ENDPOINT_RUN, RANK_BENCHMARKING_HUB})

# Per-score source pages, stamped into models[].scores_source alongside every
# score write, and the identities the ranks below are hung on. Stored
# canonicalized like every URL in llm.json. AA and Hugging Face pages are
# per-model and resolved where the score is written; SWE Atlas, evals.report and Vals AI
# resolve per benchmark key.
AA_CODING_AGENTS_SOURCE_URL = canonical(fetch_aa_coding_agents.URL)
OSWORLD_SOURCE_URL = canonical(fetch_osworld.OSWORLD_SITE_URL)
# The Verified board and OSWorld 2.0 live on two sites of the same lab, one
# credited per column; every tracked OSWorld 2.0 release shares the 2.0 site's
# page.
OSWORLD_V2_SOURCE_URL = canonical(fetch_osworld.OSWORLD_V2_SITE_URL)
OSWORLD_KEY_URLS = {
    key: canonical(fetch_osworld.source_url(key)) for key in fetch_osworld.KEYS
}
LLMSTATS_SOURCE_URL = canonical(fetch_llmstats.LEADERBOARD_URL)
TOOLATHLON_SOURCE_URL = canonical(fetch_toolathlon.URL)
MCP_ATLAS_SOURCE_URL = canonical(fetch_mcp_atlas.URL)
REAL_SWE_SOURCE_URL = canonical(fetch_real_swe.URL)
# The extended table, which is the complete board, not the site root: the
# root is the top ten of the same runs and prefix-matches everything under
# the host, including the per-run pages.
PROGRAMBENCH_SOURCE_URL = canonical(fetch_programbench.LEADERBOARD_URL)
# The board's own run only. The same page's externally-reported table is model
# cards under another roof, which is RANK_MODEL_CARD; fetch_zerobench.py never reads it.
ZEROBENCH_SOURCE_URL = canonical(fetch_zerobench.URL)
# cybergym.io publishes CyberGym and ExploitGym from one JSON file each; a
# score cites the board's page, one per column, not the file.
CYBERGYM_KEY_URLS = {
    key: canonical(fetch_cybergym.source_url(key)) for key in fetch_cybergym.KEYS
}
# The snapshot page the column is pinned to, not the site build's data file:
# the site root is the newer snapshot, a different task set.
SEC_BENCH_SOURCE_URL = canonical(fetch_sec_bench.URL)
# The leaderboard page, not the CSV it hydrates its table from.
BFCL_SOURCE_URL = canonical(fetch_bfcl.LEADERBOARD_URL)
# benchlm.ai's mirror of Datacurve's board, named for the column it feeds;
# Datacurve's own page is DATACURVE_SOURCE_URL below.
DEEPSWE_SOURCE_URL = canonical(fetch_deepswe.URL)
# The leaderboard page, not the JSON artifact it hydrates from.
DATACURVE_SOURCE_URL = canonical(fetch_datacurve.SITE_URL)
FRONTIERSWE_SOURCE_URL = canonical(fetch_frontierswe.URL)
# FrontierSWE is the one split source whose revisions live on different pages:
# the root board publishes V2 and /v1 the preserved V1 board, so each column
# cites the page that actually carries its number. Both rank as the benchmark's
# own site through the root prefix below, which /v1 matches on a path boundary.
FRONTIERSWE_KEY_URLS = {
    revision_key("frontierswe", label): canonical(url)
    for label, url in fetch_frontierswe.BOARD_URLS.items()
}
# The leaderboard page, not the JSON it loads: the page is what a reader opens.
FRONTIERCODE_SOURCE_URL = canonical(fetch_frontiercode.LEADERBOARD_URL)
SWE_MARATHON_SOURCE_URL = canonical(fetch_swe_marathon.URL)
# One versioned leaderboard page per column, not the function the boards are
# read from: that is what a reader opens, and the host root would prefix-match
# every board at once.
TBENCH_KEY_URLS = {
    key: canonical(fetch_tbench.board_url(key)) for key in fetch_tbench.BOARDS
}
# The leaderboard page, not the JSON endpoint it hydrates from.
AGENTS_LAST_EXAM_SOURCE_URL = canonical(fetch_agents_last_exam.LEADERBOARD_URL)
SWE_ATLAS_KEY_URLS = {
    key: canonical(fetch_swe_atlas.BASE_URL.format(track=track))
    for track, key in fetch_swe_atlas.TRACKS.items()
}
EVALS_REPORT_KEY_URLS = {
    key: canonical(fetch_evals_report.BASE_URL.format(slug=slug))
    for slug, key in fetch_evals_report.BENCHMARKS.items()
}
VALS_KEY_URLS = {
    key: canonical(fetch_vals.benchmark_url(slug))
    for slug, key in fetch_vals.BENCHMARKS.items()
}
# Vals sits on rung 3 for the boards it re-runs, and on rung 2 for the ones it
# owns: what keeps it off the benchmark-site rung is that the benchmarks it
# measures belong to someone else, and that reason does not apply to a board it
# authored, runs and is the only publisher of. Split by
# fetch_vals.VALS_OWN_BENCHMARKS rather than listed again here, so a board
# added to the ingest is ranked where it is classified.
VALS_OWN_KEY_URLS = {
    fetch_vals.BENCHMARKS[slug]: canonical(fetch_vals.benchmark_url(slug))
    for slug in sorted(fetch_vals.VALS_OWN_BENCHMARKS)
}
VALS_RERUN_KEY_URLS = {
    key: url for key, url in VALS_KEY_URLS.items() if key not in VALS_OWN_KEY_URLS
}

# The two per-model families, which are ranked by the path they live under
# rather than by one URL: AA writes a score with the page of the model it
# measured, and the Hugging Face ingest with the card it read.
AA_MODEL_PAGE_PREFIX = canonical(artificialanalysis.MODEL_PAGE_URL.format(""))
HUGGING_FACE_PREFIX = canonical(fetch_huggingface.HF_BASE)
# Epoch AI's hub: one page per benchmark read, each crediting its scores. The
# pages rather than the host, so a hand entry citing some other Epoch page (a
# report, a chart of a benchmark this repo does not read) stays hand-entered.
EPOCH_PAGE_URLS = {
    page: canonical(fetch_epoch.page_url(page)) for page in fetch_epoch.BENCHMARKS
}
# OpenRouter is per-model the same way: each score cites the page of the model
# it was read off, so the rank hangs on the host.
OPENROUTER_PREFIX = canonical(fetch_openrouter.SITE_URL)


def _ranked_prefixes() -> tuple[tuple[str, int], ...]:
    """(page prefix, rank) pairs, longest prefix first.

    Longest-first allows specific pages to override a host-wide rule: the
    Coding Agent Index lives under the artificialanalysis.ai host and outranks
    it.
    """
    pairs: list[tuple[str, int]] = [
        ("https://artificialanalysis.ai", RANK_AA),
        ("https://x.com/ArtificialAnlys", RANK_AA),
        ("https://twitter.com/ArtificialAnlys", RANK_AA),
        (AA_MODEL_PAGE_PREFIX, RANK_AA),
        (OSWORLD_SOURCE_URL, RANK_BENCHMARK_SITE),
        (OSWORLD_V2_SOURCE_URL, RANK_BENCHMARK_SITE),
        (TOOLATHLON_SOURCE_URL, RANK_BENCHMARK_SITE),
        (MCP_ATLAS_SOURCE_URL, RANK_BENCHMARK_SITE),
        (REAL_SWE_SOURCE_URL, RANK_BENCHMARK_SITE),
        (PROGRAMBENCH_SOURCE_URL, RANK_BENCHMARK_SITE),
        (BFCL_SOURCE_URL, RANK_BENCHMARK_SITE),
        (DATACURVE_SOURCE_URL, RANK_BENCHMARK_SITE),
        (FRONTIERSWE_SOURCE_URL, RANK_BENCHMARK_SITE),
        (FRONTIERCODE_SOURCE_URL, RANK_BENCHMARK_SITE),
        (SWE_MARATHON_SOURCE_URL, RANK_BENCHMARK_SITE),
        *((url, RANK_BENCHMARK_SITE) for url in TBENCH_KEY_URLS.values()),
        (AGENTS_LAST_EXAM_SOURCE_URL, RANK_BENCHMARK_SITE),
        (ZEROBENCH_SOURCE_URL, RANK_BENCHMARK_SITE),
        *((url, RANK_BENCHMARK_SITE) for url in CYBERGYM_KEY_URLS.values()),
        (SEC_BENCH_SOURCE_URL, RANK_BENCHMARK_SITE),
        *((url, RANK_BENCHMARK_SITE) for url in SWE_ATLAS_KEY_URLS.values()),
        *((url, RANK_CURATED) for url in EVALS_REPORT_KEY_URLS.values()),
        *((url, RANK_BENCHMARK_SITE) for url in VALS_OWN_KEY_URLS.values()),
        *((url, RANK_THIRD_PARTY_RUN) for url in VALS_RERUN_KEY_URLS.values()),
        (DEEPSWE_SOURCE_URL, RANK_CURATED),
        *((url, RANK_BENCHMARKING_HUB) for url in EPOCH_PAGE_URLS.values()),
        (AA_CODING_AGENTS_SOURCE_URL, RANK_AA_CODING_AGENTS),
        (LLMSTATS_SOURCE_URL, RANK_AGGREGATE),
        (OPENROUTER_PREFIX, RANK_ENDPOINT_RUN),
        (HUGGING_FACE_PREFIX, RANK_MODEL_CARD),
    ]
    return tuple(sorted(pairs, key=lambda pair: len(pair[0]), reverse=True))


RANKED_PREFIXES = _ranked_prefixes()


def source_rank(url: str | None) -> int:
    """Rank of the source that published a score, 0 (strongest) to 9.

    An unrecognised page ranks as hand-entered, and so does None: both mean the
    number reached llm.json through a person rather than through a scraper this
    repo runs. A prefix matches only on a path boundary, so lookalike hosts
    and social account names cannot inherit a trusted source's rank.
    """
    if not url:
        return RANK_HAND_ENTERED
    candidate = canonical(url)
    for prefix, rank in RANKED_PREFIXES:
        if candidate == prefix or candidate.startswith(f"{prefix}/"):
            return rank
    return RANK_HAND_ENTERED


def is_fetcher_source(url: str | None) -> bool:
    """Whether url is a page one of this repo's fetchers writes scores from.

    RANKED_PREFIXES is built from the fetchers' own URL constants, so it is the
    complete list of pages a scraper stamps; anything else -- a model card or
    vendor post cited by hand, a news story, or no page at all -- is a custom
    source. Every fetcher may replace a custom-sourced value, the fill-only
    ones included, and one reporting the same number takes over its credit.
    """
    return source_rank(url) < RANK_HAND_ENTERED


def yields_to_fill_only(new_url: str | None, stored_url: str | None) -> bool:
    """Whether a fill-only fetcher reading new_url may replace stored_url's value.

    Fill-only ingests never replace another fetcher's number -- except one on
    a rung that exists to be taken over: OpenRouter's endpoint runs, which seed
    GPQA Diamond for models nobody else measured, and Epoch AI's hub, a second
    source for every column it feeds. Every better-ranked fetcher, llm-stats
    included, may replace either. The model cards rank below both and so still
    leave them alone.
    """
    stored = source_rank(stored_url)
    return stored in YIELDING_RANKS and source_rank(new_url) < stored


def may_overwrite(new_url: str | None, stored_url: str | None) -> bool:
    """Whether a score read from new_url may replace one credited to stored_url.

    Equal ranks pass: that is a source refreshing its own number, or another
    page of the same source (one AA model page replacing another's credit).
    """
    return source_rank(new_url) <= source_rank(stored_url)
