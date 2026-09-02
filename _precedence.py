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

  1. ``RANK_AA`` -- Artificial Analysis' own evaluations, from the API and the
     model pages. First-party runs of one harness across the whole field, and
     the leading source for 21 of the columns. Locked at the top: nothing
     overwrites an AA number except a later AA number.
  2. ``RANK_BENCHMARK_SITE`` -- the leaderboard run by the team that owns the
     benchmark. First-party for the one column it publishes, and no two members
     of this rung publish the same column, so their relative order is
     unobservable and none is needed.
  3. ``RANK_CURATED`` -- a third party that re-runs or vets what it publishes.
     evals.report qualifies because it keeps only Official and Verified rows
     (``fetch_evals_report.TRUSTED_STATUSES``); benchlm.ai republishes without
     a status of its own, and is here because it is a compiler of results
     rather than a lab reporting on itself.
  4. ``RANK_AA_CODING_AGENTS`` -- AA's Coding Agent Index. AA-published, but
     these are AA's *own harness* over someone else's benchmark, and they
     disagree systematically with that benchmark's board, so the index does not
     inherit rank 1: it sits above the aggregates as a gap-filler, which is the
     role the fill-only flag on its ingest already gave it.
  5. ``RANK_AGGREGATE`` -- cross-benchmark aggregates that republish numbers
     nobody in the chain ran: llm-stats and the Hugging Face model cards. Both
     ingests are fill-only, so in practice they reach a column only where it is
     still null.
  6. ``RANK_HAND_ENTERED`` -- a value typed in through ``add.py`` or
     ``edit.py``. Its attribution is whatever page the entry cited, or null
     when a hand edit cleared it (``stamp_score_source`` takes None for exactly
     that), and either way it is the weakest rung: a hand entry seeds a column
     until a source measures it, and any scraper may overwrite it.

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
import fetch_bfcl
import fetch_datacurve
import fetch_deepswe
import fetch_evals_report
import fetch_frontiercode
import fetch_frontierswe
import fetch_huggingface
import fetch_llmstats
import fetch_mcp_atlas
import fetch_osworld
import fetch_swe_atlas
import fetch_swe_marathon
import fetch_toolathlon
from fill_source_urls import canonical

RANK_AA = 1
RANK_BENCHMARK_SITE = 2
RANK_CURATED = 3
RANK_AA_CODING_AGENTS = 4
RANK_AGGREGATE = 5
RANK_HAND_ENTERED = 6

# Per-score source pages, stamped into models[].scores_source alongside every
# score write, and the identities the ranks below are hung on. Stored
# canonicalized like every URL in llm.json. AA and Hugging Face pages are
# per-model and resolved where the score is written; SWE Atlas and evals.report
# resolve per benchmark key.
AA_CODING_AGENTS_SOURCE_URL = canonical(fetch_aa_coding_agents.URL)
OSWORLD_SOURCE_URL = canonical(fetch_osworld.OSWORLD_SITE_URL)
LLMSTATS_SOURCE_URL = canonical(fetch_llmstats.LEADERBOARD_URL)
TOOLATHLON_SOURCE_URL = canonical(fetch_toolathlon.URL)
MCP_ATLAS_SOURCE_URL = canonical(fetch_mcp_atlas.URL)
# The leaderboard page, not the CSV it hydrates its table from.
BFCL_SOURCE_URL = canonical(fetch_bfcl.LEADERBOARD_URL)
DEEPSWE_SOURCE_URL = canonical(fetch_deepswe.URL)
# The leaderboard page, not the JSON artifact it hydrates from.
DATACURVE_SOURCE_URL = canonical(fetch_datacurve.SITE_URL)
FRONTIERSWE_SOURCE_URL = canonical(fetch_frontierswe.URL)
# The leaderboard page, not the JSON it loads: the page is what a reader opens.
FRONTIERCODE_SOURCE_URL = canonical(fetch_frontiercode.LEADERBOARD_URL)
SWE_MARATHON_SOURCE_URL = canonical(fetch_swe_marathon.URL)
SWE_ATLAS_KEY_URLS = {
    key: canonical(fetch_swe_atlas.BASE_URL.format(track=track))
    for track, key in fetch_swe_atlas.TRACKS.items()
}
EVALS_REPORT_KEY_URLS = {
    key: canonical(fetch_evals_report.BASE_URL.format(slug=slug))
    for slug, key in fetch_evals_report.BENCHMARKS.items()
}

# The two per-model families, which are ranked by the path they live under
# rather than by one URL: AA writes a score with the page of the model it
# measured, and the Hugging Face ingest with the card it read.
AA_MODEL_PAGE_PREFIX = canonical(artificialanalysis.MODEL_PAGE_URL.format(""))
HUGGING_FACE_PREFIX = canonical(fetch_huggingface.HF_BASE)


def _ranked_prefixes() -> tuple[tuple[str, int], ...]:
    """(page prefix, rank) pairs, longest prefix first.

    Longest-first is what separates two sources on one host: AA's model pages
    are rank 1 and its Coding Agent Index rank 4, both under
    ``artificialanalysis.ai``, and the longer of the two prefixes has to be
    tried first or every AA URL would answer to whichever came earlier.
    """
    pairs: list[tuple[str, int]] = [
        (AA_MODEL_PAGE_PREFIX, RANK_AA),
        (OSWORLD_SOURCE_URL, RANK_BENCHMARK_SITE),
        (TOOLATHLON_SOURCE_URL, RANK_BENCHMARK_SITE),
        (MCP_ATLAS_SOURCE_URL, RANK_BENCHMARK_SITE),
        (BFCL_SOURCE_URL, RANK_BENCHMARK_SITE),
        (DATACURVE_SOURCE_URL, RANK_BENCHMARK_SITE),
        (FRONTIERSWE_SOURCE_URL, RANK_BENCHMARK_SITE),
        (FRONTIERCODE_SOURCE_URL, RANK_BENCHMARK_SITE),
        (SWE_MARATHON_SOURCE_URL, RANK_BENCHMARK_SITE),
        *((url, RANK_BENCHMARK_SITE) for url in SWE_ATLAS_KEY_URLS.values()),
        *((url, RANK_CURATED) for url in EVALS_REPORT_KEY_URLS.values()),
        (DEEPSWE_SOURCE_URL, RANK_CURATED),
        (AA_CODING_AGENTS_SOURCE_URL, RANK_AA_CODING_AGENTS),
        (LLMSTATS_SOURCE_URL, RANK_AGGREGATE),
        (HUGGING_FACE_PREFIX, RANK_AGGREGATE),
    ]
    return tuple(sorted(pairs, key=lambda pair: len(pair[0]), reverse=True))


RANKED_PREFIXES = _ranked_prefixes()


def source_rank(url: str | None) -> int:
    """Rank of the source that published a score, 1 (strongest) to 6.

    An unrecognised page ranks as hand-entered, and so does None: both mean the
    number reached llm.json through a person rather than through a scraper this
    repo runs. A prefix matches only on a path boundary, so a leaderboard that
    someday lives at ``.../models-v2`` does not inherit the rank of
    ``.../models``.
    """
    if not url:
        return RANK_HAND_ENTERED
    candidate = canonical(url)
    for prefix, rank in RANKED_PREFIXES:
        if candidate == prefix or candidate.startswith(f"{prefix}/"):
            return rank
    return RANK_HAND_ENTERED


def may_overwrite(new_url: str | None, stored_url: str | None) -> bool:
    """Whether a score read from new_url may replace one credited to stored_url.

    Equal ranks pass: that is a source refreshing its own number, or two pages
    of equal standing, where the later write is the newer measurement.
    """
    return source_rank(new_url) <= source_rank(stored_url)
