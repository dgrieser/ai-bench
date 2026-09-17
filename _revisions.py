"""Benchmark revisions: their labels, their order, and the column each feeds.

Four benchmarks in this table have published more than one revision of
themselves, and in every case the revisions are *not* comparable -- a re-run
changes the task set, the verification or the scoring, so a 1.0 number and a
1.1 number are two different measurements that happen to share a name:

  * **DeepSWE** -- deepswe.datacurve.ai serves one JSON artifact per revision
    and toggles between them. Re-runs moved models by more than 50 points
    (DeepSeek V4 Pro: 7.5 -> 62.8), and 1.0 is the only revision that ever
    scored a dozen models retired before the re-run.
  * **FrontierCode** -- Cognition's payload carries one block per revision
    ("v1_1", "v1"), the current one covering only what was re-run. It also
    scores every run twice, over two nested task sets -- Main, the 100 hardest
    tasks, and Extended, all 150 -- and publishes them as two leaderboards that
    are not to be mixed: Extended reads about thirteen points higher, because
    the tasks it adds back are the ones Main drops for being easy. So a subset
    splits a column the same way a revision does, and SUBSET_BASES below says
    which base each one's columns are spelled with.
  * **FrontierSWE** -- frontierswe.com serves V2 at its root and keeps V1 at
    /v1 "preserved as published". The revisions do not even share a metric: V2
    scores 34 tasks as a mean@5 percentage, V1 ranked 17 tasks by average rank
    and by dominance, a pairwise win rate.
  * **SWE-Marathon** -- swe-marathon.org ships a "v1.0 Archive" board beside
    the "v1.1 Current" one; 1.1 updated all 20 tasks with tighter verification
    and closed-internet execution, and its leader sits 21 points above the
    archive's.

llm.json therefore gives each revision its own column, the way
``terminal_bench_2_0`` and ``terminal_bench_2_1`` are already separate, rather
than blending them into one. This module owns the naming so a scraper, the
ingest and the mappings cannot disagree about which column a row belongs in.

The rule for a *source* is the one the README's version traps already state for
BFCL and Toolathlon: a publisher that does not say which revision it measured
does not write to a revision column. ``KNOWN_REVISIONS`` is what "say which"
resolves against -- a revision llm.json has no column for is skipped and
reported rather than folded into a neighbouring one.
"""

from __future__ import annotations

import re

_REVISION_NUM_RE = re.compile(r"\d+")

# Benchmark base key -> the revision labels llm.json carries a column for.
# Adding a revision here is not enough on its own: llm.json needs the matching
# "<base>_<major>_<minor>" benchmark entry, which is what revision_key() spells.
KNOWN_REVISIONS: dict[str, tuple[str, ...]] = {
    "deepswe": ("1.0", "1.1"),
    "frontiercode": ("1.0", "1.1"),
    # Cognition scores two task sets from the same run and publishes them as
    # two leaderboards: Main, the 100 hardest tasks, and Extended, all 150.
    # Extended's columns are spelled with a base of their own (see
    # SUBSET_BASES), so 1.1 Extended is frontiercode_extended_1_1. Only 1.1 has
    # one: 1.0's Extended board is a superseded run of a subset nothing here
    # aggregates, and a column no index and no reader looks at is a column
    # nothing keeps honest.
    "frontiercode_extended": ("1.1",),
    # FrontierSWE numbered its re-run V2, not 1.1; a bare major fills out to
    # ".0" the same way, so the columns are frontierswe_1_0 and frontierswe_2_0.
    "frontierswe": ("1.0", "2.0"),
    "swe_marathon": ("1.0", "1.1"),
}


# A benchmark that scores more than one task set per run keeps a column per
# subset as well as per revision, and the subset is spelled into the base so the
# revision suffix keeps its meaning. FrontierCode is the only one: its Main and
# Extended boards are two rankings of one run, thirteen points apart, that
# Cognition tells readers not to mix.
#
# base key -> subset name -> the base that subset's columns are spelled with.
# Main keeps the plain base, so nothing already stored moves.
SUBSET_BASES: dict[str, dict[str, str]] = {
    "frontiercode": {"main": "frontiercode", "extended": "frontiercode_extended"},
}


def revision_rank(name: str) -> tuple[int, ...]:
    """Sort key ordering revisions oldest to newest by the numbers in their name.

    "v1" -> (1,), "v1_1" -> (1, 1), "v2" -> (2,) -- and a shorter tuple sorts
    before its own extensions, so v1 < v1_1 < v1_2 < v2. A name carrying no
    digits cannot be placed among them and is treated as the oldest, so an
    unexpected name never silently outranks a real revision.
    """
    numbers = tuple(int(n) for n in _REVISION_NUM_RE.findall(name))
    return numbers or (-1,)


def revision_label(name: str) -> str:
    """The label a revision is known by, from whatever spelling a source uses.

    The sources spell the same revision several ways -- Cognition's payload
    key "v1_1", DeepSWE's artifact directory "v1.1", SWE-Marathon's "v1.0",
    FrontierSWE's "v1" -- and all of them reduce to the digits, with a bare
    major filled out to ".0":

        "v1_1" -> "1.1"    "v1.1" -> "1.1"    "1.1" -> "1.1"
        "v1"   -> "1.0"    "v1.0" -> "1.0"

    A name with no digits is returned unchanged; it will not match any column,
    which is the point.
    """
    numbers = _REVISION_NUM_RE.findall(name)
    if not numbers:
        return name
    major, *rest = numbers
    return ".".join([major, *(rest or ["0"])])


def revision_key(base: str, label: str) -> str:
    """The llm.json benchmark key for one revision of a benchmark.

    ("deepswe", "1.1") -> "deepswe_1_1", matching the spelling
    ``terminal_bench_2_1`` already set for a versioned column.
    """
    return f"{base}_{revision_label(label).replace('.', '_')}"


def subset_base(base: str, subset: str | None) -> str | None:
    """The base key one task subset's columns are spelled with, or None for none.

    A benchmark that scores a single task set is returned unchanged whatever the
    row says, because its rows carry no subset to honour. For one that scores
    several, this is the same "do not write" answer known_revision_key() gives:
    a row naming no subset, or naming a subset this table does not track, gets
    None rather than being folded into the headline board's column -- a third
    FrontierCode task set would be refused, not rounded to Main.
    """
    subsets = SUBSET_BASES.get(base)
    if subsets is None:
        return base
    if not subset:
        return None
    return subsets.get(subset)


def known_revision_key(base: str, label: str | None) -> str | None:
    """The column for this revision, or None when llm.json carries no such column.

    None is the "do not write" answer, and it covers both halves of the rule
    above: a source that named no revision (label is None) and one that named a
    revision this table does not track.
    """
    if not label:
        return None
    normalized = revision_label(label)
    if normalized not in KNOWN_REVISIONS.get(base, ()):
        return None
    return revision_key(base, normalized)
