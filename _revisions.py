"""Benchmark revisions: their labels, their order, and the column each feeds.

Three benchmarks in this table have published more than one revision of
themselves, and in every case the revisions are *not* comparable -- a re-run
changes the task set, the verification or the scoring, so a 1.0 number and a
1.1 number are two different measurements that happen to share a name:

  * **DeepSWE** -- deepswe.datacurve.ai serves one JSON artifact per revision
    and toggles between them. Re-runs moved models by more than 50 points
    (DeepSeek V4 Pro: 7.5 -> 62.8), and 1.0 is the only revision that ever
    scored a dozen models retired before the re-run.
  * **FrontierCode** -- Cognition's payload carries one block per revision
    ("v1_1", "v1"), the current one covering only what was re-run.
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
    "swe_marathon": ("1.0", "1.1"),
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

    The three sources spell the same revision three ways -- Cognition's payload
    key "v1_1", DeepSWE's artifact directory "v1.1", SWE-Marathon's "v1.0" --
    and all of them reduce to the digits, with a bare major filled out to ".0":

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
