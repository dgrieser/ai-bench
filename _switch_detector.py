"""Spot a benchmark whose scores moved together: what a silent re-versioning leaves.

A source that re-versions a benchmark in place -- a new task set, a new grader,
a new index formula -- does not say so on the rows it serves. The same field
just starts carrying different numbers. When everyone is re-run at once the
column stays one measurement and only its old values are lost; when only some
models are re-run, the column ends up holding two versions side by side. Both
have happened here: AA's Intelligence Index went to v4.2, v4.3 and v5.0 in
place (the last re-scoring 33 of 170 models), AA-Omniscience to v1.1, Real-SWE
to 1.1 on the same page.

Where a source states its version, the fetcher checks it (_revisions.py,
update.AA_VERSION_GATES, the boards' own version labels). This module is the
net under the sources that state nothing: it compares a column before and
after a refresh, and reports a column in which many values changed at once and
by a lot relative to the column's own spread -- the shape of a re-versioning,
not of the odd model being re-run. A report is a prompt to look, never an
action: a board that re-graded everything is reported the same way.

The thresholds were set by replaying llm.json's git history through
``detect`` (audit_switches.py): every AA index switch, the Omniscience and
Real-SWE switches and GDPval-AA's re-anchoring all clear them; ordinary daily
churn (an Elo drifting a few points, one model re-run) does not.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

# A column is reported when at least this many of its values changed in one
# refresh and they are at least this share of the values it held before ...
MIN_CHANGED = 5
MIN_SHARE = 0.15
# ... or, on a small board, when nearly all of it moved at once.
MIN_CHANGED_SMALL = 3
MIN_SHARE_SMALL = 0.5
# ... and the median change is at least this many times the column's median
# absolute deviation: a move against the column's own spread, so an Elo and a
# percentage are judged alike.
MIN_SHIFT = 0.25


@dataclass(frozen=True)
class Shift:
    key: str
    changed: int        # values that changed
    compared: int       # values present before and after
    share: float        # changed / compared
    shift: float        # median |change| / the column's median absolute deviation
    rising: float       # share of the changes that went up
    handover: float | None = None  # share of the changes that also changed source

    @property
    def same_source(self) -> bool:
        """Whether the moved values mostly kept their source: the source itself
        re-scored them, which is what a re-versioning looks like. Otherwise a
        different source took the cells over -- a change in this table, not at
        the source."""
        return self.handover is None or self.handover < 0.5

    def describe(self) -> str:
        direction = (
            "mostly up" if self.rising >= 0.8 else "mostly down" if self.rising <= 0.2 else "both ways"
        )
        moved = (
            f"{self.key}: {self.changed} of {self.compared} scores moved at once "
            f"({self.share:.0%}, {direction}, median move {self.shift:.2f}x the column's spread)"
        )
        if self.same_source:
            return moved + (
                " under the same source -- a possible silent version switch there; "
                "check before trusting the column"
            )
        return moved + f" as {self.handover:.0%} of them changed source -- a source handover, not a re-score"


def _numbers(values: Iterable[Any]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]


def column_shift(
    key: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    sources_before: Mapping[str, Any] | None = None,
    sources_after: Mapping[str, Any] | None = None,
) -> Shift | None:
    """The shift of one column between two {model: score} maps, if it is one.

    With the {model: source} maps beside them, the share of moved values whose
    source changed too is reported, telling a source's re-score from another
    source taking the cells over."""
    held = _numbers(before.values())
    pairs = [
        (float(before[m]), float(after[m]))
        for m in after
        if m in before and _numbers([before[m]]) and _numbers([after[m]])
    ]
    changed = [(old, new) for old, new in pairs if old != new]
    if len(held) < MIN_CHANGED_SMALL or not changed:
        return None
    share = len(changed) / len(pairs)
    wide = len(changed) >= MIN_CHANGED and share >= MIN_SHARE
    small = len(changed) >= MIN_CHANGED_SMALL and share >= MIN_SHARE_SMALL
    if not (wide or small):
        return None
    centre = statistics.median(held)
    spread = statistics.median(abs(v - centre) for v in held) or (max(held) - min(held)) / 4
    if not spread:
        return None
    shift = statistics.median(abs(new - old) for old, new in changed) / spread
    if shift < MIN_SHIFT:
        return None
    rising = sum(1 for old, new in changed if new > old) / len(changed)
    handover = None
    if sources_before is not None and sources_after is not None:
        moved = [
            m for m in after
            if m in before and _numbers([before[m]]) and _numbers([after[m]])
            and float(before[m]) != float(after[m])
        ]
        handover = round(
            sum(1 for m in moved if sources_before.get(m) != sources_after.get(m)) / len(moved), 2
        )
    return Shift(key, len(changed), len(pairs), round(share, 2), round(shift, 2), round(rising, 2), handover)


def detect(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    skip: Iterable[str] = (),
    sources_before: Mapping[str, Mapping[str, Any]] | None = None,
    sources_after: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Shift]:
    """Shifts between two {model: {column: score}} states, strongest first.

    ``skip`` names columns not to judge -- the derived indexes, which move
    whenever any input does. The optional {model: {column: source}} states let
    each shift say whether the moved values changed source.
    """
    skipped = set(skip)
    keys = {k for scores in after.values() for k in scores} - skipped
    found: list[Shift] = []
    for key in sorted(keys):
        col_before = {m: s[key] for m, s in before.items() if key in s and s[key] is not None}
        col_after = {m: s[key] for m, s in after.items() if key in s and s[key] is not None}
        src_before = src_after = None
        if sources_before is not None and sources_after is not None:
            src_before = {m: (s or {}).get(key) for m, s in sources_before.items()}
            src_after = {m: (s or {}).get(key) for m, s in sources_after.items()}
        result = column_shift(key, col_before, col_after, src_before, src_after)
        if result is not None:
            found.append(result)
    found.sort(key=lambda s: -(s.share * s.shift))
    return found


def doc_scores(doc: Mapping[str, Any], field: str = "scores") -> dict[str, dict[str, Any]]:
    """{model: scores} (or another per-model map, such as scores_source) out of
    an llm.json document."""
    return {
        m["name"]: dict(m.get(field) or {})
        for m in doc.get("models") or []
        if isinstance(m, dict) and isinstance(m.get("name"), str)
    }


def derived_keys(doc: Mapping[str, Any]) -> set[str]:
    return {k for k, b in (doc.get("benchmarks") or {}).items() if isinstance(b, dict) and b.get("derived")}
