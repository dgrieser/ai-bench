"""Sanity checks every fetcher runs on what it scraped, before reporting it.

A leaderboard that changes its layout rarely fails loudly on its own: a renamed
column reads as an empty cell on every row, a moved table as no table, a board
that switches from percentages to fractions as a field of models that all
regressed at once. Each of those used to leave a fetcher reporting `[]` or a
column of tiny numbers with exit status 0, which update.py then wrote -- or,
for `[]`, quietly wrote nothing, so the column simply stopped moving.

These helpers turn each of those into an exception naming the source, so the
fetcher exits non-zero and update.py reports that source as failed instead.
They are deliberately blunt: a check that fires on a real layout change is the
point; one that fires on a real leaderboard is a bug in the check.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def require_rows(rows: Sequence[Any], source: str, what: str = "leaderboard rows") -> None:
    """Raise when a parse produced nothing.

    No public leaderboard this repo reads is ever legitimately empty, so an
    empty parse means the page moved -- never "no models today".
    """
    if not rows:
        raise ValueError(
            f"No {what} parsed from {source} -- the page layout changed; "
            "refusing to report an empty leaderboard."
        )


def require_columns(header: Iterable[str], required: Iterable[str], source: str) -> None:
    """Raise when a table's header lacks a column the parse depends on.

    Without it a renamed header reads as an empty cell on every row, which the
    row loop skips one at a time and the result is indistinguishable from an
    empty board.
    """
    present = {str(name) for name in header if name is not None}
    missing = [name for name in required if name not in present]
    if missing:
        raise ValueError(
            f"{source} is missing the column(s) {', '.join(repr(m) for m in missing)} "
            f"(header: {sorted(present)}) -- the layout changed; refusing to guess."
        )


def check_fractions(values: Iterable[Any], source: str, field: str = "score") -> None:
    """Raise unless every number is a 0-1 fraction.

    For the fetchers that multiply by 100: a board that switched to
    percentages would otherwise be stored as 4,200%.
    """
    bad = [v for v in values if _is_number(v) and not 0 <= v <= 1]
    if bad:
        raise ValueError(
            f"{source}: {field} is expected as a 0-1 fraction but reads {bad[:5]} -- "
            "the source's scale changed; refusing to multiply it by 100."
        )


# A percentage board this large whose best run is at most 1.0 is a board that
# switched to fractions, not one where every model scores under one percent.
_FRACTION_SUSPECT_MIN_ROWS = 5


def check_percentages(
    rows: Iterable[Mapping[str, Any]],
    source: str,
    field: str = "score",
    low: float = 0.0,
    high: float = 100.0,
) -> None:
    """Raise unless `field` reads as a percentage on every row.

    Two failures: a value outside [low, high], and a whole board that fits in
    0-1 -- the fraction flip that the range check alone cannot see, because
    0.42 is a perfectly valid percentage for one model but not for all of them.
    """
    values = [row.get(field) for row in rows]
    numbers = [v for v in values if _is_number(v)]
    bad = [v for v in numbers if not low <= v <= high]
    if bad:
        raise ValueError(
            f"{source}: {field} is expected in [{low:g}, {high:g}] but reads {bad[:5]} -- "
            "the source's scale changed; refusing to report it."
        )
    if len(numbers) >= _FRACTION_SUSPECT_MIN_ROWS and max(numbers) <= 1.0:
        raise ValueError(
            f"{source}: every {field} on a {len(numbers)}-row board is at most 1.0 "
            f"(max {max(numbers)}) -- it reads like 0-1 fractions, not percentages; "
            "refusing to report it."
        )
