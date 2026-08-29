"""Shared helpers for benchmark score bookkeeping."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

# Digits llm.html and llm-cli print for a benchmark without its own "decimals".
DEFAULT_SCORE_DECIMALS = 1


def score_decimals(doc: dict[str, Any], key: str) -> int:
    """Printed digits for one benchmark: benchmarks[key]["decimals"], else 1.

    The same rule llm.html and llm-cli apply, kept here so the writers can round
    a score to the precision the readers will show.
    """
    benchmark = (doc.get("benchmarks") or {}).get(key)
    if not isinstance(benchmark, dict):
        return DEFAULT_SCORE_DECIMALS
    decimals = benchmark.get("decimals")
    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
        return DEFAULT_SCORE_DECIMALS
    return decimals


def score_step(doc: dict[str, Any], key: str) -> Decimal:
    """The grid a stored score is rounded onto.

    One printed digit by default, so a stored value never carries precision the
    site does not show. A benchmark whose numbers move by more than that on
    their own -- an Elo that re-anchors as the pool grows, say -- sets
    "round_to" in llm.json to the smallest step worth recording, and its scores
    land on multiples of that instead.
    """
    benchmark = (doc.get("benchmarks") or {}).get(key)
    if isinstance(benchmark, dict):
        round_to = benchmark.get("round_to")
        if not isinstance(round_to, bool) and isinstance(round_to, (int, float)) and round_to > 0:
            return Decimal(str(round_to))
    return Decimal(1).scaleb(-score_decimals(doc, key))


def round_score(doc: dict[str, Any], key: str, value: Any) -> int | float | None:
    """Quantize one benchmark score onto its grid; None passes through.

    Every writer rounds here so a score means the same thing whichever source
    produced it: without it the scrapers that hand back a raw leaderboard number
    keep rewriting a value the site rounds to the very same digits, restamping
    its date for a change no reader can see. Halves round away from zero rather
    than to even, so the same input always yields the same output regardless of
    which side of the grid it fell on. An integral result is stored as an int,
    the shape the rest of llm.json already uses.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"score {key!r} is {type(value).__name__}, not a number")

    step = score_step(doc, key)
    steps = (Decimal(str(value)) / step).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    rounded = steps * step
    if rounded == rounded.to_integral_value():
        return int(rounded)
    return float(rounded)


def editable_benchmarks(doc: dict[str, Any]) -> dict[str, Any]:
    """Benchmarks whose score can come from a source or a human.

    A benchmark flagged "derived" in llm.json is computed from other columns
    instead (derive_indexes.py writes the Coding, Tooling, Knowledge, Vision
    and Trust indexes), so a value in or mapped onto it would be recomputed away
    on the next run. Derived keys therefore get no edit.py flag, no prompt, never count
    as missing, and are not offered as a mapping target.
    """
    benchmarks = doc.get("benchmarks")
    if not isinstance(benchmarks, dict):
        return {}
    return {
        key: benchmark
        for key, benchmark in benchmarks.items()
        if not (isinstance(benchmark, dict) and benchmark.get("derived") is True)
    }


def stamp_score_updated(model: dict[str, Any], key: str, when: str | None = None) -> None:
    """Record the last-updated date for one benchmark score, today by default.

    The default is computed at call time so a long-running process does not
    stamp a stale date after crossing a midnight boundary. `when` (an ISO date
    string) is for backfilling the date a score was actually published, which is
    not today -- see fill_missing_source_urls.py.

    Raises TypeError if the model carries a non-dict "scores_updated" so
    corrupt data surfaces loudly instead of scores being written with the
    date silently dropped.
    """
    updated = model.setdefault("scores_updated", {})
    if not isinstance(updated, dict):
        name = model.get("name", "<unknown>")
        raise TypeError(
            f"model {name!r} has a non-dict 'scores_updated' "
            f"({type(updated).__name__}); cannot stamp {key!r}"
        )
    updated[key] = when if when is not None else date.today().isoformat()


def stamp_score_source(model: dict[str, Any], key: str, url: str | None) -> None:
    """Record the page one benchmark score was read from.

    None is a valid value: a hand edit (edit.py) has no source page, so it
    clears whatever attribution the previous automated write left behind.

    Raises TypeError if the model carries a non-dict "scores_source" so
    corrupt data surfaces loudly instead of scores being written with the
    source silently dropped.
    """
    sources = model.setdefault("scores_source", {})
    if not isinstance(sources, dict):
        name = model.get("name", "<unknown>")
        raise TypeError(
            f"model {name!r} has a non-dict 'scores_source' "
            f"({type(sources).__name__}); cannot stamp {key!r}"
        )
    sources[key] = url
