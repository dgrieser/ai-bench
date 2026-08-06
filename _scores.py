"""Shared helpers for benchmark score bookkeeping."""

from __future__ import annotations

from datetime import date
from typing import Any


def editable_benchmarks(doc: dict[str, Any]) -> dict[str, Any]:
    """Benchmarks whose score can come from a source or a human.

    A benchmark flagged "derived" in llm.json is computed from other columns
    instead (derive_coding_index.py writes the Coding index), so a value typed
    in or mapped onto it would be recomputed away on the next run. Derived keys
    therefore get no edit.py flag, no prompt, never count as missing, and are
    not offered as a mapping target.
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
