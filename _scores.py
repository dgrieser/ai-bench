"""Shared helpers for benchmark score bookkeeping."""

from __future__ import annotations

from datetime import date
from typing import Any


def stamp_score_updated(model: dict[str, Any], key: str) -> None:
    """Record today's date as the last-updated date for one benchmark score.

    The date is computed at call time so a long-running process does not stamp
    a stale date after crossing a midnight boundary.

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
    updated[key] = date.today().isoformat()
