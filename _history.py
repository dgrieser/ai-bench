"""Per-score change history for llm.json.

``scores``/``scores_updated``/``scores_source`` record only the last write: a
score that moves overwrites its number and restamps its date, and what it was
before is gone. ``scores_history`` keeps that past -- one list per benchmark
key, oldest entry first::

    "scores_history": {
      "browsecomp": [
        { "date": "2026-03-18", "score": 69.0, "source": "https://llm-stats.com/..." },
        { "date": "2026-09-15", "score": 51.6, "source": "https://huggingface.co/..." }
      ]
    }

The rules the readers rely on, all of them enforced by :func:`sync`:

* **The last entry mirrors the current value** -- ``scores[key]``,
  ``scores_updated[key]`` and ``scores_source[key]`` are that entry's ``score``,
  ``date`` and ``source``. The history can therefore never disagree with the
  table, and a reader that wants "now" can read either.
* **``score: null`` means the score was withdrawn.** Keeping the removal in the
  list is what lets the mirror rule hold for a score that has gone away, and it
  is the only way the site can say a column emptied rather than silently
  dropping the row.
* **The kind of change is derived, not stored.** The first entry of a list, or
  one whose predecessor is ``null``, is an addition; an entry that is ``null``
  is a removal; anything else is an update, and the entry before it carries
  what the score was. Nothing redundant is written down, so nothing can rot.
* **At most one entry per key per day.** Dates are day-granular and the refresh
  cron runs every three hours, so a second write on a date the list already
  carries replaces that day's entry instead of stacking a duplicate beside it.
  A score added and withdrawn on the same day leaves nothing behind, which is
  the truth: nobody ever saw it.
* **Derived index columns are not tracked.** Coding/Tooling/Knowledge/Vision/
  Trust are recomputed from the other columns on every run, so their history
  would be arithmetic catching up rather than anything measured -- the same
  reason llm.html's freshness marking skips them.
* **The map is sparse.** Unlike the three full-key maps, a benchmark appears
  only once it has held a value, which is what keeps the file from doubling.

llm.json's top-level ``history_since`` says which day the history starts. The
entries before it are what the file said on that day -- one per score, dated by
the stamp it was carrying -- so a score whose first entry predates it was
*first recorded* then, not necessarily added then: the write it records may have
replaced a number nobody kept. From that day on, an entry with nothing before it
really is a score arriving. backfill_history.py writes the field once; nothing
else touches it.

:func:`sync` reconciles the history against the current score maps rather than
being called at each write, so every writer -- update.py's ingests, edit.py's
hand edits, the source-URL backfills -- records history by the same rules
without having to remember to. Writers call it just before they write the file.
"""

from __future__ import annotations

from datetime import date
from typing import Any

HISTORY_FIELD = "scores_history"

# Top-level: the first day the history was recorded from. See the module
# docstring -- it is what keeps the site from calling a seeded entry an
# addition when all that is known about it is the day it was last written.
SINCE_FIELD = "history_since"

# Where the map is placed in a model object, so a hand-read llm.json keeps the
# score maps together and in the order they build on each other.
_AFTER_FIELD = "scores_source"


def derived_keys(doc: dict[str, Any]) -> set[str]:
    """Benchmark keys computed from the other columns, which carry no history."""
    benchmarks = doc.get("benchmarks")
    if not isinstance(benchmarks, dict):
        return set()
    return {
        key
        for key, benchmark in benchmarks.items()
        if isinstance(benchmark, dict) and benchmark.get("derived") is True
    }


def history(model: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """One model's history map, or an empty one; never raises on bad data."""
    stored = model.get(HISTORY_FIELD)
    if not isinstance(stored, dict):
        return {}
    return {
        key: entries
        for key, entries in stored.items()
        if isinstance(entries, list) and entries
    }


def entries(model: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """The history of one score, oldest first; empty when nothing is recorded."""
    return history(model).get(key, [])


def _entry(when: str, score: Any, source: Any) -> dict[str, Any]:
    return {"date": when, "score": score, "source": source}


def _place(model: dict[str, Any], value: dict[str, Any]) -> None:
    """Set the history map, keeping it just after ``scores_source``.

    Mirrors update.ensure_scores_source: a model that already carries the field
    keeps its position, and a model meeting it for the first time gets it
    beside the maps it is about, not appended after the VRAM estimates.
    """
    if HISTORY_FIELD in model:
        model[HISTORY_FIELD] = value
        return
    items = list(model.items())
    model.clear()
    for field, existing in items:
        model[field] = existing
        if field == _AFTER_FIELD:
            model[HISTORY_FIELD] = value
    if HISTORY_FIELD not in model:
        model[HISTORY_FIELD] = value


def _record(
    log: list[dict[str, Any]],
    value: Any,
    when: str,
    source: Any,
) -> list[dict[str, Any]]:
    """One score's history after a write of `value` on `when`. Pure."""
    if not log:
        return [_entry(when, value, source)]

    last = log[-1]
    if last.get("score") == value:
        # Not a change: a re-read of the same number, or a backfill putting a
        # date or a source page on a score that was missing one. The entry is
        # corrected in place rather than a second entry claiming the score
        # moved when it did not.
        return [*log[:-1], _entry(when, value, source)]

    if last.get("date") != when:
        return [*log, _entry(when, value, source)]

    # A second write on a day the list already carries. The day's final value
    # is what that day meant, so it replaces the entry -- except where the
    # value it replaces was itself first recorded today and is now being
    # withdrawn, which leaves nothing worth keeping.
    if value is None and (len(log) == 1 or log[-2].get("score") is None):
        return log[:-1]
    return [*log[:-1], _entry(when, value, source)]


def sync(doc: dict[str, Any], today: str | None = None) -> int:
    """Reconcile every model's history with its current scores.

    Returns the number of models whose history changed. `today` overrides the
    date a removal is stamped with, which is what lets backfill_history.py
    replay the file's git revisions through these very rules instead of a
    second copy of them that could drift.
    """
    now = today or date.today().isoformat()
    derived = derived_keys(doc)
    benchmarks = doc.get("benchmarks")
    known = list(benchmarks) if isinstance(benchmarks, dict) else []
    changed = 0

    for model in doc.get("models") or []:
        scores = model.get("scores")
        if not isinstance(scores, dict):
            continue
        updated = model.get("scores_updated")
        updated = updated if isinstance(updated, dict) else {}
        sources = model.get("scores_source")
        sources = sources if isinstance(sources, dict) else {}

        before = model.get(HISTORY_FIELD)
        logs = dict(history(model))

        for key, value in scores.items():
            if key in derived:
                logs.pop(key, None)
                continue
            log = logs.get(key) or []

            if value is None:
                # A removal clears the date along with the score, so there is
                # no stamp to read: it is recorded as happening now. Recorded
                # once -- a run that finds the removal already in the list
                # leaves its date alone rather than walking it forward.
                if not log or log[-1].get("score") is None:
                    continue
                log = _record(log, None, now, None)
            else:
                # A score with no date of its own is filed under the day the
                # model arrived, which is the earliest day it could have been
                # read; fill_missing_source_urls.py is what asks a human for
                # the real one, and the entry is corrected in place when they
                # answer.
                when = updated.get(key) or model.get("date_added") or now
                log = _record(log, value, when, sources.get(key))

            if log:
                logs[key] = log
            else:
                logs.pop(key, None)

        # A column that has left llm.json takes its history with it, the same
        # way it leaves the three score maps -- nothing on the site could name
        # it any more.
        order = [key for key in known if key in logs]
        order += [key for key in logs if key not in known and key in scores]
        after = {key: logs[key] for key in order}

        if after != before:
            changed += 1
        if after:
            _place(model, after)
        elif HISTORY_FIELD in model:
            del model[HISTORY_FIELD]

    return changed


def validate(doc: dict[str, Any]) -> list[str]:
    """Problems with the stored history, as human-readable lines.

    Checks the promises the site is written against: a list per benchmark key,
    entries carrying a date, and the last entry mirroring the current score.
    Used by sync_score_dates.py, which is where llm.json's cross-map invariants
    are already checked.
    """
    problems: list[str] = []
    derived = derived_keys(doc)

    for model in doc.get("models") or []:
        name = model.get("name", "?")
        stored = model.get(HISTORY_FIELD)
        if stored is None:
            continue
        if not isinstance(stored, dict):
            problems.append(f"{name}: {HISTORY_FIELD} is not an object")
            continue

        scores = model.get("scores") or {}
        updated = model.get("scores_updated") or {}
        sources = model.get("scores_source") or {}

        for key, log in stored.items():
            label = f"{name}.{key}"
            if key in derived:
                problems.append(f"{label}: derived columns carry no history")
                continue
            if not isinstance(log, list) or not log:
                problems.append(f"{label}: history is not a non-empty list")
                continue
            if any(not isinstance(entry, dict) for entry in log):
                problems.append(f"{label}: history holds a non-object entry")
                continue
            if any(not entry.get("date") for entry in log):
                problems.append(f"{label}: history holds an entry with no date")
                continue

            last = log[-1]
            if last.get("score") != scores.get(key):
                problems.append(
                    f"{label}: history ends at {last.get('score')!r}, "
                    f"score is {scores.get(key)!r}"
                )
                continue
            if last.get("score") is None:
                continue
            if last.get("date") != updated.get(key):
                problems.append(
                    f"{label}: history ends on {last.get('date')}, "
                    f"scores_updated says {updated.get(key)}"
                )
            if last.get("source") != sources.get(key):
                problems.append(f"{label}: history ends on a different source URL")

    return problems
