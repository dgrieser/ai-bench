"""Ranking changes on the derived indexes, for llm.json.

``scores_history`` deliberately leaves the derived index columns out: their
values move whenever anything in the table moves, so a value history would be
arithmetic catching up. What a reader *does* want to know about an index is
the place a model holds on it -- that it entered the OpenBench Index at #4,
that it climbed from #9 to #6 on Coding, that it dropped out of Trust. That is
what this module records, in llm.json's top-level ``index_rankings``::

    "index_rankings": {
      "since": "2026-10-02",
      "ranks": { "coding_index": { "model-a": 1, "model-b": 2, ... }, ... },
      "changes": [
        { "date": "2026-10-14", "index": "coding_index", "model": "model-c",
          "from": null, "to": 3, "of": 118 },
        { "date": "2026-10-14", "index": "coding_index", "model": "model-a",
          "from": 1, "to": 2, "of": 118 },
        ...
      ]
    }

The rules, all of them enforced by :func:`sync` and pinned by
test_rankings.py:

* **The indexes are the derived columns** -- every benchmark llm.json declares
  ``"derived": true`` (the OpenBench Index and its Coding, Tooling, Knowledge,
  Vision, Trust and Security indexes). Higher is better on all of them.
* **A rank is a place in the whole ranked field**, closed reference rows
  included, because that is the field derive_indexes.py ranks over. Models
  sharing a value share a place (1, 2, 2, 4).
* **``ranks`` is the ranking as of the last sync**; it is what the next sync
  diffs against. It is a cache of the index columns, not a second copy of the
  truth: sync always rebuilds it from ``scores``.
* **The kind of change is derived, not stored.** ``from: null`` is a model
  entering the index, ``to: null`` is a model leaving it (no longer ranked, or
  gone from the file), and both set is a move. ``of`` is how many models the
  index ranked when the change was recorded (``to``'s field, or ``from``'s
  for a model leaving).
* **A move is a change of order, not of number.** A model entering at #1
  pushes everyone below it down a place, and recording that as two hundred
  moves would bury the one thing that happened. So a model counts as having
  moved only when its place among the models ranked both before and after
  changed -- something overtook it, or it overtook something. Its ``from`` and
  ``to`` are still the real places, so a model that was overtaken while a
  newcomer also landed above it reads "#5 -> #7".
* **At most one change per model per index per day**, like the score history:
  the refresh runs every three hours, and the day's net effect is what the day
  meant. A second change on a day already carrying one is folded into it --
  ``from`` stays the day's first place, ``to`` becomes its last -- and a pair
  that cancels out (entered and left, or #4 -> #5 -> #4) leaves nothing.
* **The open field is recorded beside the whole one.** The site hides the
  closed reference rows by default, and a place counted over a field the
  reader cannot see is a place they cannot check. So every change to an open
  model also carries ``open``: ``{"from", "to", "of"}`` counted over the open
  models only (ties kept, as above), and ``"quiet": true`` when, among the open
  models, nothing changed order -- the model only overtook, or was overtaken
  by, a reference row. A change to a reference model carries no ``open``.
  Which models are reference rows is read off their ``"reference"`` flag.
* **An index the previous ranking did not carry starts silently.** A column
  that is new to the file would otherwise report every model as entering it.
* **Kept for ``KEEP_DAYS`` days.** The panel on the site reads the recent
  past; the full record of every score is ``scores_history``'s job.

Every writer of the index columns calls :func:`sync` -- derive_indexes.py's
``refresh()`` (which update.py, edit.py, prune.py and the admin answers go
through) and its own ``--write`` path, which the nightly calibration runs.
backfill_rankings.py replays llm.json's git history through this same
function, so a backfilled change and a live one are the same thing.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

FIELD = "rankings"
ROOT_FIELD = "index_rankings"

# How long a recorded change is kept. The site's panel shows the last 3 days by
# default and the rest behind its "show all" toggle. A month of changes is about
# 600 kB of llm.json (pretty-printed), so the record is kept short rather than
# letting it outgrow the table it describes.
KEEP_DAYS = 60


def index_keys(doc: dict[str, Any]) -> list[str]:
    """The derived columns, in the order llm.json declares them."""
    benchmarks = doc.get("benchmarks")
    if not isinstance(benchmarks, dict):
        return []
    return [
        key
        for key, benchmark in benchmarks.items()
        if isinstance(benchmark, dict) and benchmark.get("derived") is True
    ]


def _value(model: dict[str, Any], key: str) -> float | None:
    scores = model.get("scores")
    if not isinstance(scores, dict):
        return None
    value = scores.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def ranks(doc: dict[str, Any], key: str) -> dict[str, int]:
    """model name -> place on index `key`, highest value first; ties share a
    place (standard competition ranking). Unranked models are absent."""
    valued = []
    for model in doc.get("models") or []:
        name = model.get("name") if isinstance(model, dict) else None
        if not isinstance(name, str):
            continue
        value = _value(model, key)
        if value is not None:
            valued.append((value, name))
    valued.sort(key=lambda pair: (-pair[0], pair[1]))
    placed: dict[str, int] = {}
    previous: float | None = None
    place = 0
    for position, (value, name) in enumerate(valued, start=1):
        if value != previous:
            place = position
            previous = value
        placed[name] = place
    return placed


def current(doc: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Every derived index's ranking, as ``ranks`` stores it."""
    return {key: ranks(doc, key) for key in index_keys(doc)}


def _relative(placed: dict[str, int], common: set[str]) -> dict[str, int]:
    """The places of `common` among themselves only, ties kept as ties."""
    order = sorted((placed[name], name) for name in common)
    relative: dict[str, int] = {}
    previous: int | None = None
    place = 0
    for position, (rank, name) in enumerate(order, start=1):
        if rank != previous:
            place = position
            previous = rank
        relative[name] = place
    return relative


def diff(
    before: dict[str, int], after: dict[str, int]
) -> list[tuple[str, int | None, int | None]]:
    """(model, from, to) for every model that entered, left or changed order
    between two rankings of one index. Pure."""
    common = set(before) & set(after)
    was = _relative(before, common)
    now = _relative(after, common)
    changes: list[tuple[str, int | None, int | None]] = []
    for name in sorted(set(before) | set(after)):
        if name not in before:
            changes.append((name, None, after[name]))
        elif name not in after:
            changes.append((name, before[name], None))
        elif was[name] != now[name]:
            changes.append((name, before[name], after[name]))
    return changes


def _fold(
    log: list[dict[str, Any]], change: dict[str, Any]
) -> list[dict[str, Any]]:
    """`log` with `change` added, folded into the same model's change on the
    same day and index if there is one. Pure."""
    for position, entry in enumerate(log):
        if (
            entry.get("date") == change["date"]
            and entry.get("index") == change["index"]
            and entry.get("model") == change["model"]
        ):
            merged = {**change, "from": entry.get("from")}
            if merged["from"] == merged["to"]:
                # Entered and left, or moved and moved back: nobody saw it.
                return log[:position] + log[position + 1:]
            was_open = entry.get("open")
            now_open = change.get("open")
            if isinstance(was_open, dict) and isinstance(now_open, dict):
                merged["open"] = _open_entry(
                    was_open.get("from"),
                    now_open.get("to"),
                    now_open.get("of"),
                    bool(was_open.get("quiet")) and bool(now_open.get("quiet")),
                )
            return log[:position] + [merged] + log[position + 1:]
    return [*log, change]


def _open_entry(
    was: int | None, now: int | None, of: int | None, quiet: bool
) -> dict[str, Any]:
    """The ``open`` part of a change. A model whose open place ends the day
    where it started has nothing to report among the open models."""
    entry: dict[str, Any] = {"from": was, "to": now, "of": of}
    if quiet or was == now:
        entry["quiet"] = True
    return entry


def reference_names(doc: dict[str, Any]) -> set[str]:
    """The models llm.json flags ``"reference": true``."""
    return {
        model["name"]
        for model in doc.get("models") or []
        if isinstance(model, dict)
        and isinstance(model.get("name"), str)
        and model.get("reference") is True
    }


def _state(doc: dict[str, Any]) -> dict[str, Any]:
    stored = doc.get(ROOT_FIELD)
    return stored if isinstance(stored, dict) else {}


def changes(doc: dict[str, Any]) -> list[dict[str, Any]]:
    stored = _state(doc).get("changes")
    return [entry for entry in stored if isinstance(entry, dict)] if isinstance(stored, list) else []


def _place(doc: dict[str, Any], value: dict[str, Any]) -> None:
    """Set the field, keeping it beside ``history_since`` (the other top-level
    record of how the table got where it is) when it is new to the file."""
    if ROOT_FIELD in doc:
        doc[ROOT_FIELD] = value
        return
    items = list(doc.items())
    doc.clear()
    for field, existing in items:
        doc[field] = existing
        if field == "history_since":
            doc[ROOT_FIELD] = value
    if ROOT_FIELD not in doc:
        doc[ROOT_FIELD] = value


def sync(
    doc: dict[str, Any],
    today: str | None = None,
    reference: set[str] | None = None,
) -> int:
    """Record how the index rankings moved since the last sync, and store the
    new ranking. Returns the number of changes recorded (before folding).

    `today` overrides the date changes are filed under, which is what lets
    backfill_rankings.py replay llm.json's git revisions through this same
    function.

    `reference` overrides which models are closed reference rows (default:
    the ones the doc flags); the backfill passes the current set, since an
    old revision may predate the flag.
    """
    now = today or date.today().isoformat()
    closed = reference_names(doc) if reference is None else set(reference)
    state = _state(doc)
    stored = state.get("ranks") if isinstance(state.get("ranks"), dict) else {}
    after = current(doc)
    log = changes(doc)

    recorded = 0
    for key, placed in after.items():
        before = stored.get(key)
        if not isinstance(before, dict):
            # A column new to the file starts silently; see the module notes.
            continue
        before = {
            name: rank
            for name, rank in before.items()
            if isinstance(name, str) and isinstance(rank, int) and not isinstance(rank, bool)
        }
        open_before = {name: rank for name, rank in before.items() if name not in closed}
        open_after = {name: rank for name, rank in placed.items() if name not in closed}
        open_was = _relative(before, set(open_before))
        open_now = _relative(placed, set(open_after))
        open_moved = {name for name, _, _ in diff(open_before, open_after)}
        for name, was, now_rank in diff(before, placed):
            change: dict[str, Any] = {
                "date": now,
                "index": key,
                "model": name,
                "from": was,
                "to": now_rank,
                "of": len(placed) if now_rank is not None else len(before),
            }
            if name not in closed:
                change["open"] = _open_entry(
                    open_was.get(name),
                    open_now.get(name),
                    len(open_now) if now_rank is not None else len(open_was),
                    name not in open_moved,
                )
            log = _fold(log, change)
            recorded += 1
        # A change already filed today says where the model ended the day, so
        # a later shift that changed nobody's order (a newcomer landing above
        # it) still moves its `to` -- and a model shifted back to where it
        # started the day has nothing left to report.
        refreshed: list[dict[str, Any]] = []
        for entry in log:
            if (
                entry.get("date") == now
                and entry.get("index") == key
                and entry.get("to") is not None
                and entry.get("model") in placed
            ):
                entry = {**entry, "to": placed[entry["model"]], "of": len(placed)}
                if entry["to"] == entry.get("from"):
                    continue
                if isinstance(entry.get("open"), dict) and entry["model"] in open_now:
                    entry["open"] = _open_entry(
                        entry["open"].get("from"),
                        open_now[entry["model"]],
                        len(open_now),
                        bool(entry["open"].get("quiet")),
                    )
            refreshed.append(entry)
        log = refreshed

    cutoff = (date.fromisoformat(now) - timedelta(days=KEEP_DAYS)).isoformat()
    log = [entry for entry in log if str(entry.get("date") or "") >= cutoff]
    # Newest day first, then by index in column order, then by where the model
    # landed -- the order the site reads them in.
    order = {key: position for position, key in enumerate(index_keys(doc))}
    log.sort(
        key=lambda entry: (
            _negate_date(entry.get("date")),
            order.get(entry.get("index"), len(order)),
            entry.get("to") if entry.get("to") is not None else 10**9,
            entry.get("from") if entry.get("from") is not None else 10**9,
            str(entry.get("model")),
        )
    )

    _place(
        doc,
        {
            "since": state.get("since") or now,
            "ranks": after,
            "changes": log,
        },
    )
    return recorded


def _negate_date(value: Any) -> tuple[int, ...]:
    """A sort key putting later ISO dates first."""
    text = str(value or "0000-00-00")
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        return (0,)
    return (-parsed.toordinal(),)


def rename(doc: dict[str, Any], old: str, new: str) -> None:
    """Carry a model rename through the stored ranking and its changes, so the
    next sync does not read it as one model leaving and another entering."""
    state = _state(doc)
    stored = state.get("ranks")
    if isinstance(stored, dict):
        for placed in stored.values():
            if isinstance(placed, dict) and old in placed:
                placed[new] = placed.pop(old)
    for entry in changes(doc):
        if entry.get("model") == old:
            entry["model"] = new


def validate(doc: dict[str, Any]) -> list[str]:
    """Problems with the stored record, as human-readable lines."""
    state = doc.get(ROOT_FIELD)
    if state is None:
        return []
    if not isinstance(state, dict):
        return [f"{ROOT_FIELD} is not an object"]
    problems: list[str] = []
    if state.get("ranks") != current(doc):
        problems.append(
            f"{ROOT_FIELD}.ranks does not match the index columns; "
            "run ./derive_indexes.py -w"
        )
    keys = set(index_keys(doc))
    seen: set[tuple[Any, Any, Any]] = set()
    for entry in state.get("changes") or []:
        if not isinstance(entry, dict):
            problems.append(f"{ROOT_FIELD}.changes holds a non-object entry")
            continue
        label = f"{entry.get('date')} {entry.get('index')} {entry.get('model')}"
        if entry.get("index") not in keys:
            problems.append(f"{label}: not a derived index")
        if not entry.get("date"):
            problems.append(f"{label}: no date")
        if entry.get("from") is None and entry.get("to") is None:
            problems.append(f"{label}: neither from nor to")
        if entry.get("from") == entry.get("to"):
            problems.append(f"{label}: from equals to")
        marker = (entry.get("date"), entry.get("index"), entry.get("model"))
        if marker in seen:
            problems.append(f"{label}: more than one change on the day")
        seen.add(marker)
    return problems
