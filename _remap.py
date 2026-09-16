"""Re-point mapping entries at a model that has just been added.

A source name is asked about once. The answer goes in a mapping file and is
never revisited, which is what keeps the queue from re-asking a settled
question -- and what makes a mapping go quietly wrong when the model it should
have named turns up later. llm-stats publishes ``ling-3.0-flash-fin``; the
index had no Fin model, so the row was mapped onto ``ling-3-0-flash``, the
nearest thing; when ``ling-3-0-flash-fin`` was added months later nothing went
back to look. Two models' scores folded into one row and the next run turned
red, because test_propose.py replays every mapping through the matcher and
refuses a proposal that contradicts one.

So the add path asks the question the queue cannot: *does any mapping now name
the wrong model?* A source name that matches the new model by normalized
equality -- the same EXACT tier the queue is allowed to propose, and the only
tier that test holds mappings to -- and that names a *different model* is
re-pointed at it. That is the case the test forbids outright, so it is wrong by
construction and there is nothing to weigh.

Two things are therefore left alone and merely reported. A name that merely
looks similar is a judgement, and judgements stay with whoever made them. And a
row parked on a sentinel -- ``__unmappable__``, ``__pending__``,
``__closed_weights__`` -- is a decision about whether the row belongs here at
all, which is not the same question: SWE-Rebench maps its ``gpt-oss-120b-high``
row onto gpt-oss-120b and parks the plain one, and whether that was a way to
keep two rows from folding into one model or an oversight is not for this to
guess. Rewriting only what is provably wrong is what keeps a sweep that runs on
every add from quietly undoing answers.

Which files are searched is _rename.value_routes(), the same reading of
propose.ROUTES a rename already goes by, so the two cannot disagree about which
files hold model names and a source added to that table is covered the day it
is added (test_propose.py asserts the table covers every update_*_mapping.py).
The benchmark-name files are excluded by their universe, and llm.json's own
AA-slug overrides -- which map the other way, model name to slug -- by not
being a source route at all. Each file is written by its own ``add_*_mapping``
writer, so it keeps its own formatting, and stays frozen in collect mode like
every other mapping write.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import _matching
import _rename
import propose
from _openness import SENTINELS

# A parked row is an answer about whether the source row belongs in the index
# at all -- _openness.py's question, not this one's. Reported so a human can
# look, never rewritten.
_PARKED = SENTINELS


@dataclass(frozen=True)
class Stale:
    """One mapping entry that names something other than the matched model."""

    path: Path
    source_name: str
    was: str
    now: str

    def __str__(self) -> str:
        return f"{self.path.name}: {self.source_name!r} {self.was} -> {self.now}"


def model_routes() -> dict[Path, propose.Route]:
    """The mapping files whose values are llm.json model names, by path."""
    return {_rename.route_path(route): route for route in _rename.value_routes()}


def subject_of(route: propose.Route, source_name: str) -> str:
    """What of a mapping key is matched against a model name.

    Spheron keys are org/model paths and Vals keys provider/model ones, where
    the leading segment names where the weights were served rather than the
    model. Delegated to propose.match_subject so the add path and the queue
    cannot disagree about what a key says.
    """
    return propose.match_subject({"subject": source_name}, route)


def find_stale(doc: dict[str, Any], name: str) -> tuple[list[Stale], list[Stale]]:
    """(re-pointable, reportable) mapping entries for a newly added model.

    A mapping is re-pointable when its source name matches `name` by normalized
    equality against the whole model universe and it currently names a
    different model; a row parked on a sentinel is reportable instead. Matching
    against the whole universe rather than the new name alone is what makes
    ambiguity safe: two models normalizing to one string yield no proposal at
    all, so neither steals the other's rows.
    """
    names = [
        model["name"]
        for model in doc.get("models") or []
        if isinstance(model, dict) and isinstance(model.get("name"), str) and model["name"]
    ]
    if name not in names:
        return [], []

    # _matching.propose grades a name against every model in the index, which
    # is far too much work to do once per mapping entry. An EXACT match is
    # normalized-slug equality, so that is the cheap gate; propose() still has
    # the last word on the handful that pass it, and stays the only definition
    # of what "exact" means.
    target = _matching.normalize_slug(name)
    if sum(1 for other in names if _matching.normalize_slug(other) == target) != 1:
        # Another model normalizes the same way: propose() would call it
        # ambiguous and offer nothing, and neither may take the other's rows.
        return [], []

    repointable: list[Stale] = []
    reportable: list[Stale] = []
    for path, route in sorted(model_routes().items()):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        for source_name, mapped in raw.items():
            # A list value is the AA file's curated slug order, which is not a
            # model name; a non-string is not one either.
            if not isinstance(source_name, str) or not isinstance(mapped, str):
                continue
            if mapped == name:
                continue
            subject = subject_of(route, source_name)
            if _matching.normalize_slug(subject) != target:
                continue
            match, _ = _matching.propose(subject, names)
            if match is None or match.option != name:
                continue
            stale = Stale(path=path, source_name=source_name, was=mapped, now=name)
            (reportable if mapped in _PARKED else repointable).append(stale)
    return repointable, reportable


def repoint(doc: dict[str, Any], name: str, *, write: bool = True) -> tuple[list[Stale], list[Stale]]:
    """Re-point every stale mapping at `name`. Returns what was and was not moved.

    Writes through each file's own ``add_*_mapping``, which is also what keeps
    this frozen in collect mode: every mapping writer is a no-op there, so a
    collecting run reports what it would move and moves nothing.
    """
    repointable, reportable = find_stale(doc, name)
    if write:
        for stale in repointable:
            writer = _writer_for_path(stale.path)
            if writer is None:
                continue
            writer(stale.source_name, name)
    return repointable, reportable


def _writer_for_path(path: Path) -> Callable[[str, str], None] | None:
    route = model_routes().get(path)
    if route is None:
        return None
    return getattr(importlib.import_module(route.module), route.writer)
