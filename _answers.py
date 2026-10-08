#!/usr/bin/env python3
"""Apply answers to the pipeline's questions without asking anybody.

`update-all --collect-prompts` queues every question it could not ask (see
_prompts.py) and propose.py turns that queue into a reviewable PR. This module
is the other way to answer it: a batch of records, validated hard and written
through the sources' own writers, so a person can answer from somewhere that
is not a terminal and not a pull request.

The records arrive from outside -- a web form, a workflow input, a file -- so
validation here is a trust boundary, not a convenience. Two rules follow from
that and are worth stating before the code:

  * Nothing from a record ever names a module, a writer or a path. A record
    carries a `route` *key*, looked up in propose.ROUTES, which is the only
    place a module name is allowed to come from. propose.py can import by name
    safely because that table is hard-coded; a wire-supplied module name would
    be a one-line import of anything.

  * __pending__ is refused everywhere. It is a parking marker rather than an
    answer (_openness.py), so writing it over a decision would undo the
    decision and put the question back in the loop for good.

A batch is all or nothing. Validation runs first and touches nothing, but
add.py and edit.py are subprocesses that can still fail halfway, so the files
a batch may write are snapshotted and rolled back if any record fails. Half an
answered queue with a non-zero exit is the one outcome nobody could debug.
"""

from __future__ import annotations

import fnmatch
import importlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import _new_models
import _prompts
import _reference
import _rejections
import _rename
import derive_indexes
import edit
import propose
from _openness import CLOSED_WEIGHTS, PENDING, SENTINELS, UNMAPPABLE
from _scores import editable_benchmarks

HERE = Path(__file__).resolve().parent
ADD_SCRIPT = HERE / "add.py"
EDIT_SCRIPT = HERE / "edit.py"
DEFAULT_LLM_JSON = HERE / "llm.json"

# The answers a person can give to a mapping question. __closed_weights__ is
# missing on purpose: it is the machine's verdict about a source's own claim
# (see _openness.is_closed_weights), and a person who means "do not track this"
# means __unmappable__.
HUMAN_SENTINELS = frozenset({UNMAPPABLE})

MAPPING = "mapping"
AA_IGNORE = "aa-ignore"
# "None of these": the candidates offered for a queued name are ruled out, and
# nothing else is recorded -- the name stays unanswered, drops out of the queue
# while nothing new matches it, and is asked again, with only the new
# candidates, once something does. Any route; see _rejections.py. AA_IGNORE is
# the older, Artificial-Analysis-only spelling of the same thing.
CANDIDATES_SKIP = "candidates-skip"
NEW_MODEL = "new-model"
MODEL_ADD = "model-add"
MODEL_CREATE = "model-create"
MODEL_EDIT = "model-edit"
MODEL_RENAME = "model-rename"
# The reference list (see _reference.py). Both kinds move reference-models.json
# *and* llm.json together, because the two are one decision: a slug on the list
# with no entry behind it is inert, and an entry left behind after its slug
# goes is a closed model the table shows as an open-weight one. Editing a
# reference model is not a third kind -- it is MODEL_RENAME, which already
# carries the name through every mapping file and the list with it.
REFERENCE_ADD = "reference-add"
REFERENCE_REMOVE = "reference-remove"
# The columns themselves: llm.json's `benchmarks` object, which says what each
# column is called, what run of its benchmark it holds and where to read about
# it. Neither kind names a fetcher -- a new column is filled by hand on the
# Models tab, or by a mapping onto it, until a scraper is written for it.
BENCHMARK_CREATE = "benchmark-create"
BENCHMARK_EDIT = "benchmark-edit"
KINDS = frozenset(
    {
        MAPPING,
        AA_IGNORE,
        CANDIDATES_SKIP,
        NEW_MODEL,
        MODEL_ADD,
        MODEL_CREATE,
        MODEL_EDIT,
        MODEL_RENAME,
        REFERENCE_ADD,
        REFERENCE_REMOVE,
        BENCHMARK_CREATE,
        BENCHMARK_EDIT,
    }
)

BENCHMARK_KINDS = frozenset({BENCHMARK_CREATE, BENCHMARK_EDIT})

# The kinds that act on one entry in llm.json. A batch may touch each entry
# once: the records are applied in order and rolled back together, so an edit
# that follows a rename of the same model looks for a name that is no longer
# there and takes every other answer down with it.
MODEL_KINDS = frozenset(
    {MODEL_ADD, MODEL_CREATE, MODEL_EDIT, MODEL_RENAME, REFERENCE_ADD, REFERENCE_REMOVE}
)

# The route whose mapping file runs the other way round: its keys are llm.json
# model names and its values are Artificial Analysis slugs, one or a list.
AA_ROUTE = "update_artificialanalysis_mapping.py"
AA_MODULE = "_artificialanalysis_mapping"

# check_new.py records its own command, so that is the route a new-model
# question appears under in the queue.
NEW_MODEL_ROUTE = "check_new.py"

# Enough to answer a sitting so far, small enough that a batch cannot evict its
# own successor from the workflow's concurrency group (one run may be running
# and one queued; a third cancels the queued one).
MAX_RECORDS = 25

# Read off edit.py rather than restated: it is the script that has a flag per
# field, so a list here could drift into asking for one that does not exist.
METADATA_FIELDS = tuple(sorted(edit.METADATA_FIELDS))

# A benchmark key: lowercase words joined by single underscores, like every key
# already in llm.json. It becomes edit.py's --flag (underscores to dashes) and a
# mapping file's value, so it is an identifier, not a label.
BENCHMARK_KEY_RE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
MAX_BENCHMARK_KEY = 48

# What a benchmark record may set, and how long each text may be. The four
# text fields and the three lists are what the site shows a reader; the four
# numeric ones say how a score is read and stored (see _scores.py). Nothing
# else is accepted: `derived` is derive_indexes.py's to declare, and icon_svg
# names a file in icons/ that a record cannot upload.
BENCHMARK_TEXT_FIELDS = {
    "name": 80,
    "short_name": 24,
    "category": 40,
    "description": 2000,
}
BENCHMARK_LIST_FIELDS = ("settings", "excludes", "urls")
BENCHMARK_NUMBER_FIELDS = ("decimals", "range", "round_to", "lower_is_better")
BENCHMARK_FIELDS = (*BENCHMARK_TEXT_FIELDS, *BENCHMARK_LIST_FIELDS, *BENCHMARK_NUMBER_FIELDS)
# Every column says these: test_benchmark_settings.py holds the file to it, and
# the admin queue shows them as the whole of what a mapping is decided on.
BENCHMARK_REQUIRED = ("name", "short_name", "category", "description", "settings", "urls")
# Long enough for "best of native / prompted", short enough to stay a chip --
# test_benchmark_settings.MAX_TAG.
MAX_SETTING_TAG = 30
MAX_LIST_ITEMS = 12
MAX_DECIMALS = 4


class AnswerError(Exception):
    """A record that must not be applied. The message is shown to the sender."""


@dataclass(frozen=True)
class Answer:
    """One validated record, ready to write."""

    index: int
    kind: str
    subject: str
    route: str | None = None
    route_kind: str | None = None
    value: Any = None
    fields: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, Any] = field(default_factory=dict)
    # Provenance for the scores in one edit: the date they were read and the
    # page they were read from. None each means edit.py's own default -- today,
    # credited to nobody.
    score_date: str | None = None
    score_url: str | None = None

    def describe(self) -> str:
        if self.kind == MAPPING:
            return f"{self.route}: {self.subject!r} -> {self.value!r}"
        if self.kind == CANDIDATES_SKIP:
            return f"{self.route}: {self.subject!r} is none of {', '.join(self.value)}"
        if self.kind == AA_IGNORE:
            return f"AA suggestions rejected for {self.subject!r}: {', '.join(self.value)}"
        if self.kind == NEW_MODEL:
            return f"new model {self.subject!r} -> {self.value}"
        if self.kind == MODEL_ADD:
            return f"add model {self.subject!r}"
        if self.kind == MODEL_CREATE:
            named = ", ".join(f"{k}={v!r}" for k, v in sorted(self.fields.items()))
            return f"create model {self.subject!r}" + (f" ({named})" if named else "")
        if self.kind == MODEL_RENAME:
            return f"rename model {self.subject!r} -> {self.value!r}"
        if self.kind == REFERENCE_ADD:
            return f"carry {self.subject!r} as a reference model"
        if self.kind == REFERENCE_REMOVE:
            return f"stop carrying {self.subject!r} as a reference model"
        if self.kind == BENCHMARK_CREATE:
            placed = f" after {self.value!r}" if self.value else ""
            return f"create benchmark {self.subject!r}{placed}"
        if self.kind == BENCHMARK_EDIT:
            return f"edit benchmark {self.subject!r}: {', '.join(sorted(self.fields))}"
        changes = sorted([*self.fields, *self.scores])
        described = f"edit model {self.subject!r}: {', '.join(changes)}"
        if self.score_date:
            described += f", dated {self.score_date}"
        if self.score_url:
            described += f", from {self.score_url}"
        return described


@dataclass(frozen=True)
class Failure:
    index: int
    message: str

    def __str__(self) -> str:
        return f"record {self.index}: {self.message}"


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------


def resolve_route(route: Any, route_kind: Any) -> propose.Route:
    """The Route a record names, or raise. Never touches the filesystem."""
    if not isinstance(route, str):
        raise AnswerError("'route' must be a string naming an update_*_mapping.py script")
    by_kind = propose.ROUTES.get(route)
    if by_kind is None:
        known = ", ".join(sorted(propose.ROUTES))
        raise AnswerError(f"unknown route {route!r}; expected one of: {known}")
    if route_kind is None:
        route_kind = "*"
    if not isinstance(route_kind, str):
        raise AnswerError("'route_kind' must be a string")
    resolved = by_kind.get(route_kind, by_kind.get("*"))
    if resolved is None:
        known = ", ".join(sorted(by_kind))
        raise AnswerError(
            f"route {route!r} has no kind {route_kind!r}; expected one of: {known}"
        )
    return resolved


def mapping_path(route: propose.Route) -> Path:
    """The file a Route owns, checked to be one of ours.

    The path comes from the hard-coded table, so this cannot fail today; it is
    here so that a future route pointing somewhere odd fails loudly instead of
    letting a caller write outside the repo.
    """
    module = importlib.import_module(route.module)
    path = getattr(module, route.mapping_const)
    if path.parent != HERE or not fnmatch.fnmatch(path.name, "*mapping*.json"):
        raise AnswerError(f"route {route.module} points outside the mapping files: {path}")
    return path


def writer_for(route: propose.Route):
    return getattr(importlib.import_module(route.module), route.writer)


def current_value(route: propose.Route, subject: str) -> list[str] | None:
    """What the mapping file says about subject now, as a list, or None.

    A list because the Artificial Analysis file's values may be one -- a curated
    priority order of slugs -- and propose.recorded_value returns None for those,
    which would make a staleness check silently blind to exactly the entries
    most worth checking.
    """
    path = mapping_path(route)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or subject not in raw:
        return None
    return as_list(raw[subject])


def as_list(value: Any) -> list[str] | None:
    if isinstance(value, str):
        return [value] if value else None
    if isinstance(value, list):
        items = [v for v in value if isinstance(v, str) and v]
        return items or None
    return None


# --------------------------------------------------------------------------
# the queue
# --------------------------------------------------------------------------


def load_queue(path: str | Path | None) -> set[tuple[str, str, str]]:
    """(route, kind, subject) for every question the pipeline actually asked.

    Reads either the raw JSONL _prompts.record() writes or the published
    pending.json. Requiring an answer's subject to appear here is what keeps a
    valid-looking batch from attaching scores to models nobody asked about: the
    answer space stops being "any string" and becomes "one of these questions".
    """
    if path is None:
        return set()
    path = Path(path)
    if not path.exists():
        raise AnswerError(f"queue file not found: {path}")

    # Both formats start with "{", so sniff by parsing rather than by first
    # character: a multi-line JSONL report is not one JSON document.
    entries: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8").strip()
    doc: Any = None
    if path.suffix != ".jsonl":
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            doc = None
    if isinstance(doc, dict) and "questions" in doc:
        entries = [e for e in doc["questions"] if isinstance(e, dict)]
    else:
        entries = _prompts.load(path)

    queue: set[tuple[str, str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        route = entry.get("route") or propose.script_of(entry)
        kind = entry.get("route_kind") or entry.get("kind") or ""
        subject = entry.get("subject") or ""
        if route and subject:
            queue.add((route, kind, subject))
    return queue


def in_queue(queue: set[tuple[str, str, str]], route: str, kind: str, subject: str) -> bool:
    # The queue records the prompt's own kind ("aa-mapping", "llmstats-model",
    # ...), which is also the route discriminator; "*" means the route asks one
    # kind of question, so any recorded kind for it matches.
    if (route, kind, subject) in queue:
        return True
    return any(r == route and s == subject for r, _k, s in queue)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def needs_aa_slugs(records: Sequence[dict[str, Any]]) -> bool:
    """True when some record needs the live Artificial Analysis slug list."""
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("kind") in (AA_IGNORE, REFERENCE_ADD):
            return True
        if record.get("kind") in (MAPPING, CANDIDATES_SKIP) and record.get("route") == AA_ROUTE:
            return True
    return False


def validate(
    records: Sequence[Any],
    *,
    llm_path: Path = DEFAULT_LLM_JSON,
    universes: dict[str, list[str]] | None = None,
    queue: set[tuple[str, str, str]] | None = None,
    require_queue: bool = True,
) -> tuple[list[Answer], list[Failure]]:
    """(answers, failures). Reads only -- nothing here writes."""
    if not isinstance(records, (list, tuple)):
        return [], [Failure(0, "expected a JSON array of records")]
    if not records:
        return [], [Failure(0, "no records to apply")]
    if len(records) > MAX_RECORDS:
        return [], [Failure(0, f"{len(records)} records; at most {MAX_RECORDS} per batch")]

    if universes is None:
        universes = propose.build_universes(llm_path, skip_aa=not needs_aa_slugs(records))
    doc = json.loads(llm_path.read_text(encoding="utf-8"))
    model_names = {m.get("name") for m in doc.get("models", []) if isinstance(m, dict)}
    benchmarks = editable_benchmarks(doc)
    queue = queue if queue is not None else set()
    created = batch_created_models(records, model_names)
    created_benchmarks = batch_created_benchmarks(records, doc)
    # A score may go to a column the same batch creates: apply() makes the
    # columns first, so a new benchmark and the numbers it was added for are
    # one sitting rather than two runs.
    scoreable = {**benchmarks, **{key: {} for key in created_benchmarks}}
    if created_benchmarks:
        universes = {
            **universes,
            propose.BENCHMARKS: sorted(
                {*(universes.get(propose.BENCHMARKS) or []), *created_benchmarks}
            ),
        }

    answers: list[Answer] = []
    failures: list[Failure] = []
    for index, record in enumerate(records):
        try:
            answers.append(
                _validate_one(
                    index,
                    record,
                    universes=universes,
                    model_names=model_names,
                    benchmarks=scoreable,
                    queue=queue,
                    require_queue=require_queue,
                    created=created,
                    doc=doc,
                )
            )
        except AnswerError as exc:
            failures.append(Failure(index, str(exc)))

    seen: set[tuple[str, str | None, str | None, str]] = set()
    reported: set[int] = set()
    for answer in answers:
        key = (answer.kind, answer.route, answer.route_kind, answer.subject)
        if key in seen:
            failures.append(Failure(answer.index, f"answered twice in one batch: {answer.subject!r}"))
            reported.add(answer.index)
        seen.add(key)

    # One entry, one record. The batch is applied in order and rolled back
    # together, so an edit sent alongside a rename of the same model would be
    # looking for a name that no longer exists -- and would take every other
    # answer in the sitting down with it. A rename's new name counts as touched
    # too: creating it in the same batch is the same collision seen from the
    # other end.
    touched: dict[str, str] = {}
    for answer in answers:
        if answer.kind not in MODEL_KINDS:
            continue
        names = [answer.subject] + ([answer.value] if answer.kind == MODEL_RENAME else [])
        for name in names:
            if name in touched and answer.index not in reported:
                failures.append(
                    Failure(
                        answer.index,
                        f"{name!r} is touched twice in one batch (also by a "
                        f"{touched[name]} record); send the second one after this run",
                    )
                )
                reported.add(answer.index)
            touched.setdefault(name, answer.kind)

    # One column, one record, for the same reason as one model, one record: an
    # edit applied after a create of the same key would be a second, partial
    # description of a column the first record already described in full.
    columns: set[str] = set()
    for answer in answers:
        if answer.kind not in BENCHMARK_KINDS:
            continue
        if answer.subject in columns and answer.index not in reported:
            failures.append(
                Failure(
                    answer.index,
                    f"benchmark {answer.subject!r} is touched twice in one batch; "
                    "send the second one after this run",
                )
            )
            reported.add(answer.index)
        columns.add(answer.subject)

    failures.extend(_reference_floor_failures(answers, reported))

    return answers, failures


def batch_created_benchmarks(records: Sequence[Any], doc: dict[str, Any]) -> frozenset[str]:
    """The benchmark keys this batch adds to llm.json.

    Read for shape only, like batch_created_models(): the create record is
    validated on its own, and the batch fails with it if it is refused, so a
    score or a mapping can never be left pointing at a column that was not made.
    """
    existing = doc.get("benchmarks") or {}
    keys = set()
    for record in records:
        if not isinstance(record, dict) or record.get("kind") != BENCHMARK_CREATE:
            continue
        key = record.get("key")
        if isinstance(key, str) and BENCHMARK_KEY_RE.fullmatch(key) and key not in existing:
            keys.add(key)
    return frozenset(keys)


def batch_created_models(records: Sequence[Any], model_names: set[str]) -> frozenset[str]:
    """The models this batch adds to llm.json, by name.

    A mapping may answer with one of them: a release a board lists before
    Artificial Analysis does is added and mapped in one sitting, rather than
    added in one run and mapped in the next once the queue offers it. Only the
    shape is read here -- the create record is validated on its own, and if it
    fails, the batch fails with it, so a mapping can never be left pointing at a
    model that was not made. apply() runs the creates first, so the order the
    records arrive in does not matter.
    """
    names = set()
    for record in records:
        if not isinstance(record, dict) or record.get("kind") not in (MODEL_CREATE, MODEL_ADD):
            continue
        name = record.get("name")
        if isinstance(name, str) and _rename.SLUG_RE.fullmatch(name) and name not in model_names:
            names.add(name)
    return frozenset(names)


def _reference_floor_failures(
    answers: Sequence[Answer], reported: set[int]
) -> list[Failure]:
    """Refuse a batch that would leave no reference models at all.

    Per-record validation cannot see this: each remove is legal on its own, and
    it is only the last one that empties the list. The count is taken over the
    whole batch -- adds included, so swapping the final model for another in
    one sitting is allowed, which is the one shape this floor must not block.
    Reported against the last remove, because that is the record to drop.
    """
    removes = [a for a in answers if a.kind == REFERENCE_REMOVE]
    if not removes:
        return []
    slugs = set(_reference.load_reference_slugs())
    slugs.difference_update(a.subject for a in removes)
    slugs.update(a.subject for a in answers if a.kind == REFERENCE_ADD)
    if len(slugs) >= _reference.MIN_REFERENCE_MODELS:
        return []
    last = removes[-1]
    if last.index in reported:
        return []
    return [
        Failure(
            last.index,
            "this batch would leave no reference models; the index needs at "
            "least one closed model to be measured against, so keep one or add "
            "its replacement in the same batch",
        )
    ]


def _validate_one(
    index: int,
    record: Any,
    *,
    universes: dict[str, list[str]],
    model_names: set[str],
    benchmarks: dict[str, Any],
    queue: set[tuple[str, str, str]],
    require_queue: bool,
    created: frozenset[str] = frozenset(),
    doc: dict[str, Any] | None = None,
) -> Answer:
    if not isinstance(record, dict):
        raise AnswerError("expected an object")
    kind = record.get("kind")
    if kind not in KINDS:
        raise AnswerError(f"unknown kind {kind!r}; expected one of: {', '.join(sorted(KINDS))}")

    for forbidden in ("module", "writer", "mapping_const", "path", "file"):
        if forbidden in record:
            raise AnswerError(
                f"{forbidden!r} is not accepted from a record; a route names the file"
            )

    if kind == MAPPING:
        return _validate_mapping(index, record, universes, queue, require_queue, created)
    if kind == CANDIDATES_SKIP:
        return _validate_candidates_skip(index, record, universes, queue, require_queue)
    if kind == AA_IGNORE:
        return _validate_aa_ignore(index, record, universes, model_names, queue, require_queue)
    if kind == NEW_MODEL:
        return _validate_new_model(index, record, model_names, queue, require_queue)
    if kind == MODEL_ADD:
        return _validate_model_add(index, record, model_names, queue, require_queue)
    if kind == MODEL_CREATE:
        return _validate_model_create(index, record, model_names)
    if kind == MODEL_RENAME:
        return _validate_model_rename(index, record, model_names)
    if kind in (REFERENCE_ADD, REFERENCE_REMOVE):
        return _validate_reference(index, record, kind, universes)
    if kind == BENCHMARK_CREATE:
        return _validate_benchmark_create(index, record, doc or {})
    if kind == BENCHMARK_EDIT:
        return _validate_benchmark_edit(index, record, doc or {})
    # A model-edit answers no queued question -- it is free-form maintenance of
    # an entry that already exists -- so it is bounded by the model having to
    # exist and by the field and benchmark whitelists instead.
    return _validate_model_edit(index, record, model_names, benchmarks)


def _subject_of(record: dict[str, Any], field_name: str = "subject") -> str:
    subject = record.get(field_name)
    if not isinstance(subject, str) or not subject.strip():
        raise AnswerError(f"{field_name!r} must be a non-empty string")
    return subject


def _check_previous(record: dict[str, Any], actual: list[str] | None) -> None:
    """Refuse a record written against a stale view of the file.

    The page may be working from a queue up to three hours old, so between
    render and submit the key can have been answered another way -- by the
    other phone, by a merged proposal PR, by somebody at a terminal. Carrying
    what the sender believed the file held turns that from a silent clobber
    into a rejected record they can look at again.
    """
    if "if_previous" not in record:
        raise AnswerError(
            "'if_previous' is required: send what the queue said the file held "
            "(null when the name is unanswered)"
        )
    expected = as_list(record["if_previous"])
    if expected != actual:
        raise AnswerError(
            f"stale: the file now says {actual!r}, not {expected!r}; re-read the queue"
        )


def _validate_mapping(
    index: int,
    record: dict[str, Any],
    universes: dict[str, list[str]],
    queue: set[tuple[str, str, str]],
    require_queue: bool,
    created: frozenset[str] = frozenset(),
) -> Answer:
    route_name = record.get("route")
    route = resolve_route(route_name, record.get("route_kind"))
    mapping_path(route)  # rejects a route pointing outside the mapping files
    subject = _subject_of(record)
    route_kind = record.get("route_kind") or "*"

    if require_queue and not in_queue(queue, route_name, route_kind, subject):
        raise AnswerError(
            f"{subject!r} is not a question {route_name} asked; "
            "only queued questions can be answered"
        )

    value = record.get("answer")
    if not isinstance(value, str) or not value:
        raise AnswerError("'answer' must be a non-empty string")
    if value == PENDING:
        raise AnswerError(
            "__pending__ is a parking marker, not an answer: writing it would "
            "undo a recorded decision and re-queue the name for good"
        )
    if value == CLOSED_WEIGHTS:
        raise AnswerError(
            "__closed_weights__ is recorded from the source's own claim, not by "
            "hand; use __unmappable__ to decline a name"
        )
    # A model this batch creates is as good as one llm.json already has: see
    # batch_created_models(). Only for a route whose answers are model names.
    if value not in HUMAN_SENTINELS and not (route.universe == propose.MODELS and value in created):
        options = universes.get(route.universe) or []
        if not options:
            # Never fall back to the queue's own candidate list the way
            # propose.py does: that is graceful degradation for a suggestion and
            # a validation bypass for an answer.
            raise AnswerError(
                f"the {route.universe} list is empty (its source was unreachable), "
                "so no answer for this route can be checked"
            )
        if value not in options:
            raise AnswerError(f"{value!r} is not one of the known {route.universe}")

    if route_name == AA_ROUTE:
        # This file runs the other way round: the key is an llm.json model name
        # and the value is the AA slug. Checking the value alone would let a
        # perfectly plausible, entirely wrong line through.
        if subject not in (universes.get(propose.MODELS) or []):
            raise AnswerError(f"{subject!r} is not a model in llm.json")

    _check_previous(record, current_value(route, subject))
    return Answer(
        index=index,
        kind=MAPPING,
        subject=subject,
        route=route_name,
        route_kind=route_kind,
        value=value,
    )


def _validate_candidates_skip(
    index: int,
    record: dict[str, Any],
    universes: dict[str, list[str]],
    queue: set[tuple[str, str, str]],
    require_queue: bool,
) -> Answer:
    """Rule candidates out for a queued name, answering nothing else.

    Held to the same rules as a mapping, minus the value: the route comes from
    the table, the question has to be one the queue asked, and every candidate
    named has to be in the route's own list -- a rejection of a name nothing
    could ever propose is noise in a file somebody has to read.
    """
    route_name = record.get("route")
    resolve_route(route_name, record.get("route_kind"))
    subject = _subject_of(record)
    route_kind = record.get("route_kind") or "*"
    if require_queue and not in_queue(queue, route_name, route_kind, subject):
        raise AnswerError(
            f"{subject!r} is not a question {route_name} asked; "
            "only queued questions can be answered"
        )
    options = record.get("answer")
    if not isinstance(options, list) or not options or not all(
        isinstance(o, str) and o for o in options
    ):
        raise AnswerError("'answer' must be a non-empty list of the candidates being ruled out")
    if any(o in SENTINELS for o in options):
        raise AnswerError("a sentinel is an answer, not a candidate; it cannot be ruled out")
    route = resolve_route(route_name, route_kind)
    known = universes.get(route.universe) or []
    if not known:
        raise AnswerError(
            f"the {route.universe} list is empty (its source was unreachable), "
            "so the candidates cannot be checked"
        )
    unknown = [o for o in options if o not in known]
    if unknown:
        raise AnswerError(f"not among the known {route.universe}: {', '.join(sorted(unknown))}")
    return Answer(
        index=index,
        kind=CANDIDATES_SKIP,
        subject=subject,
        route=route_name,
        route_kind=route_kind,
        value=sorted(set(options)),
    )


def _require_queued(
    queue: set[tuple[str, str, str]],
    require_queue: bool,
    route: str,
    kind: str,
    subject: str,
    asked_by: str,
) -> None:
    if require_queue and not in_queue(queue, route, kind, subject):
        raise AnswerError(
            f"{subject!r} is not a question {asked_by} asked; "
            "only queued questions can be answered"
        )


def _validate_aa_ignore(
    index: int,
    record: dict[str, Any],
    universes: dict[str, list[str]],
    model_names: set[str],
    queue: set[tuple[str, str, str]],
    require_queue: bool,
) -> Answer:
    subject = _subject_of(record)
    if subject not in model_names:
        raise AnswerError(f"{subject!r} is not a model in llm.json")
    _require_queued(queue, require_queue, AA_ROUTE, "aa-mapping", subject, AA_ROUTE)
    slugs = record.get("answer")
    if not isinstance(slugs, list) or not slugs or not all(isinstance(s, str) and s for s in slugs):
        raise AnswerError("'answer' must be a non-empty list of the AA slugs being rejected")
    known = universes.get(propose.AA_SLUGS) or []
    if not known:
        raise AnswerError(
            "the Artificial Analysis slug list is empty (its source was unreachable), "
            "so rejected slugs cannot be checked"
        )
    unknown = [s for s in slugs if s not in known]
    if unknown:
        raise AnswerError(f"not Artificial Analysis slugs: {', '.join(sorted(unknown))}")
    return Answer(index=index, kind=AA_IGNORE, subject=subject, value=list(slugs))


def _validate_new_model(
    index: int,
    record: dict[str, Any],
    model_names: set[str],
    queue: set[tuple[str, str, str]],
    require_queue: bool,
) -> Answer:
    subject = _subject_of(record)
    _require_queued(queue, require_queue, NEW_MODEL_ROUTE, NEW_MODEL, subject, NEW_MODEL_ROUTE)
    value = record.get("answer")
    if value != _new_models.IGNORED:
        # __added__ is not the other half of this answer. apply_decisions() only
        # keeps it when the entry is already in llm.json; for a slug that is not,
        # it logs "left alone", clears the line and never dismisses the slug, so
        # check_new.py offers the model again on the very next run, forever.
        # "Yes, add it" is a model-add record: run add.py, then record __added__.
        raise AnswerError(
            f"'answer' must be {_new_models.IGNORED!r}; to add a model send a "
            f"{MODEL_ADD!r} record instead"
        )
    if subject in model_names:
        raise AnswerError(
            f"{subject!r} is already in llm.json; ignoring it would remove the entry "
            "and its scores -- send a model-edit or delete it deliberately"
        )
    return Answer(index=index, kind=NEW_MODEL, subject=subject, value=value)


def _validate_model_add(
    index: int,
    record: dict[str, Any],
    model_names: set[str],
    queue: set[tuple[str, str, str]],
    require_queue: bool,
) -> Answer:
    subject = _subject_of(record, "name")
    if subject in model_names:
        raise AnswerError(f"{subject!r} is already in llm.json")
    # Without this an arbitrary string becomes an llm.json entry: add.py builds
    # one from whatever name it is handed, prefilling what Artificial Analysis
    # knows and leaving the rest blank, so a junk slug lands as a junk model.
    _require_queued(queue, require_queue, NEW_MODEL_ROUTE, NEW_MODEL, subject, NEW_MODEL_ROUTE)
    return Answer(index=index, kind=MODEL_ADD, subject=subject)


def _metadata_fields(record: dict[str, Any]) -> dict[str, Any]:
    """The metadata a record sets, checked field by field.

    The whitelist is edit.py's own METADATA_FIELDS, and each value goes through
    that script's parser for its field, so a URL that is not one and a date that
    is not one are refused here rather than reaching llm.json. A null clears the
    field, which is the same thing `--flag=null` means on the command line.
    """
    fields = record.get("fields") or {}
    if not isinstance(fields, dict):
        raise AnswerError("'fields' must be an object")

    checked: dict[str, Any] = {}
    for key, value in fields.items():
        if key not in METADATA_FIELDS:
            raise AnswerError(
                f"{key!r} is not editable; expected one of: {', '.join(METADATA_FIELDS)}"
            )
        if value is not None and not isinstance(value, str):
            raise AnswerError(f"{key!r} must be a string or null")
        try:
            edit.parse_field_value(key, value)
        except ValueError as exc:
            raise AnswerError(str(exc)) from None
        checked[key] = value
    return checked


def _model_slug(record: dict[str, Any]) -> str:
    """A name that may become an entry in llm.json, checked to be a slug.

    Every name in the file is lowercase words joined by single dots or dashes,
    and the name is the model's identity everywhere -- mapping files, the AA
    lookup, the site's own links -- so a stray capital or space is not a style
    preference to fix later.
    """
    name = _subject_of(record, "name")
    if not _rename.SLUG_RE.fullmatch(name):
        raise AnswerError(
            f"{name!r} is not a model slug: lowercase letters, digits, and single "
            "dots or dashes between them"
        )
    return name


def _score_date(record: dict[str, Any], has_scores: bool) -> str | None:
    """The date the scores in this record were read, or None for today.

    Absent means today, stamped on the runner rather than here, which is what
    edit.py has always done. A date that is present is checked hard: it lands in
    scores_updated, which is what the site prints as a score's age and what
    sync_score_dates.py reconciles, so a malformed or invented one is a lie the
    file then carries.
    """
    raw = record.get("score_date")
    if raw is None:
        return None
    if not has_scores:
        raise AnswerError("'score_date' stamps a score, so send at least one score with it")
    if not isinstance(raw, str):
        raise AnswerError("'score_date' must be a date like 2026-08-06, or null for today")
    try:
        parsed = date.fromisoformat(raw.strip())
    except ValueError:
        raise AnswerError(f"{raw!r} is not a date like 2026-08-06") from None
    if parsed > date.today():
        # A score cannot have been read from a page that has not happened yet,
        # and a stray year would sit at the top of every "recently updated"
        # view until somebody noticed.
        raise AnswerError(f"{parsed.isoformat()} is in the future; a score cannot be read yet")
    return parsed.isoformat()


def _score_url(record: dict[str, Any], has_scores: bool, writes_score: bool) -> str | None:
    """The page the scores in this record were read from, or None.

    Required whenever the record writes a number (edit.py refuses the run
    otherwise, which would take the rest of the batch down after it): the page
    moves the value onto that page's rung of _precedence.source_rank(), and it
    is the only thing a reader can check the number against. A record that only
    clears scores has nothing to credit.
    """
    raw = record.get("score_url")
    if raw is not None and not has_scores:
        raise AnswerError("'score_url' credits a score, so send at least one score with it")
    if raw is not None and not isinstance(raw, str):
        raise AnswerError("'score_url' must be a URL string")
    url = (raw or "").strip()
    if not url:
        if writes_score:
            raise AnswerError(
                "'score_url' is required with a score: name the page it was read from"
            )
        return None
    if not url.startswith(("http://", "https://")):
        raise AnswerError(f"{raw!r} is not a URL starting with http:// or https://")
    return url


def _validate_model_create(
    index: int,
    record: dict[str, Any],
    model_names: set[str],
) -> Answer:
    """A model somebody adds deliberately, rather than one the queue offered.

    model-add is the queue's answer and stays gated on the question: check_new
    found the slug on Artificial Analysis, and add.py fills the entry in from
    what AA knows. This is the other direction -- a model no source has offered,
    typed in by hand -- so there is no question to check it against, and the
    guard is the name having to be a slug, the entry having to be new, and every
    metadata value being one its field accepts.
    """
    name = _model_slug(record)
    if name in model_names:
        raise AnswerError(f"{name!r} is already a model in llm.json; edit it instead")
    return Answer(
        index=index,
        kind=MODEL_CREATE,
        subject=name,
        fields=_metadata_fields(record),
    )


def _validate_model_rename(
    index: int,
    record: dict[str, Any],
    model_names: set[str],
) -> Answer:
    """Give an entry a different slug -- usually the one AA ended up using.

    update.py reads Artificial Analysis directly for a model whose name is an AA
    slug, so this is how a hand-added entry starts collecting AA scores without
    a mapping to maintain. _rename.py is what makes it safe: the name is written
    down in every source's mapping file too, and one left behind does not fail,
    it silently stops matching.
    """
    old = _subject_of(record, "name")
    if old not in model_names:
        raise AnswerError(f"{old!r} is not a model in llm.json")
    new = _model_slug({"name": record.get("new_name")})
    if new == old:
        raise AnswerError(f"{old!r} is already its name")
    if new in model_names:
        raise AnswerError(f"{new!r} is already a model in llm.json; merge them by hand")
    return Answer(index=index, kind=MODEL_RENAME, subject=old, value=new)


def _validate_reference(
    index: int,
    record: dict[str, Any],
    kind: str,
    universes: dict[str, list[str]],
) -> Answer:
    """Put an Artificial Analysis slug on the reference list, or take it off.

    The list is AA slugs by definition (that is what makes a reference model
    resolvable without a mapping), so an add is checked against AA's own list
    -- softly, because build_universes leaves it empty when AA cannot be
    reached and a dead source must not refuse a batch it has no opinion about.

    Nothing is checked against the queue: nobody queues these. The guards are
    the slug shape, AA knowing the name, and the list's own before-and-after
    state -- including the floor, which validate() applies to the batch as a
    whole because two removes are only too many together.
    """
    slug = _model_slug(record)
    on_list = _reference.load_reference_slugs()

    if kind == REFERENCE_REMOVE:
        if slug not in on_list:
            raise AnswerError(f"{slug!r} is not a reference model")
        return Answer(index=index, kind=REFERENCE_REMOVE, subject=slug)

    if slug in on_list:
        raise AnswerError(f"{slug!r} is already a reference model")
    aa_slugs = universes.get(propose.AA_SLUGS) or []
    if aa_slugs and slug not in aa_slugs:
        raise AnswerError(
            f"{slug!r} is not an Artificial Analysis slug; a reference model is "
            "named after one so every source's mapping resolves onto it"
        )
    return Answer(index=index, kind=REFERENCE_ADD, subject=slug)


def _benchmark_key(record: dict[str, Any]) -> str:
    key = record.get("key")
    if not isinstance(key, str) or not key:
        raise AnswerError("'key' must be a non-empty string naming the benchmark column")
    if len(key) > MAX_BENCHMARK_KEY or not BENCHMARK_KEY_RE.fullmatch(key):
        raise AnswerError(
            f"{key!r} is not a benchmark key: lowercase letters and digits, words "
            f"joined by single underscores, at most {MAX_BENCHMARK_KEY} characters"
        )
    return key


def _clean_text(name: str, value: Any, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnswerError(f"{name!r} must be a non-empty string")
    text = value.strip()
    if len(text) > limit:
        raise AnswerError(f"{name!r} is {len(text)} characters; at most {limit}")
    if any(ch in text for ch in "\r\n\t") and name != "description":
        raise AnswerError(f"{name!r} must be one line")
    return text


def _clean_list(name: str, value: Any) -> list[str]:
    """One of a benchmark's three lists, held to test_benchmark_settings.py.

    Settings are drawn as chips and excludes are joined by a separator, so each
    item has to stand on its own: one line, no trailing full stop, no repeats.
    """
    if not isinstance(value, list):
        raise AnswerError(f"{name!r} must be a list of strings")
    if len(value) > MAX_LIST_ITEMS:
        raise AnswerError(f"{name!r} has {len(value)} items; at most {MAX_LIST_ITEMS}")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise AnswerError(f"every item of {name!r} must be a non-empty string")
        text = " ".join(item.split())
        if name == "urls":
            if not text.startswith(("http://", "https://")) or " " in text:
                raise AnswerError(f"{item!r} is not a URL starting with http:// or https://")
        else:
            if text.endswith("."):
                raise AnswerError(f"{text!r} in {name!r} is a phrase, not a sentence: drop the full stop")
            if name == "settings" and len(text) > MAX_SETTING_TAG:
                raise AnswerError(
                    f"{text!r} is not a tag: at most {MAX_SETTING_TAG} characters per setting"
                )
            if len(text) > 200:
                raise AnswerError(f"{text!r} in {name!r} is too long for one phrase")
        if text in items:
            raise AnswerError(f"{text!r} is listed twice in {name!r}")
        items.append(text)
    return items


def _benchmark_fields(
    record: dict[str, Any], *, derived: bool, scores: Sequence[float] = ()
) -> dict[str, Any]:
    """The benchmark fields a record sets, each checked and normalised.

    A null clears a field that may be absent -- excludes, and the four numeric
    ones, whose absence is _scores.py's default -- and is refused for the ones
    every column must carry. `scores` are the values already stored in this
    column: a range they fall outside of would make every later write of the
    same number fail check_score_range, so it is refused here instead.
    """
    fields = record.get("fields")
    if not isinstance(fields, dict):
        raise AnswerError("'fields' must be an object")
    checked: dict[str, Any] = {}
    for key, value in fields.items():
        if key not in BENCHMARK_FIELDS:
            raise AnswerError(
                f"{key!r} is not a benchmark field a record may set; expected one of: "
                f"{', '.join(BENCHMARK_FIELDS)}"
            )
        if derived and key in BENCHMARK_NUMBER_FIELDS:
            raise AnswerError(
                f"{key!r} is fixed for a derived index: derive_indexes.py computes "
                "it on its own scale"
            )
        if value is None:
            if key in BENCHMARK_REQUIRED:
                raise AnswerError(f"{key!r} cannot be cleared: every benchmark carries one")
            checked[key] = None
            continue
        if key in BENCHMARK_TEXT_FIELDS:
            checked[key] = _clean_text(key, value, BENCHMARK_TEXT_FIELDS[key])
        elif key in BENCHMARK_LIST_FIELDS:
            items = _clean_list(key, value)
            if not items and key in BENCHMARK_REQUIRED:
                raise AnswerError(f"{key!r} needs at least one item")
            # An empty excludes is not a statement; the key is left out instead.
            checked[key] = items or None
        elif key == "decimals":
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_DECIMALS:
                raise AnswerError(f"'decimals' must be a whole number from 0 to {MAX_DECIMALS}")
            checked[key] = value
        elif key == "round_to":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise AnswerError("'round_to' must be a positive number")
            checked[key] = value
        elif key == "range":
            if (
                not isinstance(value, list)
                or len(value) != 2
                or any(
                    isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b)
                    for b in value
                )
                or not value[0] < value[1]
            ):
                raise AnswerError("'range' must be [low, high] with low below high")
            outside = [v for v in scores if not value[0] <= v <= value[1]]
            if outside:
                raise AnswerError(
                    f"{len(outside)} stored score(s) fall outside {value}, e.g. "
                    f"{outside[0]}; correct those first"
                )
            checked[key] = list(value)
        else:  # lower_is_better
            if not isinstance(value, bool):
                raise AnswerError("'lower_is_better' must be true or false")
            # False is the default; stored as its absence, like every column
            # that never declared it.
            checked[key] = True if value else None
    return checked


def _check_flag_free(key: str) -> None:
    """Refuse a key edit.py could not take as a --flag of its own.

    edit.py adds one option per benchmark, so a key that spells one of its own
    options -- --model, --after, --score-url -- would break every model edit,
    the admin page's included, until somebody renamed the column by hand.
    """
    reserved = {
        "json_file", "model", "missing", "benchmark", "field", "after", "help",
        "score_date", "score_url", *METADATA_FIELDS,
    }
    if key in reserved:
        raise AnswerError(f"{key!r} is one of edit.py's own options; pick another key")


def _validate_benchmark_create(index: int, record: dict[str, Any], doc: dict[str, Any]) -> Answer:
    """A new column: its key, everything a reader is shown about it, and where.

    Every required field has to be sent -- test_benchmark_settings.py holds the
    file to them, and a column without its settings is mapped from its key
    alone. `after` places it: the column it follows, by default the last one
    of its own category, so the table's groups stay together.
    """
    key = _benchmark_key(record)
    existing = doc.get("benchmarks") or {}
    if key in existing:
        raise AnswerError(f"benchmark {key!r} already exists; edit it instead")
    _check_flag_free(key)
    fields = _benchmark_fields(record, derived=False)
    missing = [f for f in BENCHMARK_REQUIRED if fields.get(f) is None]
    if missing:
        raise AnswerError(f"a new benchmark needs {', '.join(missing)}")
    after = record.get("after")
    if after is not None:
        if not isinstance(after, str) or after not in existing:
            raise AnswerError(f"'after' must name an existing benchmark, not {after!r}")
    return Answer(
        index=index,
        kind=BENCHMARK_CREATE,
        subject=key,
        value=after,
        fields={k: v for k, v in fields.items() if v is not None},
    )


def _validate_benchmark_edit(index: int, record: dict[str, Any], doc: dict[str, Any]) -> Answer:
    key = _benchmark_key(record)
    bench = (doc.get("benchmarks") or {}).get(key)
    if not isinstance(bench, dict):
        raise AnswerError(f"{key!r} is not a benchmark in llm.json")
    stored = [
        value
        for model in doc.get("models") or []
        if isinstance(model, dict)
        for value in [(model.get("scores") or {}).get(key)]
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    fields = _benchmark_fields(record, derived=bench.get("derived") is True, scores=stored)
    # Only what differs: an unchanged field is no edit, and leaving it out keeps
    # the description of the change honest.
    changed = {k: v for k, v in fields.items() if bench.get(k) != v}
    if not changed:
        raise AnswerError(f"nothing to change: send at least one field of {key!r} that differs")
    return Answer(index=index, kind=BENCHMARK_EDIT, subject=key, fields=changed)


def _validate_model_edit(
    index: int,
    record: dict[str, Any],
    model_names: set[str],
    benchmarks: dict[str, Any],
) -> Answer:
    subject = _subject_of(record, "name")
    if subject not in model_names:
        raise AnswerError(f"{subject!r} is not a model in llm.json")

    fields = _metadata_fields(record)
    scores = record.get("scores") or {}
    if not isinstance(scores, dict):
        raise AnswerError("'scores' must be an object")
    if not fields and not scores:
        raise AnswerError("nothing to change: send at least one field or score")

    for key, value in scores.items():
        if key not in benchmarks:
            # editable_benchmarks drops the derived index columns: a score
            # written to one is recomputed away by derive_indexes.py on the next
            # run, so it would be a silent no-op that reads like an answer.
            raise AnswerError(f"{key!r} is not a benchmark a score can be written to")
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AnswerError(f"the score for {key!r} must be a number or null")
        if not math.isfinite(value):
            # JSON has no inf, but 1e400 parses to one, and it would reach
            # edit.py as the text "inf" and be rejected there instead -- after
            # earlier records in the batch had already been written.
            raise AnswerError(f"the score for {key!r} must be a finite number")

    return Answer(
        index=index,
        kind=MODEL_EDIT,
        subject=subject,
        fields=dict(fields),
        scores=dict(scores),
        score_date=_score_date(record, bool(scores)),
        score_url=_score_url(
            record, bool(scores), any(value is not None for value in scores.values())
        ),
    )


# --------------------------------------------------------------------------
# applying
# --------------------------------------------------------------------------


def touchable_paths(answers: Iterable[Answer], llm_path: Path) -> list[Path]:
    """Every file a batch could write, for the rollback snapshot."""
    paths = {
        llm_path,
        _new_models.DECISIONS_FILE,
        _new_models.DISMISSED_FILE,
        _reference.REFERENCE_MODELS,
    }
    aa_module = importlib.import_module(AA_MODULE)
    for answer in answers:
        if answer.kind == MAPPING:
            paths.add(mapping_path(resolve_route(answer.route, answer.route_kind)))
        elif answer.kind == AA_IGNORE:
            paths.add(aa_module.AA_MODEL_IGNORES)
        elif answer.kind == CANDIDATES_SKIP:
            paths.add(_rejections.path_for(answer.route))
        elif answer.kind == MODEL_RENAME:
            # A rename walks every mapping file, so the snapshot has to as well:
            # a failure part way through is precisely the half-renamed model
            # nothing else in the repository can detect.
            paths.update(_rename.touched_paths(llm_path))
    return sorted(paths)


def _snapshot(paths: Iterable[Path]) -> dict[Path, bytes | None]:
    return {p: (p.read_bytes() if p.exists() else None) for p in paths}


def _restore(snapshot: dict[Path, bytes | None]) -> None:
    for path, data in snapshot.items():
        if data is None:
            if path.exists():
                path.unlink()
        elif path.read_bytes() != data:
            path.write_bytes(data)


def apply(answers: Sequence[Answer], *, llm_path: Path = DEFAULT_LLM_JSON) -> list[str]:
    """Write every answer, or none of them. Returns one log line each.

    Refuses under collect mode: every mapping writer is a no-op while
    _prompts.freeze_decisions() is true (that is what keeps CI from answering
    its own questions), so applying there would report success and write
    nothing.
    """
    if _prompts.collecting():
        raise AnswerError(
            f"{_prompts.ENV_VAR} is set, which freezes every mapping writer; "
            "unset it before applying answers"
        )

    snapshot = _snapshot(touchable_paths(answers, llm_path))
    log: list[str] = []
    # Models first, so a mapping onto a model the same batch creates finds it
    # there (see batch_created_models). Stable, so everything else keeps the
    # order it was sent in.
    # Columns before that, so a score or a mapping onto a column the batch adds
    # finds it too, and a range the batch widens covers the scores it writes.
    ordered = sorted(
        answers,
        key=lambda answer: (
            answer.kind not in BENCHMARK_KINDS,
            answer.kind not in (MODEL_CREATE, MODEL_ADD),
        ),
    )
    try:
        for answer in ordered:
            log.extend(_apply_one(answer, llm_path))
    except Exception:
        _restore(snapshot)
        raise
    return log


def _apply_one(answer: Answer, llm_path: Path) -> list[str]:
    if answer.kind == MAPPING:
        route = resolve_route(answer.route, answer.route_kind)
        writer_for(route)(answer.subject, answer.value)
        return [f"{mapping_path(route).name}: {answer.subject!r} -> {answer.value}"]

    if answer.kind == CANDIDATES_SKIP:
        written = _rejections.add(answer.route, answer.subject, list(answer.value))
        return [f"{written.name}: {answer.subject!r} is none of {', '.join(answer.value)}"]

    if answer.kind == AA_IGNORE:
        module = importlib.import_module(AA_MODULE)
        module.add_ignored_aa_suggestions(answer.subject, list(answer.value))
        return [f"{module.AA_MODEL_IGNORES.name}: {answer.subject!r} rejects {', '.join(answer.value)}"]

    if answer.kind == NEW_MODEL:
        decisions = _new_models.load_decisions()
        decisions[answer.subject] = answer.value
        _new_models.write_decisions(decisions)
        return [f"{_new_models.DECISIONS_FILE.name}: {answer.subject!r} -> {answer.value}"]

    if answer.kind == MODEL_ADD:
        _add_model(answer.subject, answer.fields, llm_path)
        # The half that stops check_new.py offering the slug again: without it
        # the model is in llm.json but nothing says the question was answered.
        _new_models.record_proposed(answer.subject)
        return [f"{llm_path.name}: added {answer.subject!r} (scores land on the next refresh)"]

    if answer.kind == MODEL_CREATE:
        _add_model(answer.subject, answer.fields, llm_path)
        # No record_proposed here, unlike model-add: that line answers a
        # question check_new.py asked about an AA slug, and nobody asked this
        # one. If AA does turn out to publish the same slug, the entry is
        # already in llm.json, which is what check_new.py filters on.
        named = ", ".join(sorted(answer.fields)) or "name only"
        return [f"{llm_path.name}: created {answer.subject!r} ({named})"]

    if answer.kind == MODEL_RENAME:
        return _rename.rename(answer.subject, answer.value, llm_path)

    if answer.kind == REFERENCE_ADD:
        return _add_reference(answer, llm_path)

    if answer.kind == REFERENCE_REMOVE:
        return _remove_reference(answer, llm_path)

    if answer.kind == BENCHMARK_CREATE:
        return _create_benchmark(answer, llm_path)

    if answer.kind == BENCHMARK_EDIT:
        return _edit_benchmark(answer, llm_path)

    return _apply_edit(answer, llm_path)


# The order a benchmark's keys are written in, so a column added from the page
# reads like one written by hand.
BENCHMARK_KEY_ORDER = (
    "name", "short_name", "category", "derived", "decimals", "round_to", "range",
    "lower_is_better", "description", "settings", "excludes", "icon_svg", "urls",
)


def _ordered_benchmark(bench: dict[str, Any]) -> dict[str, Any]:
    known = [k for k in BENCHMARK_KEY_ORDER if k in bench]
    return {k: bench[k] for k in known + [k for k in bench if k not in known]}


def _insert_after(mapping: dict[str, Any], anchor: str | None, key: str, value: Any) -> dict[str, Any]:
    """`mapping` with key inserted right after anchor, or at the end."""
    if anchor is None or anchor not in mapping:
        return {**{k: v for k, v in mapping.items() if k != key}, key: value}
    out: dict[str, Any] = {}
    for k, v in mapping.items():
        if k == key:
            continue
        out[k] = v
        if k == anchor:
            out[key] = value
    return out


def _write_llm(doc: dict[str, Any], llm_path: Path) -> None:
    llm_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _create_benchmark(answer: Answer, llm_path: Path) -> list[str]:
    """Add the column, and a null score for it on every model.

    The null is the file's own convention for "no score yet" (add.py writes one
    per column for a new model), and it lands next to the column's neighbours
    so a model's scores keep the order the benchmarks are declared in.
    """
    doc = json.loads(llm_path.read_text(encoding="utf-8"))
    benchmarks = doc.get("benchmarks")
    if not isinstance(benchmarks, dict):
        raise AnswerError("llm.json has no benchmarks object")
    if answer.subject in benchmarks:
        raise AnswerError(f"benchmark {answer.subject!r} already exists")
    anchor = answer.value
    if anchor is None:
        same = [k for k, b in benchmarks.items() if isinstance(b, dict)
                and b.get("category") == answer.fields.get("category")]
        anchor = same[-1] if same else None
    doc["benchmarks"] = _insert_after(
        benchmarks, anchor, answer.subject, _ordered_benchmark(dict(answer.fields))
    )
    keys = list(doc["benchmarks"])
    earlier = keys[: keys.index(answer.subject)]
    previous = earlier[-1] if earlier else None
    for model in doc.get("models") or []:
        scores = model.get("scores") if isinstance(model, dict) else None
        if not isinstance(scores, dict) or answer.subject in scores:
            continue
        # After the nearest earlier column this model has a score entry for:
        # not every model carries every column (one added since it was), and
        # appending would put the new one out of order for exactly those.
        anchor = next((k for k in reversed(earlier) if k in scores), None)
        if anchor is None:
            model["scores"] = {answer.subject: None, **scores}
        else:
            model["scores"] = _insert_after(scores, anchor, answer.subject, None)
    _write_llm(doc, llm_path)
    placed = f"after {previous!r}" if previous else "first"
    return [f"{llm_path.name}: created benchmark {answer.subject!r} ({placed})"]


def _edit_benchmark(answer: Answer, llm_path: Path) -> list[str]:
    doc = json.loads(llm_path.read_text(encoding="utf-8"))
    bench = (doc.get("benchmarks") or {}).get(answer.subject)
    if not isinstance(bench, dict):
        raise AnswerError(f"{answer.subject!r} is not a benchmark in llm.json")
    for key, value in answer.fields.items():
        if value is None:
            bench.pop(key, None)
        elif key in bench:
            bench[key] = value
        else:
            # A field the column never had goes where a hand would put it,
            # without reordering the ones already there: the diff stays the
            # change and nothing else.
            rank = BENCHMARK_KEY_ORDER.index(key)
            before = [k for k in bench if k in BENCHMARK_KEY_ORDER[:rank]]
            bench = _insert_after(bench, before[-1] if before else None, key, value)
            if not before:
                bench = {key: value, **{k: v for k, v in bench.items() if k != key}}
    doc["benchmarks"][answer.subject] = bench
    _write_llm(doc, llm_path)
    return [f"{llm_path.name}: edited benchmark {answer.subject!r} ({', '.join(sorted(answer.fields))})"]


def _add_reference(answer: Answer, llm_path: Path) -> list[str]:
    """Carry one more closed model: the list, then the entry behind it.

    add.py fills the entry from Artificial Analysis, which for these models is
    the source that has the numbers -- the creator and the context window come
    back, the parameter count does not, because nobody published one. The URL
    is passed rather than left to AA: a closed model has no weights repository,
    so the model page its scores are read off is what the row is checked
    against.
    """
    _reference.add_reference_slug(answer.subject)
    log = [f"{_reference.REFERENCE_MODELS.name}: carrying {answer.subject!r}"]

    doc = json.loads(llm_path.read_text(encoding="utf-8"))
    if answer.subject not in {m.get("name") for m in doc.get("models", [])}:
        _add_model(
            answer.subject, {"url": _reference.aa_model_page(answer.subject)}, llm_path
        )
        log.append(f"{llm_path.name}: added {answer.subject!r} (scores land on the next refresh)")
    return log + _sync_reference_flags(llm_path)


def _remove_reference(answer: Answer, llm_path: Path) -> list[str]:
    """Stop carrying one: the slug, and the entry with it.

    The entry goes too, on purpose. llm.json holds open-weight models plus
    exactly this list; an entry left behind would be a closed model with its
    flag cleared -- shown in the table as an open-weight one, exported as one,
    and offered as one to a reader filtering for what they can host. Its scores
    are scraped, so a row added back later fills in again on the next refresh.
    """
    _reference.remove_reference_slug(answer.subject)
    log = [f"{_reference.REFERENCE_MODELS.name}: no longer carrying {answer.subject!r}"]

    doc = json.loads(llm_path.read_text(encoding="utf-8"))
    models = doc.get("models") or []
    kept = [m for m in models if not (isinstance(m, dict) and m.get("name") == answer.subject)]
    if len(kept) != len(models):
        doc["models"] = kept
        llm_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log.append(f"{llm_path.name}: dropped {answer.subject!r}")
    return log + _sync_reference_flags(llm_path)


def _sync_reference_flags(llm_path: Path) -> list[str]:
    """Bring llm.json's `reference` flags and derived indexes back in step.

    update.py and derive_indexes.py both do this on a refresh, but a record-only
    run never reaches either -- so without this an added model would sit in the
    table as an open-weight row, and a removed one would leave the indexes
    ranked against a field that no longer exists, for as long as the next
    scheduled refresh takes.
    """
    doc = json.loads(llm_path.read_text(encoding="utf-8"))
    changes = _reference.apply_reference_flags(doc)
    derive_indexes.refresh_and_report(doc)
    llm_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return [
        f"{llm_path.name}: {name} is {'now' if marked else 'no longer'} a reference model"
        for name, marked in changes
    ]


def _add_model(name: str, fields: dict[str, Any], llm_path: Path) -> None:
    """Run add.py for one model, with whatever metadata the record carried.

    add.py fills anything left out from Artificial Analysis, and leaves it null
    when AA has never heard of the model -- which is the usual case for one
    added by hand. A null field is therefore left off the command line rather
    than sent as an empty value: an empty one is add.py's way of saying "no,
    really, nothing here", which would suppress that prefill. Same argv rules as
    the edit below: --flag=VALUE, and the path after a bare --.
    """
    argv = [sys.executable, str(ADD_SCRIPT), f"--name={name}"]
    for key, value in sorted(fields.items()):
        if value is None:
            continue
        argv.append(f"--{key.replace('_', '-')}={value}")
    argv += ["--", str(llm_path)]
    _run(argv, f"add.py failed for {name!r}")


def _apply_edit(answer: Answer, llm_path: Path) -> list[str]:
    # Always --flag=VALUE, and the path after a bare --. edit.py hand-parses
    # argv in infer_json_file() before argparse sees it, assuming "--flag value"
    # for every long option; a value that looks like a path would otherwise
    # silently redirect the write.
    argv = [sys.executable, str(EDIT_SCRIPT), f"--model={answer.subject}"]
    for key, value in sorted(answer.fields.items()):
        # creator_url is --creator-url; the record keeps the underscore because
        # that is the key on the model and argparse's own dest.
        argv.append(f"--{key.replace('_', '-')}={'null' if value is None else value}")
    for key, value in sorted(answer.scores.items()):
        flag = key.replace("_", "-")
        argv.append(f"--{flag}={'null' if value is None else value}")
    # Left off entirely when the record said nothing, so edit.py applies its own
    # defaults rather than being told them second-hand.
    if answer.score_date is not None:
        argv.append(f"--score-date={answer.score_date}")
    if answer.score_url is not None:
        argv.append(f"--score-url={answer.score_url}")
    argv += ["--", str(llm_path)]
    _run(argv, f"edit.py failed for {answer.subject!r}")
    changed = ", ".join(sorted([*answer.fields, *answer.scores]))
    return [f"{llm_path.name}: edited {answer.subject!r} ({changed})"]


def _run(argv: list[str], label: str) -> None:
    """Run a helper script, treating any failure as fatal to the batch.

    propose.py warns and carries on when add.py fails, which is right for a
    best-effort suggestion. An answer is not best-effort: reporting success for
    a write that did not happen is the one thing the sender cannot detect.
    """
    env = _prompts.child_env()
    proc = subprocess.run(argv, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise AnswerError(f"{label} (exit {proc.returncode}): {detail[-1] if detail else ''}")
