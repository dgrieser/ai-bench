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
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import _new_models
import _prompts
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
NEW_MODEL = "new-model"
MODEL_ADD = "model-add"
MODEL_EDIT = "model-edit"
KINDS = frozenset({MAPPING, AA_IGNORE, NEW_MODEL, MODEL_ADD, MODEL_EDIT})

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

    def describe(self) -> str:
        if self.kind == MAPPING:
            return f"{self.route}: {self.subject!r} -> {self.value!r}"
        if self.kind == AA_IGNORE:
            return f"AA suggestions rejected for {self.subject!r}: {', '.join(self.value)}"
        if self.kind == NEW_MODEL:
            return f"new model {self.subject!r} -> {self.value}"
        if self.kind == MODEL_ADD:
            return f"add model {self.subject!r}"
        changes = sorted([*self.fields, *self.scores])
        return f"edit model {self.subject!r}: {', '.join(changes)}"


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
        if record.get("kind") == AA_IGNORE:
            return True
        if record.get("kind") == MAPPING and record.get("route") == AA_ROUTE:
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
                    benchmarks=benchmarks,
                    queue=queue,
                    require_queue=require_queue,
                )
            )
        except AnswerError as exc:
            failures.append(Failure(index, str(exc)))

    seen: set[tuple[str, str | None, str | None, str]] = set()
    for answer in answers:
        key = (answer.kind, answer.route, answer.route_kind, answer.subject)
        if key in seen:
            failures.append(Failure(answer.index, f"answered twice in one batch: {answer.subject!r}"))
        seen.add(key)

    return answers, failures


def _validate_one(
    index: int,
    record: Any,
    *,
    universes: dict[str, list[str]],
    model_names: set[str],
    benchmarks: dict[str, Any],
    queue: set[tuple[str, str, str]],
    require_queue: bool,
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
        return _validate_mapping(index, record, universes, queue, require_queue)
    if kind == AA_IGNORE:
        return _validate_aa_ignore(index, record, universes, model_names, queue, require_queue)
    if kind == NEW_MODEL:
        return _validate_new_model(index, record, model_names, queue, require_queue)
    if kind == MODEL_ADD:
        return _validate_model_add(index, record, model_names, queue, require_queue)
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
    if value not in HUMAN_SENTINELS:
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


def _validate_model_edit(
    index: int,
    record: dict[str, Any],
    model_names: set[str],
    benchmarks: dict[str, Any],
) -> Answer:
    subject = _subject_of(record, "name")
    if subject not in model_names:
        raise AnswerError(f"{subject!r} is not a model in llm.json")

    fields = record.get("fields") or {}
    scores = record.get("scores") or {}
    if not isinstance(fields, dict) or not isinstance(scores, dict):
        raise AnswerError("'fields' and 'scores' must be objects")
    if not fields and not scores:
        raise AnswerError("nothing to change: send at least one field or score")

    for key, value in fields.items():
        if key not in METADATA_FIELDS:
            raise AnswerError(
                f"{key!r} is not editable; expected one of: {', '.join(METADATA_FIELDS)}"
            )
        if value is not None and not isinstance(value, str):
            raise AnswerError(f"{key!r} must be a string or null")
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
    )


# --------------------------------------------------------------------------
# applying
# --------------------------------------------------------------------------


def touchable_paths(answers: Iterable[Answer], llm_path: Path) -> list[Path]:
    """Every file a batch could write, for the rollback snapshot."""
    paths = {llm_path, _new_models.DECISIONS_FILE, _new_models.DISMISSED_FILE}
    aa_module = importlib.import_module(AA_MODULE)
    for answer in answers:
        if answer.kind == MAPPING:
            paths.add(mapping_path(resolve_route(answer.route, answer.route_kind)))
        elif answer.kind == AA_IGNORE:
            paths.add(aa_module.AA_MODEL_IGNORES)
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
    try:
        for answer in answers:
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
        _run(
            [sys.executable, str(ADD_SCRIPT), "--name", answer.subject, str(llm_path)],
            f"add.py failed for {answer.subject!r}",
        )
        # The half that stops check_new.py offering the slug again: without it
        # the model is in llm.json but nothing says the question was answered.
        _new_models.record_proposed(answer.subject)
        return [f"{llm_path.name}: added {answer.subject!r} (scores land on the next refresh)"]

    return _apply_edit(answer, llm_path)


def _apply_edit(answer: Answer, llm_path: Path) -> list[str]:
    # Always --flag=VALUE, and the path after a bare --. edit.py hand-parses
    # argv in infer_json_file() before argparse sees it, assuming "--flag value"
    # for every long option; a value that looks like a path would otherwise
    # silently redirect the write.
    argv = [sys.executable, str(EDIT_SCRIPT), f"--model={answer.subject}"]
    for key, value in sorted(answer.fields.items()):
        argv.append(f"--{key}={'null' if value is None else value}")
    for key, value in sorted(answer.scores.items()):
        flag = key.replace("_", "-")
        argv.append(f"--{flag}={'null' if value is None else value}")
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
