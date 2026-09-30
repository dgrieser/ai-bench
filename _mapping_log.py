"""A running record of every change to a mapping file: when, what, and who.

A mapping file says what a source name means *now*, and nothing else. Once an
answer lands, the queue question it closed is gone, and the only trace of the
decision is a line in a JSON file that git can date but nobody can read from a
phone. The admin page's Updates tab wants the opposite view -- "what was mapped
this week, and did I do it or did the pipeline?" -- so every change is written
down here as it is committed, with the one fact the mapping file cannot keep:
who made it.

``_log/mappings.json`` holds the entries, oldest first::

    {"entries": [
      {"at": "2026-09-30T08:33:56Z", "by": "admin", "route": "update_vals_mapping.py",
       "subject": "alibaba/qwen3.8-27b", "value": "qwen3-8-27b", "previous": null,
       "commit": "ed36807", "run": "https://github.com/.../actions/runs/..."}
    ]}

``by`` is one of:

  admin     an explicit mapping answer sent from the admin page
  auto      a write the pipeline made on its own -- a mapping add.py re-pointed
            at a model it just added, a rename carried through every file, or
            anything a refresh wrote without being asked
  proposal  a proposal PR merged on GitHub (backfilled history only)
  hand      any other commit: a terminal session, a pull request of code

``value`` and ``previous`` are the mapping's value after and before -- a string,
or a list for the Artificial Analysis file, whose values may be one; ``null``
means the key was absent. ``why`` is added when the cause is known: the rename
or the added model that a side-effect write followed.

The file lives under ``_log/`` rather than beside the queue in ``_pending/``
because it is not a question, and under an underscore directory for the same
reason ``_pending/`` is: Jekyll does not publish it, while raw.githubusercontent
still serves it to the admin page. It is capped at MAX_ENTRIES, newest kept --
a history of mapping *decisions* is worth a long tail, but the page reads the
whole file on a phone.

Entries are appended, never rewritten: :func:`record` compares the mapping files
on disk against a git revision (``HEAD`` by default, i.e. what the commit about
to be made changes) and adds what moved; :func:`backfill` walks the first-parent
history once to seed the file from what git already knows.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import _answers
import propose

HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / "_log" / "mappings.json"
MAX_ENTRIES = 4000

ADMIN = "admin"
AUTO = "auto"
PROPOSAL = "proposal"
HAND = "hand"
BY = frozenset({ADMIN, AUTO, PROPOSAL, HAND})

# The commit subjects the workflow writes, which is how backfill tells the three
# automated kinds of commit apart. Matched as substrings: a rebase or a squash
# may add a prefix, never remove these words.
ADMIN_SUBJECT = "apply answers from the admin UI"
PROPOSAL_BRANCH = "chore/mapping-proposals"
BOT_AUTHOR = "github-actions[bot]"


@dataclass(frozen=True)
class Tracked:
    """One mapping file, and the route whose questions it answers."""

    route: str
    relpath: str


def tracked_files() -> list[Tracked]:
    """Every mapping file a route owns, read off propose.ROUTES.

    One entry per file: llm-stats has two kinds under one route, each with a
    file of its own, and two routes never share one.
    """
    seen: dict[Path, Tracked] = {}
    for name, by_kind in sorted(propose.ROUTES.items()):
        for route in by_kind.values():
            path = _answers.mapping_path(route)
            seen.setdefault(path, Tracked(name, path.relative_to(HERE).as_posix()))
    return sorted(seen.values(), key=lambda t: t.relpath)


# --------------------------------------------------------------------------
# reading two versions of one file
# --------------------------------------------------------------------------


def _as_mapping(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _git(args: list[str], cwd: Path = HERE) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def _show(rev: str, relpath: str, cwd: Path = HERE) -> str | None:
    """The file at `rev`, or None when it did not exist there."""
    proc = subprocess.run(
        ["git", "show", f"{rev}:{relpath}"], cwd=cwd, capture_output=True, text=True
    )
    return proc.stdout if proc.returncode == 0 else None


def diff(before: dict[str, Any], after: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """(key, previous, value) for every key whose value differs, sorted by key."""
    changed = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key), after.get(key)
        if old != new:
            changed.append((key, old, new))
    return changed


# --------------------------------------------------------------------------
# who made a change
# --------------------------------------------------------------------------


def _norm(value: Any) -> list[str] | None:
    """A mapping value as a list, so "x" and ["x"] compare equal."""
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


@dataclass(frozen=True)
class Batch:
    """What an admin batch asked for, as answer.py's --result reports it.

    Only validated records reach that file, so nothing here is taken from the
    dispatch input directly.
    """

    mappings: frozenset[tuple[str, str]]          # (route, subject) answered explicitly
    values: dict[tuple[str, str], list[str] | None]
    renames: dict[str, str]                       # old name -> new name
    added: frozenset[str]                         # models the batch created

    @classmethod
    def empty(cls) -> "Batch":
        return cls(frozenset(), {}, {}, frozenset())

    @classmethod
    def from_result(cls, path: Path | None) -> "Batch":
        if path is None:
            return cls.empty()
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls.empty()
        records = doc.get("records") if isinstance(doc, dict) else None
        mappings: set[tuple[str, str]] = set()
        values: dict[tuple[str, str], list[str] | None] = {}
        renames: dict[str, str] = {}
        added: set[str] = set()
        for record in records or []:
            if not isinstance(record, dict):
                continue
            kind = record.get("kind")
            subject = str(record.get("subject") or "")
            if kind == _answers.MAPPING and record.get("route"):
                key = (str(record["route"]), subject)
                mappings.add(key)
                values[key] = _norm(record.get("value"))
            elif kind == _answers.MODEL_RENAME and record.get("value"):
                renames[subject] = str(record["value"])
            elif kind in (_answers.MODEL_ADD, _answers.MODEL_CREATE, _answers.REFERENCE_ADD):
                added.add(subject)
        return cls(frozenset(mappings), values, renames, frozenset(added))

    def classify(self, route: str, key: str, previous: Any, value: Any) -> tuple[str, str | None]:
        """(by, why) for one change the batch's commit carries."""
        if (route, key) in self.mappings and self.values.get((route, key)) == _norm(value):
            return ADMIN, None
        new = _norm(value) or []
        old = _norm(previous) or []
        for was, now in self.renames.items():
            # A value repointed from the old name, or the AA file's key moving.
            if (was in old and now in new) or key in (was, now):
                return AUTO, f"follows the rename of {was} to {now}"
        for name in sorted(self.added):
            if name in new and name not in old:
                return AUTO, f"re-pointed at {name}, added in the same batch"
        return AUTO, None


# --------------------------------------------------------------------------
# the file
# --------------------------------------------------------------------------


def load(path: Path = LOG_PATH) -> list[dict[str, Any]]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = doc.get("entries") if isinstance(doc, dict) else None
    return [e for e in entries or [] if isinstance(e, dict)]


def save(entries: list[dict[str, Any]], path: Path = LOG_PATH) -> None:
    kept = entries[-MAX_ENTRIES:]
    path.parent.mkdir(parents=True, exist_ok=True)
    # One entry per line: appending a run's changes is then a diff of exactly
    # those lines, and the file is half the size an indented dump would be.
    lines = ",\n".join(json.dumps(e, ensure_ascii=False, sort_keys=True) for e in kept)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        '{"entries": [\n' + lines + ("\n" if lines else "") + "]}\n", encoding="utf-8"
    )
    temporary.replace(path)


def _entry(
    *, at: str, by: str, tracked: Tracked, key: str, previous: Any, value: Any,
    commit: str | None = None, run: str | None = None, why: str | None = None,
    inferred: bool = False,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "at": at,
        "by": by,
        "route": tracked.route,
        "subject": key,
        "value": value,
        "previous": previous,
    }
    if commit:
        entry["commit"] = commit
    if run:
        entry["run"] = run
    if why:
        entry["why"] = why
    if inferred:
        # Classified from the commit it came in rather than recorded as it
        # happened: an admin batch's side effects read as "admin" here.
        entry["inferred"] = True
    return entry


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def record(
    by: str,
    *,
    base: str = "HEAD",
    batch: Batch | None = None,
    run: str | None = None,
    at: str | None = None,
    cwd: Path = HERE,
    files: Iterable[Tracked] | None = None,
) -> list[dict[str, Any]]:
    """The changes the working tree makes to the mapping files since `base`.

    Called just before a commit, so `base` is the commit's parent and the
    working tree is the commit. `by` is what an unexplained change is credited
    to; with a `batch`, each change is classified against what the batch asked
    for instead.
    """
    if by not in BY:
        raise ValueError(f"unknown author kind {by!r}")
    stamp = at or now_utc()
    found: list[dict[str, Any]] = []
    for tracked in files if files is not None else tracked_files():
        path = cwd / tracked.relpath
        before = _as_mapping(_show(base, tracked.relpath, cwd))
        after = _as_mapping(path.read_text(encoding="utf-8") if path.exists() else None)
        for key, previous, value in diff(before, after):
            who, why = (batch.classify(tracked.route, key, previous, value)
                        if batch is not None else (by, None))
            found.append(_entry(at=stamp, by=who, tracked=tracked, key=key,
                                previous=previous, value=value, run=run, why=why))
    return found


def classify_commit(author: str, subject: str) -> str:
    """Who made a commit, from what the workflow and GitHub write into it."""
    if ADMIN_SUBJECT in subject:
        return ADMIN
    if PROPOSAL_BRANCH in subject or "propose mappings" in subject:
        return PROPOSAL
    if author == BOT_AUTHOR:
        return AUTO
    return HAND


def commit_changes(
    sha: str, *, by: str | None = None, cwd: Path = HERE
) -> list[dict[str, Any]]:
    """The mapping changes one commit made against its first parent.

    `by` defaults to what the commit's author and subject say (classify_commit).
    A root commit has no parent to diff against and yields nothing: its files
    are the starting point, not a change.
    """
    tracked = {t.relpath: t for t in tracked_files()}
    line = _git(["log", "-1", "--format=%H%x1f%P%x1f%an%x1f%cI%x1f%s", sha], cwd).strip()
    full, parents, author, when, subject = line.split("\x1f", 4)
    parent = parents.split()[0] if parents.strip() else None
    if parent is None:
        return []
    names = _git(["diff", "--name-only", parent, full, "--", *tracked], cwd).split()
    if not names:
        return []
    who = by or classify_commit(author, subject)
    at = datetime.fromisoformat(when).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    found: list[dict[str, Any]] = []
    for name in names:
        if name not in tracked:
            continue
        before = _as_mapping(_show(parent, name, cwd))
        after = _as_mapping(_show(full, name, cwd))
        for key, previous, value in diff(before, after):
            found.append(_entry(at=at, by=who, tracked=tracked[name], key=key,
                                previous=previous, value=value, commit=full[:7],
                                inferred=by is None and who == ADMIN))
    return found


def backfill(rev: str = "HEAD", cwd: Path = HERE) -> list[dict[str, Any]]:
    """Every mapping change on the first-parent history of `rev`, oldest first.

    First-parent, so a merged pull request counts once, as the merge that
    brought it onto main, rather than once per commit on its branch -- and the
    merge's diff against its first parent is exactly what it changed on main.
    An admin batch's commit cannot be split into what was asked and what
    followed from it after the fact, so its changes are all marked `inferred`.
    """
    tracked = [t.relpath for t in tracked_files()]
    shas = _git(["log", "--first-parent", "--reverse", "--format=%H", rev, "--", *tracked],
                cwd).split()
    found: list[dict[str, Any]] = []
    for sha in shas:
        found.extend(commit_changes(sha, cwd=cwd))
    return found
