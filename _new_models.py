#!/usr/bin/env python3
"""New-model decisions: what the proposal PR asks, and what the next run applies.

check_new.py finds open-weights models on Artificial Analysis that llm.json does
not track yet. Interactively it asks one question per model -- add it, or never
ask again. Unattended (the GitHub Actions run) it can answer neither, so the
question travels to the proposal PR instead. That PR has to carry *both*
answers, or the only reviewable outcome is "add" and declining a model means
editing files by hand.

The second answer is a line in check_new-decisions.json:

    { "some-new-model": "__added__" }

``__added__`` is what propose.py wrote: the model is in llm.json in the same PR,
prefilled from AA with null scores. Flipping that line to ``__ignored__`` -- one
click on the suggestion the PR attaches -- is the other answer: drop the entry
from llm.json again and record the slug in check_new-dismissed.json, so AA never
offers it again.

apply_decisions() carries that out, and runs at the top of check_new.py, before
any score is fetched -- so an ignored model is gone before the mapping scripts
and update.py could attach anything to it.

It is the one writer here that collect mode does *not* freeze. The freeze exists
so that an unattended run never records an answer nobody gave; these lines are
answers a person gave, in a PR they merged, and the unattended run is the only
place they are ever carried out. Freezing them would mean an ignored model is
re-added on the next run, forever.
"""

from __future__ import annotations

import json
from pathlib import Path

import _prompts

HERE = Path(__file__).resolve().parent
DECISIONS_FILE = HERE / "check_new-decisions.json"
DISMISSED_FILE = HERE / "check_new-dismissed.json"
DEFAULT_LLM_JSON = HERE / "llm.json"

# Written by propose.py for a model it added to llm.json in the same PR. Not a
# question mark: merging it as-is keeps the model, and the entry is consumed by
# the next run.
ADDED = "__added__"

# The reviewer's other answer, one click away in the PR: undo the addition and
# stop offering the slug.
IGNORED = "__ignored__"

DECISIONS = frozenset({ADDED, IGNORED})

JSON_DUMP_KWARGS = {"indent": 2, "ensure_ascii": False}


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# check_new-dismissed.json -- slugs never to offer again
# --------------------------------------------------------------------------


def load_dismissed(path: Path | None = None) -> set[str]:
    path = path or DISMISSED_FILE
    if not path.exists():
        return set()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return set(raw) if isinstance(raw, list) else set()


def dismiss(slug: str, path: Path | None = None, *, force: bool = False) -> bool:
    """Record a slug as never-offer-again. Returns True when the file changed.

    force=True is for apply_decisions(): collect mode freezes the *prompt*
    answers so they are asked again, but a decision already recorded in a merged
    PR has to land even in an unattended run -- that run is the only one that
    ever sees it.
    """
    if not force and _prompts.freeze_decisions():
        return False
    path = path or DISMISSED_FILE
    slugs = load_dismissed(path)
    if slug in slugs:
        return False
    slugs.add(slug)
    _write_json(path, sorted(slugs))
    return True


# --------------------------------------------------------------------------
# check_new-decisions.json -- the question the proposal PR carries
# --------------------------------------------------------------------------


def load_decisions(path: Path | None = None) -> dict[str, str]:
    path = path or DECISIONS_FILE
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def write_decisions(decisions: dict[str, str], path: Path | None = None) -> None:
    """Rewrite the file, sorted. Emptied means ``{}``, not deleted: the file is
    part of the tree so its format stays documented and its diffs stay small."""
    path = path or DECISIONS_FILE
    _write_json(path, {k: decisions[k] for k in sorted(decisions)})


def record_proposed(slug: str, path: Path | None = None) -> None:
    """Put a model propose.py just added in front of the reviewer as ``__added__``.

    An entry the file already carries is left alone: overwriting an ``__ignored__``
    a reviewer wrote would answer their question for them.
    """
    path = path or DECISIONS_FILE
    decisions = load_decisions(path)
    if slug in decisions:
        return
    decisions[slug] = ADDED
    write_decisions(decisions, path)


def find_line(slug: str, path: Path | None = None) -> int | None:
    """1-based line of a slug's entry, for hanging a review comment on it."""
    path = path or DECISIONS_FILE
    if not path.exists():
        return None
    needle = json.dumps(slug, ensure_ascii=False) + ":"
    for index, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if text.strip().startswith(needle):
            return index
    return None


# --------------------------------------------------------------------------
# Applying what the reviewer decided
# --------------------------------------------------------------------------


def remove_model(doc: dict, name: str) -> bool:
    models = doc.get("models")
    if not isinstance(models, list):
        return False
    kept = [m for m in models if not (isinstance(m, dict) and m.get("name") == name)]
    if len(kept) == len(models):
        return False
    doc["models"] = kept
    return True


def _score_count(doc: dict, name: str) -> int:
    for model in doc.get("models", []):
        if isinstance(model, dict) and model.get("name") == name:
            return sum(1 for v in (model.get("scores") or {}).values() if v is not None)
    return 0


def apply_decisions(
    llm_path: Path | None = None,
    decisions_path: Path | None = None,
    dismissed_path: Path | None = None,
) -> list[str]:
    """Carry out the merged decisions and clear the ones that are done.

    Returns one log line per decision acted on (empty when there is nothing to
    do), so the caller can print them into the run's log.
    """
    decisions_path = decisions_path or DECISIONS_FILE
    decisions = load_decisions(decisions_path)
    if not decisions:
        return []

    llm_path = llm_path or DEFAULT_LLM_JSON
    doc = json.loads(llm_path.read_text(encoding="utf-8"))

    log: list[str] = []
    remaining: dict[str, str] = {}
    llm_changed = False

    for slug, value in sorted(decisions.items()):
        if value == IGNORED:
            scored = _score_count(doc, slug)
            if remove_model(doc, slug):
                llm_changed = True
                # Worth saying: a scored entry means a refresh got to the model
                # before this decision did. The decision still wins -- it is the
                # newer answer -- but the log should not hide what it dropped.
                extra = f" (dropped {scored} score(s))" if scored else ""
                log.append(f"  {slug}: ignored -> removed from {llm_path.name}{extra}")
            else:
                log.append(f"  {slug}: ignored -> not in {llm_path.name}, nothing to remove")
            dismiss(slug, dismissed_path, force=True)
            log.append(f"  {slug}: recorded in {(dismissed_path or DISMISSED_FILE).name}")
        elif value == ADDED:
            if any(
                isinstance(m, dict) and m.get("name") == slug for m in doc.get("models", [])
            ):
                log.append(f"  {slug}: kept; the entry in {llm_path.name} is the record now")
            else:
                # Merged as "added" but absent: someone removed the entry by
                # hand. Nothing to undo, and dismissing it would answer a
                # question they did not answer.
                log.append(f"  {slug}: kept, but absent from {llm_path.name}; left alone")
        else:
            remaining[slug] = value
            log.append(
                f"  {slug}: unknown decision {value!r}; left in "
                f"{decisions_path.name} (expected {ADDED} or {IGNORED})"
            )
            continue
        # Decided either way: the question is answered and must not come back.

    if llm_changed:
        _write_json(llm_path, doc)
    if remaining != decisions:
        write_decisions(remaining, decisions_path)
    if log:
        log.insert(0, f"Applying {len(decisions)} recorded new-model decision(s):")
    return log
