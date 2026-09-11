#!/usr/bin/env python3
"""The closed-weight models the index carries for reference only.

llm.json tracks open-weight models. A handful of closed frontier models are
carried alongside them anyway, because "how far behind is the open field?" is
the question every number on the page is really answering, and it cannot be
read off a table that holds only one side of it.

`reference-models.json` is that list, and it holds nothing but Artificial
Analysis slugs -- the same spelling every other mapping file resolves to, so a
reference model is added exactly the way an open one is: name the llm.json
entry after its AA slug and every scraper's mapping lands on it (see
"A hand-added model meets Artificial Analysis" in README.md).

What the list changes, wherever it is read:

  * llm.html / llm-cli hide these rows unless the reader asks for them (the
    "Closed models" checkbox in the filter panel's head band), and colour their
    names apart when shown. A model's own page and a comparison always carry
    them -- that is the whole point of having them.
  * derive_indexes.py ranks the open field among itself, so a reference model
    never moves an open model's index. The reference rows are then ranked into
    the combined field, which is what makes them comparable at all.
  * _openness.py never lets a source's "this model is closed" verdict bury a
    name that belongs to one of these models -- that verdict is right, and for
    every other closed model it is also the correct action, which is exactly
    why the exception has to be explicit.

Deliberately not a flag on the llm.json entries alone: the entries carry
`"reference": true` for consumers that read only llm.json, but this file is
what decides it, and apply_reference_flags() is what keeps the two in step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

# Every reader and writer below resolves this at call time rather than binding
# it as a default argument, so a caller -- a test, a second checkout -- can
# point the whole module at another file by rebinding the name.
REFERENCE_MODELS = Path(__file__).resolve().with_name("reference-models.json")

# The llm.json field that carries the verdict to consumers which read only
# llm.json -- the web page, the CLI, an export.
REFERENCE_FIELD = "reference"

# The list may never be emptied. Not an arbitrary floor: with none of these
# rows the open field is back to being measured against nothing, which is the
# state this file exists to end -- and the page would carry a toggle, a colour
# and a tab about a set with no members. Taking the last one off is therefore
# refused rather than obeyed; deleting the file is the way to mean it.
MIN_REFERENCE_MODELS = 1

# The page a reference row is checked against. A closed model has no weights
# repository, so the Hugging Face card every open row links to does not exist;
# Artificial Analysis' model page is where its numbers are published and is
# what `url` means for these entries. Spelled out rather than imported because
# this module deliberately pulls in nothing -- test_reference.py holds it to
# artificialanalysis.MODEL_PAGE_URL so the two cannot drift.
AA_MODEL_PAGE_URL = "https://artificialanalysis.ai/models/{}"


def aa_model_page(slug: str) -> str:
    return AA_MODEL_PAGE_URL.format(slug)


def load_reference_slugs(path: Path | None = None) -> list[str]:
    """Every Artificial Analysis slug the reference list names, in file order.

    A missing file is an empty list rather than an error: the list is an
    addition to the index, and nothing here should stop working without it.
    """
    path = path or REFERENCE_MODELS
    if not path.exists():
        return []
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array of AA slugs in {path}")
    slugs: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip() and item not in slugs:
            slugs.append(item.strip())
    return slugs


def reference_slug_set(path: Path | None = None) -> set[str]:
    return set(load_reference_slugs(path))


def write_reference_slugs(slugs: list[str], path: Path | None = None) -> None:
    """Rewrite the list, keeping the file's shape (one slug per line)."""
    path = path or REFERENCE_MODELS
    path.write_text(
        json.dumps(slugs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def add_reference_slug(slug: str, path: Path | None = None) -> bool:
    """Put an AA slug on the list. False if it was already there.

    Sorted on the way in, because the file is read by people as often as by
    scripts and an append-ordered list of seven turns into an unreadable one of
    twenty. Adding the slug does not create the llm.json entry -- see
    missing_reference_models(); _answers.py is what does both at once.
    """
    slugs = load_reference_slugs(path)
    if slug in slugs:
        return False
    write_reference_slugs(sorted([*slugs, slug]), path)
    return True


def remove_reference_slug(slug: str, path: Path | None = None) -> bool:
    """Take an AA slug off the list. False if it was not on it.

    Refuses to empty the list (MIN_REFERENCE_MODELS). The caller is expected to
    have said so first -- _answers.validate() refuses the batch, with the
    message the sender reads -- so reaching the ValueError here means something
    bypassed that, and a raise beats quietly leaving an index with nothing to
    be measured against.
    """
    slugs = load_reference_slugs(path)
    if slug not in slugs:
        return False
    kept = [s for s in slugs if s != slug]
    if len(kept) < MIN_REFERENCE_MODELS:
        raise ValueError(
            f"{slug!r} is the last reference model; the list may not be emptied"
        )
    write_reference_slugs(kept, path)
    return True


def rename_reference_slug(
    old: str, new: str, path: Path | None = None
) -> bool:
    """Follow a renamed model onto the list. True if the list held `old`.

    The list is keyed by model name like every mapping file, so a rename that
    skipped it would leave the entry silently no longer a reference row --
    still in llm.json, still closed, but back in the open field's ranking. This
    is what _rename.py calls so that cannot happen; the flag itself is fixed by
    the next apply_reference_flags().
    """
    slugs = load_reference_slugs(path)
    if old not in slugs:
        return False
    write_reference_slugs([new if slug == old else slug for slug in slugs], path)
    return True


def is_reference_model(model: dict[str, Any], slugs: Iterable[str] | None = None) -> bool:
    """Whether one llm.json model entry is a closed reference row.

    The list decides; the stored flag is the fallback, so a doc read without
    the file next to it (a downloaded llm.json) still knows which rows these
    are.
    """
    known = reference_slug_set() if slugs is None else set(slugs)
    name = model.get("name")
    if isinstance(name, str) and name in known:
        return True
    return model.get(REFERENCE_FIELD) is True


def split_models(
    models: list[dict[str, Any]], slugs: Iterable[str] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(open-weight models, reference models), in the order they were given."""
    known = reference_slug_set() if slugs is None else set(slugs)
    open_models = [m for m in models if not is_reference_model(m, known)]
    reference = [m for m in models if is_reference_model(m, known)]
    return open_models, reference


def reference_names(doc: dict[str, Any]) -> set[str]:
    """The names of the reference models this document actually carries."""
    models = doc.get("models")
    if not isinstance(models, list):
        return set()
    known = reference_slug_set()
    return {
        model["name"]
        for model in models
        if isinstance(model.get("name"), str) and is_reference_model(model, known)
    }


def missing_reference_models(doc: dict[str, Any]) -> list[str]:
    """Slugs on the list that llm.json has no entry for.

    Adding a slug to the list does not create the model -- ./add.py does, and
    the entry has to be named after the slug. This is what says so out loud
    instead of leaving the addition quietly inert.
    """
    models = doc.get("models")
    present = (
        {model.get("name") for model in models if isinstance(model, dict)}
        if isinstance(models, list)
        else set()
    )
    return [slug for slug in load_reference_slugs() if slug not in present]


def apply_reference_flags(doc: dict[str, Any]) -> list[tuple[str, bool]]:
    """Stamp `reference` on every model to match the list. Returns the changes.

    Idempotent, and it clears the flag as well as sets it: a slug taken off the
    list has to stop being a reference row, or it would stay hidden from the
    table with nothing left saying why.
    """
    models = doc.get("models")
    if not isinstance(models, list):
        return []
    known = reference_slug_set()
    changes: list[tuple[str, bool]] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        name = model.get("name")
        if not isinstance(name, str) or not name:
            continue
        wanted = name in known
        current = model.get(REFERENCE_FIELD) is True
        if wanted == current:
            continue
        model.pop(REFERENCE_FIELD, None)
        if wanted:
            # Second key, right behind the name it qualifies: the flag decides
            # how every consumer reads the rest of the entry, so it should not
            # be something you find at the bottom of a hundred scores.
            rest = list(model.items())
            model.clear()
            for index, (field, value) in enumerate(rest):
                model[field] = value
                if index == 0:
                    model[REFERENCE_FIELD] = True
        changes.append((name, wanted))
    return changes
