"""Candidates a person has ruled out for one queued name, without answering it.

A mapping question has three answers that close it -- a model, a benchmark or
a slug, or __unmappable__ -- and sometimes none of them is true yet. llm-stats
lists ``minimax-m3-1-flash-preview``; the index has ``minimax-m3`` and
``minimax-m2-1``, and neither is that model. Mapping it to one would fold two
models' scores together, and __unmappable__ would bury the name for good, so
that on the day the index adds the real model nobody is asked again.

This is the fourth answer: *not these*. The candidates named are recorded
against the name, the queue stops offering them, and a question whose every
candidate has been ruled out leaves the queue -- until the matcher finds a
candidate that has not been, at which point it is asked again with only the
new ones. Nothing is written to the mapping file, so the name stays unreviewed
and the source keeps asking about it.

The Artificial Analysis route already had exactly this, written by its own
interactive prompt into ``model-name-mapping-llm-to-artificialanalysis-
ignored.json`` and honoured by its updater, so that route keeps using that
file. Every other route shares ``rejected-candidates.json``::

    {"update_vals_mapping.py": {"alibaba/qwen3.8-max": ["qwen3-8-27b"]}}

keyed by route, then by the name the source publishes. A rejection belongs to
that one name: the same candidate is still offered for every other name it
matches. The file is named
without "mapping" on purpose: it is not one of the source mapping files, and
the checks that treat every ``*mapping*.json`` as one would misread it.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Iterable

from _prompts import freeze_decisions

HERE = Path(__file__).resolve().parent
REJECTIONS = HERE / "rejected-candidates.json"

AA_ROUTE = "update_artificialanalysis_mapping.py"
AA_MODULE = "_artificialanalysis_mapping"


def _aa():
    return importlib.import_module(AA_MODULE)


def load(path: Path | None = None) -> dict[str, dict[str, set[str]]]:
    # Read at call time rather than bound as a default, so a test can point it
    # somewhere else.
    path = path or REJECTIONS
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    loaded: dict[str, dict[str, set[str]]] = {}
    for route, by_subject in raw.items():
        if not isinstance(route, str) or not isinstance(by_subject, dict):
            continue
        for subject, options in by_subject.items():
            if isinstance(subject, str) and isinstance(options, list):
                loaded.setdefault(route, {})[subject] = {o for o in options if isinstance(o, str)}
    return loaded


def write(rejections: dict[str, dict[str, set[str]]], path: Path | None = None) -> None:
    if freeze_decisions():
        return
    path = path or REJECTIONS
    doc = {
        route: {subject: sorted(options) for subject, options in sorted(by_subject.items()) if options}
        for route, by_subject in sorted(rejections.items())
    }
    doc = {route: by_subject for route, by_subject in doc.items() if by_subject}
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")


def rejected(route: str, subject: str, path: Path | None = None) -> set[str]:
    """The candidates ruled out for `subject` on `route`."""
    if route == AA_ROUTE:
        return set(_aa().load_ignored_aa_suggestions().get(subject, set()))
    return set(load(path).get(route, {}).get(subject, set()))


def add(route: str, subject: str, options: Iterable[str], path: Path | None = None) -> Path:
    """Rule `options` out for `subject`; returns the file written."""
    options = [o for o in options if o]
    if route == AA_ROUTE:
        module = _aa()
        module.add_ignored_aa_suggestions(subject, options)
        return module.AA_MODEL_IGNORES
    rejections = load(path)
    rejections.setdefault(route, {}).setdefault(subject, set()).update(options)
    write(rejections, path)
    return path or REJECTIONS


def path_for(route: str) -> Path:
    """The file a rejection on `route` is written to."""
    return _aa().AA_MODEL_IGNORES if route == AA_ROUTE else REJECTIONS
