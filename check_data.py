#!/usr/bin/env python3
"""The one check update-benchmarks.yml runs before it commits llm.json.

    ./check_data.py                 llm.json against the committed HEAD
    ./check_data.py FILE --base REF

The refresh used to run twenty test suites before it fetched anything -- a
minute and a half of every run, guarding code that had not changed since the
last run passed them. full-tests.yml now runs every suite daily, so the
refresh keeps only what no test run can do for it: look at the file it is
about to publish. Everything here reads llm.json and nothing else, and the
whole check takes well under a second.

What it refuses, each a way a refresh can damage the published data:

  * a file that does not parse, or has no models -- a write cut short;
  * models or scores vanishing en masse against HEAD -- a writer or a source
    gone wrong, nulling what it should have kept (SHRINK_LIMIT);
  * scores_history out of step with the scores (_history.validate), or a
    score left unrecorded -- the history is the only record of what a score
    used to be, and the next write would lose it for good;
  * a score that names no source page (test_hand_sources.py's rule);
  * a score that is not a finite number.

Exit status: 0 when the file may be committed, 1 with a line per problem when
it may not.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import _history
from _scores import editable_benchmarks, score_source

HERE = Path(__file__).resolve().parent

# The share of HEAD's models, or of its non-null scores, that may disappear in
# one write. A run adds and drops a handful (a model ignored, a score a source
# withdrew); a writer that broke drops far more.
SHRINK_LIMIT = 0.10


def score_count(doc: dict[str, Any]) -> int:
    return sum(
        1
        for model in doc.get("models") or []
        for value in (model.get("scores") or {}).values()
        if value is not None
    )


def load_base(ref: str, path: Path) -> dict[str, Any] | None:
    """llm.json as committed at `ref`, or None when there is nothing to compare
    with (no git, a first commit, a file outside the repository)."""
    try:
        relative = path.resolve().relative_to(HERE)
        shown = subprocess.run(
            ["git", "-C", str(HERE), "show", f"{ref}:{relative.as_posix()}"],
            capture_output=True,
            check=True,
        )
        return json.loads(shown.stdout)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def check(doc: Any, base: dict[str, Any] | None) -> list[str]:
    if not isinstance(doc, dict) or not isinstance(doc.get("models"), list) or not doc["models"]:
        return ["llm.json has no models"]
    problems: list[str] = []
    models = doc["models"]

    if base and isinstance(base.get("models"), list) and base["models"]:
        names = {model.get("name") for model in models}
        lost = sorted(
            str(model.get("name"))
            for model in base["models"]
            if model.get("name") not in names
        )
        if len(lost) > SHRINK_LIMIT * len(base["models"]):
            problems.append(
                f"{len(lost)} of {len(base['models'])} models disappeared: "
                + ", ".join(lost[:10])
                + (" ..." if len(lost) > 10 else "")
            )
        before, after = score_count(base), score_count(doc)
        if before and after < (1 - SHRINK_LIMIT) * before:
            problems.append(f"scores fell from {before} to {after}")

    for model in models:
        for key, value in (model.get("scores") or {}).items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                problems.append(f"{model.get('name')}.{key} is not a number: {value!r}")

    problems.extend(f"history: {problem}" for problem in _history.validate(doc))
    unrecorded = _history.sync(copy.deepcopy(doc))
    if unrecorded:
        problems.append(f"history: {unrecorded} score change(s) left unrecorded")

    keys = set(editable_benchmarks(doc))
    unsourced = [
        f"{model.get('name')}.{key}"
        for model in models
        for key, value in (model.get("scores") or {}).items()
        if key in keys and value is not None and score_source(model, key) is None
    ]
    if unsourced:
        problems.append(
            f"{len(unsourced)} score(s) name no source page: " + ", ".join(unsourced[:10])
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("file", nargs="?", default=str(HERE / "llm.json"))
    parser.add_argument("--base", default="HEAD", help="git ref to compare with (default HEAD)")
    args = parser.parse_args()
    path = Path(args.file)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"check_data: {path} does not parse: {exc}", file=sys.stderr)
        return 1
    problems = check(doc, load_base(args.base, path))
    if problems:
        print(f"check_data: refusing {path.name}:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"check_data: {path.name} ok ({len(doc['models'])} models, {score_count(doc)} scores)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
