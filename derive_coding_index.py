#!/usr/bin/env python3
"""Compute the Coding index column in llm.json from the coding benchmarks.

Unlike every other column, "coding_index" is not scraped: it is derived from
the coding benchmarks already in llm.json, so it has to be recomputed whenever
any of them changes (update-all does this after the scrapers have run).

The math is the one llm.html and llm-cli use for a sort group, so the column
ranks models the way the old "Coding (grouped)" sort did:

  * Every contributing benchmark is turned into a tie-averaged percentile rank
    across the models that have a score for it, which removes the scale
    differences between a pass rate and an index.
  * Percentiles are averaged with the per-benchmark reliability weights in
    CONTRIBUTING below, so the benchmarks worth trusting lead and the weaker
    ones fill gaps and break ties.
  * A missing score is imputed, not counted as zero: absence is ignorance, not
    evidence of a bottom rank. The fill starts at the median (percentile 0.5)
    and slides toward the level the model has actually demonstrated in
    proportion to its coverage, so the penalty for a blank grows with the
    square of the missing weight share. A model measured on almost everything
    is barely docked; one measured on almost nothing stays pinned near the
    median, so a single lucky score cannot top a well-tested model.
  * A benchmark nobody can be ranked on (fewer than two scored models) is left
    out of the total weight so it dilutes nobody.
  * A model measured on less than MIN_SCORED_FRACTION of the total weight is
    left unranked (null) rather than scored, because its value would be mostly
    imputation.

The weighted percentile is reported as index points: it is multiplied by SCALE
and rounded to a whole number, so a model scores something like 89136 rather
than 89.1 -- a percentage would read as a saturated score once a good model
neared 100, and these are ranks, not a share of tasks solved. SCALE is also
what keeps the ranking strict: neighbouring models can sit thousandths of a
percentile apart, so the scale has to be coarse enough to be readable and fine
enough that two distinct composites never round onto the same integer.

Because percentiles are relative to the current model set, every model's value
can move when a model or a score is added. That also means null is a real
result here, so -- unlike the scrapers, which never overwrite a value with null
because a source can drop out for a day -- this script does clear a value (and
its date) when a model no longer qualifies.

Default is a dry-run; pass -w/--write to persist changes (same convention as
update.py and prune.py).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from _scores import stamp_score_updated

DEFAULT_LLM_JSON = Path(__file__).resolve().parent / "llm.json"
JSON_DUMP_KWARGS = {"indent": 2, "ensure_ascii": False}

# The derived column this script fills. Its label, description and icon live in
# llm.json like every other benchmark's.
INDEX_KEY = "coding_index"

# Contributing benchmarks and their reliability weights. Weights are relative,
# so only their ratios matter; scale them all and the ranking does not move.
CONTRIBUTING: list[tuple[str, float]] = [
    ("deepswe", 1.0),
    ("frontierswe", 0.9),
    ("frontiercode", 0.9),
    ("swe_marathon", 0.9),
    ("terminal_bench_2_1", 0.85),
    ("swe_rebench", 0.8),
    ("swe_bench_pro", 0.4),
    ("livecodebench", 0.4),
    ("terminal_bench_2_0", 0.35),
    ("scicode", 0.35),
    ("swe_atlas_rf", 0.17),
    ("swe_atlas_tw", 0.17),
    ("swe_atlas_qna", 0.17),
    ("swe_bench_verified", 0.15),
]

# Below this share of the total weight a model is left unranked.
MIN_SCORED_FRACTION = 0.2

# Index points per full percentile: a model that tops every contributing
# benchmark scores SCALE, the median model half of it. Sized for headroom --
# the closest pair of models in the current set is ~15 points apart, so the
# model count would have to grow many times over before two of them collide.
SCALE = 100_000


def to_number(value: Any) -> float:
    """Numeric value of a score, or NaN for anything unusable (None, a bool, a
    non-numeric string)."""
    if isinstance(value, bool) or value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else math.nan
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return math.nan
    return math.nan


def nearly_equal(a: float, b: float) -> bool:
    return abs(a - b) < 1e-9


def is_lower_better(doc: dict[str, Any], key: str) -> bool:
    benchmark = (doc.get("benchmarks") or {}).get(key) or {}
    return benchmark.get("lower_is_better") is True


def percentile_map(
    models: list[dict[str, Any]], key: str, doc: dict[str, Any]
) -> dict[str, float] | None:
    """Model name -> percentile rank (0 worst .. 1 best) for one benchmark, or
    None when fewer than two models are scored on it and no rank exists."""
    entries = [
        (model["name"], to_number((model.get("scores") or {}).get(key)))
        for model in models
        if isinstance(model.get("name"), str)
    ]
    entries = [(name, value) for name, value in entries if math.isfinite(value)]
    if len(entries) < 2:
        return None

    entries.sort(key=lambda entry: entry[1])
    n = len(entries)
    lower = is_lower_better(doc, key)
    result: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and nearly_equal(entries[j + 1][1], entries[i][1]):
            j += 1
        pct = ((i + j) / 2) / (n - 1)  # average rank of the tie group
        for k in range(i, j + 1):
            # A percentile is "how good", so a lower-is-better benchmark flips:
            # the smallest value is the 1.0 end.
            result[entries[k][0]] = 1 - pct if lower else pct
        i = j + 1
    return result


def compute_index(
    models: list[dict[str, Any]], doc: dict[str, Any]
) -> dict[str, int | None]:
    """Model name -> Coding index in points (0..SCALE), or None when unranked."""
    pct_maps = [percentile_map(models, key, doc) for key, _ in CONTRIBUTING]
    weights = [
        weight if pct_map is not None else 0.0
        for (_, weight), pct_map in zip(CONTRIBUTING, pct_maps)
    ]
    total_weight = sum(weights)
    min_scored_weight = MIN_SCORED_FRACTION * total_weight

    result: dict[str, int | None] = {}
    for model in models:
        name = model.get("name")
        if not isinstance(name, str) or not name:
            continue

        sum_weight = 0.0
        sum_weighted = 0.0
        for weight, pct_map in zip(weights, pct_maps):
            if pct_map is None:
                continue
            pct = pct_map.get(name)
            if pct is None:
                continue
            sum_weight += weight
            sum_weighted += weight * pct

        if total_weight <= 0 or sum_weight <= 0 or sum_weight < min_scored_weight:
            result[name] = None
            continue

        missing_weight = total_weight - sum_weight
        own = sum_weighted / sum_weight
        fill = 0.5 + (own - 0.5) * (sum_weight / total_weight)
        composite = (sum_weighted + fill * missing_weight) / total_weight
        result[name] = round(composite * SCALE)
    return result


def scored_count(model: dict[str, Any]) -> int:
    """How many contributing benchmarks this model actually has a score on."""
    scores = model.get("scores") or {}
    return sum(
        1 for key, _ in CONTRIBUTING if math.isfinite(to_number(scores.get(key)))
    )


def put_first(mapping: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    """Same mapping with `key` set and moved to the front, so the score keys
    stay in the benchmark order llm.json declares (the Coding index is the
    first column)."""
    rest = {k: v for k, v in mapping.items() if k != key}
    return {key: value, **rest}


def validate(doc: dict[str, Any]) -> list[str]:
    problems = []
    benchmarks = doc.get("benchmarks")
    if not isinstance(benchmarks, dict):
        return ['"benchmarks" is missing or not an object']
    if INDEX_KEY not in benchmarks:
        problems.append(
            f'benchmark "{INDEX_KEY}" is not declared in "benchmarks"; '
            "add its label, description and icon there first"
        )
    missing = [key for key, _ in CONTRIBUTING if key not in benchmarks]
    if missing:
        problems.append(
            "contributing benchmarks absent from \"benchmarks\": " + ", ".join(missing)
        )
    if INDEX_KEY in {key for key, _ in CONTRIBUTING}:
        problems.append(f'"{INDEX_KEY}" cannot contribute to itself')
    return problems


def apply_index(
    doc: dict[str, Any], index: dict[str, int | None]
) -> list[tuple[str, int | None, int | None]]:
    """Write a computed index into the models of `doc`, in memory only, and
    return the (model, old, new) triples that moved.

    Nothing reaches disk here, so a caller can decide whether to save (main()
    saves only with -w) and a rewritten score plus its refreshed index land in
    one write instead of two.
    """
    changes: list[tuple[str, int | None, int | None]] = []
    for model in doc.get("models") or []:
        name = model.get("name")
        if not isinstance(name, str) or name not in index:
            continue

        scores = model.get("scores")
        if not isinstance(scores, dict):
            scores = {}
        old = scores.get(INDEX_KEY)
        new = index[name]
        model["scores"] = put_first(scores, INDEX_KEY, new)

        updated = model.get("scores_updated")
        if not isinstance(updated, dict):
            updated = {}
        model["scores_updated"] = put_first(updated, INDEX_KEY, updated.get(INDEX_KEY))

        if old != new:
            changes.append((name, old, new))
            if new is None:
                # No value, no date: an unranked model reports neither.
                model["scores_updated"][INDEX_KEY] = None
            else:
                stamp_score_updated(model, INDEX_KEY)
    return changes


def refresh(doc: dict[str, Any]) -> list[tuple[str, int | None, int | None]]:
    """Recompute the derived column in `doc` in memory; returns the changes.

    For the tools that write llm.json themselves -- edit.py -- so a hand-edited
    score cannot leave a stale index behind. Raises ValueError when llm.json is
    shaped wrong, which is a configuration error rather than something to paper
    over.
    """
    problems = validate(doc)
    if problems:
        raise ValueError("; ".join(problems))
    models = doc.get("models")
    if not isinstance(models, list):
        raise ValueError('"models" is missing or not a list')
    return apply_index(doc, compute_index(models, doc))


def fmt(value: int | None) -> str:
    # Grouped only in this script's output; llm.json keeps a plain integer.
    return "—" if value is None else f"{value:,}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(DEFAULT_LLM_JSON),
        help='Path to JSON file to read/update (default: "./llm.json" next to this script)',
    )
    parser.add_argument(
        "--top",
        "-t",
        type=int,
        default=15,
        help="How many top-ranked models to print (default: 15, 0 for none).",
    )
    parser.add_argument(
        "--write",
        "-w",
        action="store_true",
        help="Write changes back to the input JSON file (default is dry-run).",
    )
    args = parser.parse_args()

    path = Path(args.json_file)
    doc = json.loads(path.read_text(encoding="utf-8"))

    problems = validate(doc)
    if problems:
        for problem in problems:
            print(f"Error: {problem}", file=sys.stderr)
        return 1

    models = doc.get("models")
    if not isinstance(models, list):
        print(f'Error: "models" in {path} is missing or not a list', file=sys.stderr)
        return 1

    index = compute_index(models, doc)

    ranked = sum(1 for value in index.values() if value is not None)
    print(
        f"{len(index)} model(s): {ranked} ranked, {len(index) - ranked} unranked "
        f"(< {MIN_SCORED_FRACTION:.0%} of the weight of "
        f"{len(CONTRIBUTING)} coding benchmarks)"
    )

    # In memory only; the file is written further down, and only with -w.
    changes = apply_index(doc, index)

    if args.top:
        best = sorted(
            ((value, name) for name, value in index.items() if value is not None),
            reverse=True,
        )[: args.top]
        if best:
            print(f"\nTop {len(best)}:")
            by_name = {
                model["name"]: model
                for model in models
                if isinstance(model.get("name"), str)
            }
            for rank, (value, name) in enumerate(best, start=1):
                measured = scored_count(by_name[name])
                print(
                    f"  {rank:2d}. {fmt(value):>9s}  {name:40s} "
                    f"{measured}/{len(CONTRIBUTING)} measured"
                )

    if not changes:
        print(f"\n{INDEX_KEY} is up to date. Nothing to do.")
        return 0

    print(f"\n{len(changes)} value(s) change:")
    for name, old, new in sorted(changes, key=lambda c: c[0]):
        print(f"  {name:40s} {fmt(old):>9s} -> {fmt(new):>9s}")

    if not args.write:
        print("\ndry-run only, pass --write to persist changes")
        return 0

    path.write_text(json.dumps(doc, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
