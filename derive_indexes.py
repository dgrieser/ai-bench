#!/usr/bin/env python3
"""Compute the derived index columns in llm.json from the benchmarks they
aggregate: the Coding index from the coding benchmarks, the Tooling index from
the agentic tool-use benchmarks.

Unlike every other column, a derived index is not scraped: it is computed from
scores already in llm.json, so it has to be recomputed whenever any of them
changes (update-all does this after the scrapers have run).

The math is the one llm.html and llm-cli use for a sort group, so each column
ranks models the way the old "Coding (grouped)" sort did:

  * Every contributing benchmark is turned into a tie-averaged percentile rank
    across the models that have a score for it, which removes the scale
    differences between a pass rate and an index.
  * Percentiles are averaged with the per-benchmark reliability weights in
    INDEXES below, so the benchmarks worth trusting lead and the weaker
    ones fill gaps and break ties.
  * A missing score is imputed, not counted as zero: absence is ignorance, not
    evidence of a bottom rank. The fill starts at the median (percentile 0.5)
    and slides toward the level the model has actually demonstrated in
    proportion to its coverage, but is capped at the median: a gap counts as
    an unknown opponent, and an unknown opponent is never assumed better than
    the median model. A blank can hold a strong model back or drag a weak one
    down, yet it can never lift anyone, so a sparsely measured model cannot
    outrank a well-tested one on imputed strength alone.
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

No leaderboard publishes these columns, so a value is attributed to the page
that documents how it is derived: the first URL the column declares in llm.json
(a section of this repository's README).

Default is a dry-run; pass -w/--write to persist changes (same convention as
update.py and prune.py).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, NamedTuple

from _scores import stamp_score_updated

DEFAULT_LLM_JSON = Path(__file__).resolve().parent / "llm.json"
JSON_DUMP_KWARGS = {"indent": 2, "ensure_ascii": False}


class IndexDef(NamedTuple):
    """One derived column: the key it fills, the page a value cites when
    llm.json declares no URL for the column (there is no leaderboard behind a
    derived score, so the source is the README section documenting the method),
    and the contributing benchmarks with their reliability weights. Weights are
    relative, so only their ratios matter; scale them all and the ranking does
    not move."""

    key: str
    fallback_source_url: str
    contributing: list[tuple[str, float]]


# The derived columns this script fills, in the order their score keys are
# kept in each model's maps. Label, description and icon live in llm.json like
# every other benchmark's.
INDEXES: list[IndexDef] = [
    IndexDef(
        key="coding_index",
        fallback_source_url="https://github.com/dgrieser/ai-bench#coding-index",
        contributing=[
            ("deepswe", 1.0),
            ("frontierswe", 0.9),
            ("frontiercode", 0.9),
            ("swe_marathon", 0.9),
            ("terminal_bench_2_1", 0.85),
            ("swe_bench_pro", 0.4),
            ("livecodebench", 0.4),
            ("scicode", 0.35),
            ("swe_bench_multilingual", 0.3),
            # SWE Atlas contributes its Codebase Q&A track only: every model
            # scored on Refactoring or Test Writing is also scored on Q&A, so
            # the other two tracks add no coverage, correlate 0.94 with each
            # other and 0.77-0.89 with Q&A, and their weight only raised the
            # MIN_SCORED_FRACTION bar. One track at 0.25 instead of three at
            # 0.17 -- see README, "Why SWE Atlas contributes one track".
            ("swe_atlas_qna", 0.25),
            ("swe_bench_verified", 0.15),
        ],
    ),
    IndexDef(
        key="tooling_index",
        fallback_source_url="https://github.com/dgrieser/ai-bench#tooling-index",
        contributing=[
            ("tau3_bench_banking", 1.0),
            ("toolathlon", 0.9),
            ("mcp_atlas", 0.85),
            ("terminal_bench_2_1", 0.8),
            ("gdpval_aa", 0.7),
            ("itbench_aa", 0.6),
            ("bfcl_v4", 0.5),
            ("tau2_bench_telecom", 0.3),
            ("terminal_bench_hard", 0.3),
            ("ifbench", 0.2),
        ],
    ),
]

# Below this share of an index's total weight a model is left unranked. 0.18
# rather than a round 0.2 because model coverage clusters rather than spreading
# evenly: the coding group has 11 models measured on 19% of its weight and none
# between 19% and 20%, so 0.2 was cutting a natural block of two-benchmark
# models in half. The next cluster sits at 13-14%, and admitting it would make
# 90% of the ranked field less than half measured -- see README, "Why the
# evidence bar is 18%".
MIN_SCORED_FRACTION = 0.18

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


def source_url(doc: dict[str, Any], index: IndexDef) -> str:
    """The page every derived value is attributed to: the column's first URL in
    llm.json, so the link is spelled in one place and the benchmark card, the
    score tooltip and the Sources panel all point at the same page."""
    urls = ((doc.get("benchmarks") or {}).get(index.key) or {}).get("urls")
    if isinstance(urls, list):
        for url in urls:
            if isinstance(url, str) and url.strip():
                return url.strip()
    return index.fallback_source_url


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
    models: list[dict[str, Any]], doc: dict[str, Any], index: IndexDef
) -> dict[str, int | None]:
    """Model name -> index value in points (0..SCALE), or None when unranked."""
    pct_maps = [percentile_map(models, key, doc) for key, _ in index.contributing]
    weights = [
        weight if pct_map is not None else 0.0
        for (_, weight), pct_map in zip(index.contributing, pct_maps)
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
        fill = min(0.5 + (own - 0.5) * (sum_weight / total_weight), 0.5)
        composite = (sum_weighted + fill * missing_weight) / total_weight
        result[name] = round(composite * SCALE)
    return result


def scored_count(model: dict[str, Any], index: IndexDef) -> int:
    """How many contributing benchmarks this model actually has a score on."""
    scores = model.get("scores") or {}
    return sum(
        1
        for key, _ in index.contributing
        if math.isfinite(to_number(scores.get(key)))
    )


def put_first(mapping: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    """Same mapping with `key` set and moved to the front, so the score keys
    stay in the benchmark order llm.json declares (the derived indexes are the
    leading columns)."""
    rest = {k: v for k, v in mapping.items() if k != key}
    return {key: value, **rest}


def validate(doc: dict[str, Any]) -> list[str]:
    problems = []
    benchmarks = doc.get("benchmarks")
    if not isinstance(benchmarks, dict):
        return ['"benchmarks" is missing or not an object']
    index_keys = {index.key for index in INDEXES}
    for index in INDEXES:
        if index.key not in benchmarks:
            problems.append(
                f'benchmark "{index.key}" is not declared in "benchmarks"; '
                "add its label, description and icon there first"
            )
        missing = [key for key, _ in index.contributing if key not in benchmarks]
        if missing:
            problems.append(
                f'benchmarks contributing to "{index.key}" absent from '
                '"benchmarks": ' + ", ".join(missing)
            )
        derived = index_keys.intersection(key for key, _ in index.contributing)
        if derived:
            problems.append(
                f'"{index.key}" cannot aggregate a derived column: '
                + ", ".join(sorted(derived))
            )
    return problems


def apply_index(
    doc: dict[str, Any], index: IndexDef, values: dict[str, int | None]
) -> list[tuple[str, int | None, int | None]]:
    """Write one computed index into the models of `doc`, in memory only, and
    return the (model, old, new) triples that moved.

    Nothing reaches disk here, so a caller can decide whether to save (main()
    saves only with -w) and a rewritten score plus its refreshed indexes land
    in one write instead of two.
    """
    url = source_url(doc, index)
    changes: list[tuple[str, int | None, int | None]] = []
    for model in doc.get("models") or []:
        name = model.get("name")
        if not isinstance(name, str) or name not in values:
            continue

        scores = model.get("scores")
        if not isinstance(scores, dict):
            scores = {}
        old = scores.get(index.key)
        new = values[name]
        model["scores"] = put_first(scores, index.key, new)

        updated = model.get("scores_updated")
        if not isinstance(updated, dict):
            updated = {}
        model["scores_updated"] = put_first(
            updated, index.key, updated.get(index.key)
        )

        # The index is computed, not read off a leaderboard, so it cites the
        # page that documents how it is computed rather than nothing at all --
        # a reader who clicks the value gets the method. An unranked model
        # reports no source, the same way it reports no date.
        sources = model.get("scores_source")
        if not isinstance(sources, dict):
            sources = {}
        model["scores_source"] = put_first(
            sources, index.key, url if new is not None else None
        )

        if old != new:
            changes.append((name, old, new))
            if new is None:
                # No value, no date: an unranked model reports neither.
                model["scores_updated"][index.key] = None
            else:
                stamp_score_updated(model, index.key)
    return changes


def refresh(doc: dict[str, Any]) -> list[tuple[str, str, int | None, int | None]]:
    """Recompute the derived columns in `doc` in memory; returns the changes as
    (index key, model, old, new) tuples.

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
    changes: list[tuple[str, str, int | None, int | None]] = []
    # Applied back to front because put_first prepends: the last index applied
    # ends up leading each model's maps, so the keys sit in INDEXES order.
    for index in reversed(INDEXES):
        changes.extend(
            (index.key, name, old, new)
            for name, old, new in apply_index(
                doc, index, compute_index(models, doc, index)
            )
        )
    return changes


def refresh_and_report(
    doc: dict[str, Any],
) -> list[tuple[str, str, int | None, int | None]]:
    """refresh() for the tools that write llm.json for their own reasons, with
    the reporting they all want: a line per index naming how many models moved,
    and a warning instead of an exception when llm.json is shaped wrong.

    Every writer of "scores" or "models" has to call this before saving --
    update.py after a fetch, edit.py after a hand edit, prune.py after dropping
    a model -- because an index is a function of the whole table: a score that
    changes, or a model that leaves, re-ranks everyone else. Not reporting a
    problem loudly here is deliberate: the caller's own write is what the user
    asked for, and ./derive_indexes.py can repair the columns afterwards.
    """
    try:
        changes = refresh(doc)
    except ValueError as exc:
        print(
            f"Warning: could not recompute the derived indexes ({exc}); "
            "run ./derive_indexes.py once llm.json is fixed",
            file=sys.stderr,
        )
        return []
    for index in INDEXES:
        moved = sum(1 for key, *_ in changes if key == index.key)
        if moved:
            print(f"Recomputed {index.key} for {moved} model(s)")
    return changes


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
        help="How many top-ranked models to print per index (default: 15, 0 for none).",
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

    by_name = {
        model["name"]: model for model in models if isinstance(model.get("name"), str)
    }

    all_changes: list[tuple[str, str, int | None, int | None]] = []
    restamped = 0
    for index in reversed(INDEXES):
        values = compute_index(models, doc, index)

        ranked = sum(1 for value in values.values() if value is not None)
        print(
            f"{index.key}: {len(values)} model(s): {ranked} ranked, "
            f"{len(values) - ranked} unranked "
            f"(< {MIN_SCORED_FRACTION:.0%} of the weight of "
            f"{len(index.contributing)} benchmarks)"
        )

        # Counted before the write, because apply_index leaves nothing to
        # compare against afterwards. A run whose values all hold still can
        # carry a source restamp -- the column's URL changed in llm.json, or
        # the value predates the source being recorded at all -- and that is
        # worth writing.
        url = source_url(doc, index)
        restamped += sum(
            1
            for name, value in values.items()
            if (by_name[name].get("scores_source") or {}).get(index.key)
            != (url if value is not None else None)
        )

        # In memory only; the file is written further down, and only with -w.
        changes = apply_index(doc, index, values)
        all_changes.extend((index.key, name, old, new) for name, old, new in changes)

        if args.top:
            best = sorted(
                ((value, name) for name, value in values.items() if value is not None),
                reverse=True,
            )[: args.top]
            if best:
                print(f"\nTop {len(best)} by {index.key}:")
                for rank, (value, name) in enumerate(best, start=1):
                    measured = scored_count(by_name[name], index)
                    print(
                        f"  {rank:2d}. {fmt(value):>9s}  {name:40s} "
                        f"{measured}/{len(index.contributing)} measured"
                    )
        print()

    if not all_changes and not restamped:
        print("The derived indexes are up to date. Nothing to do.")
        return 0

    if all_changes:
        print(f"{len(all_changes)} value(s) change:")
        for key, name, old, new in sorted(all_changes, key=lambda c: (c[0], c[1])):
            print(f"  {key:14s} {name:40s} {fmt(old):>9s} -> {fmt(new):>9s}")
    if restamped:
        print(f"\n{restamped} source URL(s) restamped")

    if not args.write:
        print("\ndry-run only, pass --write to persist changes")
        return 0

    path.write_text(json.dumps(doc, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
