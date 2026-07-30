#!/usr/bin/env python3
"""Review the llm-stats.com mapping files against llm.json.

Convenience companion to add.py's prompts. Dry-run by default (preview the
proposed matches); pass -w/--write to run the interactive review and persist,
the same way update.py uses -w.

Under -w it walks the llm-stats model ids and benchmark labels, pre-filling an
auto-match (exact / date-suffix stripped / normalized) as the default so you can
accept with Enter, type another value, or skip. Ids and labels left blank are
recorded as __unmappable__ so add.py stops prompting for them. Each answer is
written immediately. On a non-interactive terminal, -w auto-applies every
confident match (and marks leftover ids and labels unmappable); pass
--collect-prompts instead to queue every id and label for review and record
nothing.

Model ids with no confident match whose licence is proprietary are recorded as
__closed_weights__ without prompting: llm.json tracks open-weight models only,
so a closed model can never map to one. Pass --recheck-closed to review those
again (llm-stats does get the odd licence wrong).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import _prompts
from _openness import is_closed_weights, open_index
from add import find_matches, prompt_key_for_label, prompt_select_or_new
from _llmstats_mapping import (
    add_llmstats_benchmark_mapping,
    add_llmstats_benchmark_unmappable,
    add_llmstats_closed_weights,
    add_llmstats_mapping,
    add_llmstats_unmappable,
    fetch_llmstats_benchmark_names,
    fetch_llmstats_model_names,
    fetch_llmstats_model_openness,
    load_reviewed_llmstats_benchmarks,
    load_reviewed_llmstats_names,
)

DEFAULT_LLM_JSON = Path(__file__).resolve().with_name("llm.json")

# llm-stats benchmark labels whose llm.json key is not a normalized match.
BENCHMARK_ALIASES = {
    "gpqa": "gpqa_diamond",
    "osworld": "osworld_verified",
}

_DATE_SUFFIX_RE = re.compile(r"-\d{6,8}$")


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def strip_date_suffix(model_id: str) -> str:
    return _DATE_SUFFIX_RE.sub("", model_id)


def load_doc(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("Top-level JSON value must be an object.")
    return doc


def model_slugs(doc: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for model in doc.get("models", []):
        if isinstance(model, dict):
            name = model.get("name")
            if isinstance(name, str) and name:
                out.append(name)
    return out


def benchmark_keys(doc: dict[str, Any]) -> list[str]:
    benchmarks = doc.get("benchmarks", {})
    return sorted(benchmarks.keys()) if isinstance(benchmarks, dict) else []


def auto_match_slug(model_id: str, slugs: list[str]) -> str | None:
    """Single confident llm.json slug for an llm-stats model_id, else None."""
    nm = norm(model_id)
    exact = [s for s in slugs if norm(s) == nm]
    if not exact:
        nm = norm(strip_date_suffix(model_id))
        exact = [s for s in slugs if norm(strip_date_suffix(s)) == nm]
    exact = list(dict.fromkeys(exact))
    return exact[0] if len(exact) == 1 else None


def auto_match_benchmark(label: str, keys: list[str]) -> str | None:
    by_norm = {norm(k): k for k in keys}
    target = BENCHMARK_ALIASES.get(label) or by_norm.get(norm(label))
    return target if target in keys else None


def review_models(
    doc: dict[str, Any],
    model_ids: list[str],
    interactive: bool,
    openness: dict[str, bool | None],
    recheck_closed: bool = False,
) -> int:
    slugs = model_slugs(doc)
    reviewed_ids = load_reviewed_llmstats_names(include_closed=not recheck_closed)
    changed = 0

    for model_id in model_ids:
        if model_id in reviewed_ids:
            continue
        default = auto_match_slug(model_id, slugs)

        if default is None and not recheck_closed and is_closed_weights(
            model_id, open_weights=openness.get(model_id), guard_names=slugs
        ):
            add_llmstats_closed_weights(model_id)
            changed += 1
            print(f"Skipped llm-stats model '{model_id}': closed weights per source")
            continue

        if _prompts.collecting():
            # Never auto-apply unattended: queue the id and record nothing, so
            # the same question comes back on the next local run.
            _prompts.record(
                kind="llmstats-model",
                subject=model_id,
                question=f"Which llm.json model is llm-stats id '{model_id}'?",
                candidates=[default] if default else find_matches(model_id, slugs, limit=5),
                default=default,
            )
            continue

        if interactive:
            slug = prompt_select_or_new(
                f"llm.json slug for '{model_id}'", slugs, default=default
            )
        else:
            slug = default

        if not slug:
            add_llmstats_unmappable(model_id)
            changed += 1
            print(f"Recorded llm-stats model '{model_id}' as unmappable")
            continue
        add_llmstats_mapping(model_id, slug)
        changed += 1
        print(f"Mapped llm-stats '{model_id}' -> '{slug}'")
    return changed


def review_benchmarks(doc: dict[str, Any], labels: list[str], interactive: bool) -> int:
    keys = benchmark_keys(doc)
    reviewed = load_reviewed_llmstats_benchmarks()
    changed = 0

    for label in labels:
        if label in reviewed:
            continue
        default = auto_match_benchmark(label, keys)

        if _prompts.collecting():
            _prompts.record(
                kind="llmstats-benchmark",
                subject=label,
                question=f"Which llm.json benchmark is llm-stats label '{label}'?",
                candidates=[default] if default else find_matches(label, keys, limit=5),
                default=default,
            )
            continue

        if interactive:
            key = prompt_key_for_label("llm-stats benchmark", label, keys, default=default)
        else:
            key = default

        if not key:
            add_llmstats_benchmark_unmappable(label)
            changed += 1
            print(f"Recorded llm-stats benchmark '{label}' as unmappable")
            continue
        add_llmstats_benchmark_mapping(label, key)
        changed += 1
        print(f"Mapped llm-stats benchmark '{label}' -> '{key}'")
    return changed


def preview_models(
    doc: dict[str, Any],
    model_ids: list[str],
    openness: dict[str, bool | None],
    recheck_closed: bool = False,
) -> None:
    slugs = model_slugs(doc)
    reviewed_ids = load_reviewed_llmstats_names(include_closed=not recheck_closed)
    matched: list[tuple[str, str]] = []
    closed: list[str] = []
    unmappable: list[str] = []
    for model_id in model_ids:
        if model_id in reviewed_ids:
            continue
        slug = auto_match_slug(model_id, slugs)
        if slug:
            matched.append((model_id, slug))
        elif not recheck_closed and is_closed_weights(
            model_id, open_weights=openness.get(model_id), guard_names=slugs
        ):
            closed.append(model_id)
        else:
            unmappable.append(model_id)
    print(f"model mappings ({len(matched)} proposed):")
    for model_id, slug in sorted(matched):
        arrow = "==" if model_id == slug else "->"
        print(f"  {model_id} {arrow} {slug}")
    if not matched:
        print("  (none)")
    print(f"llm-stats model ids to mark __closed_weights__: {len(closed)}")
    for model_id in closed:
        print(f"  - {model_id}")
    print(f"llm-stats model ids to mark __unmappable__: {len(unmappable)}")
    for model_id in unmappable:
        print(f"  - {model_id}")
    print()


def preview_benchmarks(doc: dict[str, Any], labels: list[str]) -> None:
    keys = benchmark_keys(doc)
    reviewed = load_reviewed_llmstats_benchmarks()
    matched: list[tuple[str, str]] = []
    unmappable: list[str] = []
    for label in labels:
        if label in reviewed:
            continue
        key = auto_match_benchmark(label, keys)
        if key:
            matched.append((label, key))
        else:
            unmappable.append(label)
    print(f"benchmark mappings ({len(matched)} proposed):")
    for label, key in sorted(matched):
        print(f"  {label} -> {key}")
    if not matched:
        print("  (none)")
    print(f"benchmark labels to mark __unmappable__: {len(unmappable)}")
    for label in unmappable:
        print(f"  - {label}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review the llm-stats.com mapping files against llm.json."
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(DEFAULT_LLM_JSON),
        help='Path to llm.json (default: "./llm.json" next to this script).',
    )
    parser.add_argument(
        "--write",
        "-w",
        action="store_true",
        help="Run the interactive review and persist answers (default is dry-run).",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip the model-name mapping review.",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip the benchmark-name mapping review.",
    )
    parser.add_argument(
        "--recheck-closed",
        action="store_true",
        help="Prompt for model ids previously skipped as closed-weight models, "
        "instead of skipping them again.",
    )
    parser.add_argument(
        "--refresh-openness",
        action="store_true",
        help="Rebuild the cached open-weight index before reviewing.",
    )
    _prompts.add_cli_flag(parser)
    args = parser.parse_args()
    _prompts.apply_cli_flag(args)
    return args


def main() -> int:
    args = parse_args()
    doc = load_doc(Path(args.json_file))

    model_ids: list[str] = []
    openness: dict[str, bool | None] = {}
    if not args.skip_models:
        try:
            model_ids = fetch_llmstats_model_names()
            openness = fetch_llmstats_model_openness()
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        open_index(refresh=args.refresh_openness)

    labels: list[str] = []
    if not args.skip_benchmarks:
        try:
            labels = fetch_llmstats_benchmark_names()
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if not args.write:
        if not args.skip_models:
            preview_models(doc, model_ids, openness, args.recheck_closed)
        if not args.skip_benchmarks:
            preview_benchmarks(doc, labels)
        print("dry-run only, pass -w/--write to apply")
        return 0

    interactive = sys.stdin.isatty()
    if _prompts.collecting():
        print("collect mode: queueing every id and label, recording nothing")
    elif not interactive:
        print("non-interactive: auto-applying confident matches")

    added = 0
    if not args.skip_models:
        added += review_models(
            doc, model_ids, interactive, openness, args.recheck_closed
        )
    if not args.skip_benchmarks:
        added += review_benchmarks(doc, labels, interactive)

    print(f"{added} mapping entries updated")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
