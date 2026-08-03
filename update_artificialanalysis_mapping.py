#!/usr/bin/env python3
"""Review llm.json model names that may map to Artificial Analysis slugs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import _prompts
from _matching import MATCH_LABELS, normalize_slug, query_variants, token_match_score
from _artificialanalysis_mapping import (
    AA_MODEL_MAPPING,
    add_aa_mapping,
    add_ignored_aa_suggestions,
    fetch_aa_model_names,
    load_ignored_aa_suggestions,
    load_llm_to_aa_mapping,
)

DEFAULT_LLM_JSON = Path(__file__).resolve().with_name("llm.json")


@dataclass(frozen=True)
class Candidate:
    slug: str
    reason: str


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\nError: {message}\n")


def parse_args() -> argparse.Namespace:
    parser = HelpOnErrorArgumentParser(
        description="Review missing llm.json Artificial Analysis model mappings."
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
        help=f"Interactively persist mappings to {AA_MODEL_MAPPING.name}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Maximum suggestions per model (default: 8).",
    )
    _prompts.add_cli_flag(parser)
    args = parser.parse_args()
    _prompts.apply_cli_flag(args)
    return args


def load_doc(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("Top-level JSON value must be an object.")
    if not isinstance(doc.get("models"), list):
        raise ValueError("JSON must contain a models array.")
    return doc


def missing_aa_models(doc: dict[str, Any]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for model in doc.get("models", []):
        if not isinstance(model, dict):
            continue
        name = model.get("name")
        if not isinstance(name, str) or not name:
            continue
        scores = model.get("scores")
        aa_score = scores.get("aa_intelligence_index") if isinstance(scores, dict) else None
        if aa_score is None:
            models.append(model)
    return models


def hf_repo_name(model: dict[str, Any]) -> str | None:
    url = model.get("url")
    if not isinstance(url, str) or not url:
        return None
    parsed = urlparse(url)
    if parsed.netloc.lower() != "huggingface.co":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None
    return parts[1]


def model_queries(model: dict[str, Any]) -> list[tuple[str, str]]:
    name = model.get("name")
    queries: list[tuple[str, str]] = []
    if isinstance(name, str) and name:
        queries.append((name, "llm.json"))

    repo_name = hf_repo_name(model)
    if repo_name and normalize_slug(repo_name) not in {normalize_slug(q) for q, _ in queries}:
        queries.append((repo_name, "huggingface"))
    return queries


def find_candidate_slugs(
    model: dict[str, Any],
    aa_slugs: list[str],
    ignored: set[str],
    limit: int,
) -> list[Candidate]:
    best: dict[str, tuple[tuple[int, int, int, int, int, str], str]] = {}
    for query_index, (query, source) in enumerate(model_queries(model)):
        for variant_index, (variant, variant_reason) in enumerate(query_variants(query)):
            for aa_slug in aa_slugs:
                if aa_slug in ignored:
                    continue
                match = token_match_score(variant, aa_slug)
                if match is None:
                    continue
                match_kind, distance = match
                score = (
                    query_index,
                    variant_index,
                    match_kind,
                    distance,
                    len(aa_slug),
                    aa_slug,
                )
                match_label = MATCH_LABELS.get(match_kind, "match")
                reason = f"{source}: {variant_reason} to {variant}, {match_label}"
                if aa_slug not in best or score < best[aa_slug][0]:
                    best[aa_slug] = (score, reason)

    ordered = sorted(best.items(), key=lambda item: item[1][0])
    return [Candidate(slug=slug, reason=reason) for slug, (_score, reason) in ordered[:limit]]


def exact_aa_slug(model: dict[str, Any], aa_by_norm: dict[str, str]) -> str | None:
    name = model.get("name")
    if not isinstance(name, str) or not name:
        return None
    return aa_by_norm.get(normalize_slug(name))


def prompt_candidate(model_name: str, candidates: list[Candidate], aa_slugs: list[str]) -> str | None:
    # Collect mode: queue the question and answer "none". The caller's
    # add_ignored_aa_suggestions() is skipped, so it is asked again next time.
    if _prompts.collecting():
        _prompts.record(
            kind="aa-mapping",
            subject=model_name,
            question=f"Which Artificial Analysis slug is llm.json model '{model_name}'?",
            candidates=[candidate.slug for candidate in candidates],
        )
        return None

    aa_by_lower = {slug.lower(): slug for slug in aa_slugs}
    while True:
        try:
            raw = input("Select number, type AA slug, or Enter for none: ").strip()
        except EOFError:
            return None
        if raw == "":
            return None
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(candidates):
                return candidates[index].slug
            print("Invalid selection.")
            continue
        canonical = aa_by_lower.get(raw.lower())
        if canonical is not None:
            return canonical
        print(f"Selection for {model_name} must match an Artificial Analysis slug.")


def print_candidates(candidates: list[Candidate]) -> None:
    if not candidates:
        print("  no candidate matches")
        return
    for idx, candidate in enumerate(candidates, start=1):
        print(f"  {idx}. {candidate.slug} ({candidate.reason})")


def main() -> int:
    args = parse_args()
    llm_path = Path(args.json_file)
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")

    doc = load_doc(llm_path)
    missing_models = missing_aa_models(doc)

    try:
        aa_slugs = fetch_aa_model_names()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    aa_by_norm = {normalize_slug(slug): slug for slug in aa_slugs}
    mapping = load_llm_to_aa_mapping()
    ignored_by_model = load_ignored_aa_suggestions()
    interactive = sys.stdin.isatty()

    print(f"models in {llm_path}: {len(doc.get('models', []))}")
    print(f"models missing aa_intelligence_index: {len(missing_models)}")
    print(f"models on artificialanalysis.py: {len(aa_slugs)}")
    print(f"existing llm -> AA mappings: {len(mapping)}")
    print()

    exact = 0
    already_mapped = 0
    suggested = 0
    without_candidates = 0
    written = 0
    ignored_written = 0

    for model in missing_models:
        model_name = model.get("name")
        if not isinstance(model_name, str):
            continue

        mapped_slug = mapping.get(model_name)
        if mapped_slug:
            already_mapped += 1
            print(f"{model_name}")
            print(f"  mapped already: {mapped_slug}")
            continue

        direct_slug = exact_aa_slug(model, aa_by_norm)
        if direct_slug:
            exact += 1
            print(f"{model_name}")
            print(f"  exact AA slug: {direct_slug} (update.py will fetch it directly)")
            continue

        ignored = ignored_by_model.get(model_name, set())
        candidates = find_candidate_slugs(model, aa_slugs, ignored, args.limit)
        print(f"{model_name}")
        print_candidates(candidates)

        if candidates:
            suggested += 1
        else:
            without_candidates += 1

        if not (args.write and (interactive or _prompts.collecting()) and candidates):
            continue

        selected = prompt_candidate(model_name, candidates, aa_slugs)
        if selected:
            add_aa_mapping(model_name, selected)
            written += 1
            print(f"  -> wrote mapping to {selected}")
            continue

        if _prompts.collecting():
            print("  -> queued for manual review, nothing recorded")
            continue

        ignored_slugs = [candidate.slug for candidate in candidates]
        if ignored_slugs:
            add_ignored_aa_suggestions(model_name, ignored_slugs)
            ignored_written += 1
            print("  -> recorded suggestions as ignored for this model")

    print()
    print(f"already mapped: {already_mapped}")
    print(f"exact AA slug matches: {exact}")
    print(f"models with candidate matches: {suggested}")
    print(f"models without candidate matches: {without_candidates}")
    print(f"new mappings written: {written}")
    print(f"ignored suggestion sets written: {ignored_written}")
    if not args.write:
        if _prompts.collecting():
            print("collect mode needs -w/--write to reach the prompts; nothing queued")
        else:
            print("dry-run only, pass --write to persist choices")
    elif not interactive:
        print("--write needs an interactive terminal to prompt")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
