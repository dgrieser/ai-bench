#!/usr/bin/env python3
"""Review and update OSWorld-Verified to Artificial Analysis name mappings.

llm.json tracks open-weight models only, so names the source reports as
closed-weight are recorded as __closed_weights__ without prompting. Pass
--recheck-closed to put those back in the review (sources do get it wrong).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import _prompts
from _openness import is_closed_weights, open_index
from add import prompt_select_or_new
from _osworld_mapping import (
    OSWORLD_MAPPING,
    add_osworld_closed_weights,
    add_osworld_mapping,
    add_osworld_unmappable,
    fetch_osworld_model_names,
    load_reviewed_osworld_names,
)

DEFAULT_LLM_JSON = Path(__file__).resolve().with_name("llm.json")


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\nError: {message}\n")


def parse_args() -> argparse.Namespace:
    parser = HelpOnErrorArgumentParser(
        description="Check for current OSWorld names that may map to existing llm.json models."
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(DEFAULT_LLM_JSON),
        help='Path to JSON file to read (default: "./llm.json" next to this script)',
    )
    parser.add_argument(
        "--write",
        "-w",
        action="store_true",
        help=f"Write selected mappings back to {OSWORLD_MAPPING.name}.",
    )
    parser.add_argument(
        "--recheck-closed",
        action="store_true",
        help="Prompt for names previously skipped as closed-weight models, "
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


def load_doc(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        doc = json.load(handle)

    if not isinstance(doc, dict):
        raise ValueError("Top-level JSON value must be an object.")
    if not isinstance(doc.get("models"), list):
        raise ValueError("JSON must contain a models array.")
    return doc


def unique_names(models: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for model in models:
        name = model.get("name")
        if isinstance(name, str) and name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def fuzzy_match(query: str, option: str) -> tuple[int, int] | None:
    haystack = option.lower()
    needle = query.lower()

    if needle in haystack:
        return (0, haystack.index(needle))

    pos = 0
    gap_score = 0
    for char in needle:
        idx = haystack.find(char, pos)
        if idx == -1:
            return None
        gap_score += idx - pos
        pos = idx + 1
    return (1, gap_score)


def find_matches(query: str, options: list[str], limit: int = 10) -> list[str]:
    if not query:
        return options[:limit]

    scored: list[tuple[tuple[int, int, int], str]] = []
    for option in options:
        match = fuzzy_match(query, option)
        if match is None:
            continue
        scored.append(((match[0], match[1], len(option)), option))
    scored.sort(key=lambda item: item[0])
    return [option for _, option in scored[:limit]]


def prompt_slug_for_osworld_name(osworld_name: str, slugs: list[str]) -> str | None:
    options_lower = {slug.lower(): slug for slug in slugs}

    while True:
        label = f"Map OSWorld model '{osworld_name}' to llm.json model"
        raw = prompt_select_or_new(label, slugs)
        if raw is None:
            return None

        canonical = options_lower.get(raw.lower())
        if canonical is not None:
            return canonical

        print("Selection must match an existing llm.json model name. Press Enter to skip.")


def main() -> int:
    args = parse_args()
    llm_path = Path(args.json_file)
    doc = load_doc(llm_path)
    llm_names = unique_names(doc["models"])

    osworld_names = fetch_osworld_model_names()
    reviewed_names = load_reviewed_osworld_names(
        include_closed=not args.recheck_closed
    )
    open_index(refresh=args.refresh_openness)
    unreviewed_osworld_names = [name for name in osworld_names if name not in reviewed_names]

    print(f"models in {llm_path}: {len(llm_names)}")
    print(f"models on osworld: {len(osworld_names)}")
    print(f"already reviewed osworld names: {len(reviewed_names)}")
    print(f"unreviewed osworld names: {len(unreviewed_osworld_names)}")
    print()

    matched = 0
    without_candidates = 0
    written = 0
    skipped = 0
    closed = 0
    interactive = sys.stdin.isatty()

    for osworld_name in unreviewed_osworld_names:
        candidates = find_matches(osworld_name, llm_names, limit=5)
        print(f"{osworld_name}")
        if candidates:
            matched += 1
            for idx, candidate in enumerate(candidates, start=1):
                print(f"  {idx}. {candidate}")
        else:
            without_candidates += 1
            print("  no candidate matches")

        if not args.recheck_closed and is_closed_weights(
            osworld_name, guard_names=llm_names
        ):
            closed += 1
            print("  -> closed weights per source, skipped without prompting")
            if args.write:
                add_osworld_closed_weights(osworld_name)
            continue

        if not ((interactive or _prompts.collecting()) and args.write):
            continue

        slug = prompt_slug_for_osworld_name(osworld_name, llm_names)
        if not slug:
            skipped += 1
            if _prompts.collecting():
                print("  -> queued for manual review, nothing recorded")
            else:
                add_osworld_unmappable(osworld_name)
                print("  -> recorded as unmappable (won't ask again)")
            continue

        add_osworld_mapping(osworld_name, slug)
        written += 1
        print(f"  -> wrote mapping to {slug}")

    print()
    print(f"osworld names with candidate matches: {matched}")
    print(f"osworld names without candidate matches: {without_candidates}")
    print(f"new mappings written: {written}")
    label = "queued for review" if _prompts.collecting() else "recorded as unmappable"
    print(f"{label}: {skipped}")
    print(f"skipped as closed weights: {closed}")
    if not args.write:
        print("dry-run only, pass --write to persist changes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
