#!/usr/bin/env python3
"""Review and update SWE-Rebench to Artificial Analysis name mappings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from add import prompt_select_or_new
from _swe_rebench_mapping import (
    SWE_REBENCH_MAPPING,
    add_rebench_mapping,
    fetch_swe_rebench_model_names,
    load_rebench_to_slug_mapping,
)

DEFAULT_LLM_JSON = Path(__file__).resolve().with_name("llm.json")


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\nError: {message}\n")


def parse_args() -> argparse.Namespace:
    parser = HelpOnErrorArgumentParser(
        description="Check for current SWE-Rebench names that may map to existing llm.json models."
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
        help=f"Write selected mappings back to {SWE_REBENCH_MAPPING.name}.",
    )
    return parser.parse_args()


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


def prompt_slug_for_rebench_name(rebench_name: str, slugs: list[str]) -> str | None:
    options_lower = {slug.lower(): slug for slug in slugs}

    while True:
        label = f"Map SWE-Rebench model '{rebench_name}' to llm.json model"
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

    rebench_names = fetch_swe_rebench_model_names()
    existing_mapping = load_rebench_to_slug_mapping()
    mapped_names = set(existing_mapping)
    unmapped_rebench_names = [name for name in rebench_names if name not in mapped_names]

    print(f"models in {llm_path}: {len(llm_names)}")
    print(f"models on swe_rebench: {len(rebench_names)}")
    print(f"already mapped swe_rebench names: {len(mapped_names)}")
    print(f"unmapped swe_rebench names: {len(unmapped_rebench_names)}")
    print()

    matched = 0
    without_candidates = 0
    written = 0
    interactive = sys.stdin.isatty()

    for rebench_name in unmapped_rebench_names:
        candidates = find_matches(rebench_name, llm_names, limit=5)
        print(f"{rebench_name}")
        if candidates:
            matched += 1
            for idx, candidate in enumerate(candidates, start=1):
                print(f"  {idx}. {candidate}")
        else:
            without_candidates += 1
            print("  no candidate matches")

        if not (interactive and args.write):
            continue

        slug = prompt_slug_for_rebench_name(rebench_name, llm_names)
        if not slug:
            continue

        add_rebench_mapping(rebench_name, slug)
        written += 1
        print(f"  -> wrote mapping to {slug}")

    print()
    print(f"rebench names with candidate matches: {matched}")
    print(f"rebench names without candidate matches: {without_candidates}")
    print(f"new mappings written: {written}")
    if not args.write:
        print("dry-run only, pass --write to persist changes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
