#!/usr/bin/env python3
"""Review and update Spheron path to Artificial Analysis name mappings.

Spheron has no leaderboard to enumerate, so the candidate model paths are
derived from each llm.json model's HuggingFace `url` (via hf_path_from_url).
Each derived "org/model" path is then bound to the llm.json model it came from
(the default suggestion) or recorded as __unmappable__ if it has no Spheron
page. Otherwise this mirrors update_swe_atlas_mapping.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from add import prompt_select_or_new
from _spheron_mapping import (
    SPHERON_MAPPING,
    add_spheron_mapping,
    add_spheron_unmappable,
    hf_path_from_url,
    load_reviewed_spheron_names,
)

DEFAULT_LLM_JSON = Path(__file__).resolve().with_name("llm.json")


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\nError: {message}\n")


def parse_args() -> argparse.Namespace:
    parser = HelpOnErrorArgumentParser(
        description="Check for Spheron model paths (derived from llm.json HuggingFace URLs) "
        "that may map to existing llm.json models."
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
        help=f"Write selected mappings back to {SPHERON_MAPPING.name}.",
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


def spheron_paths_by_origin(models: list[dict[str, Any]]) -> dict[str, str]:
    """Map each derived Spheron path -> the llm.json model name it came from."""
    origin: dict[str, str] = {}
    for model in models:
        name = model.get("name")
        path = hf_path_from_url(model.get("url"))
        if isinstance(name, str) and name and path and path not in origin:
            origin[path] = name
    return origin


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


def prompt_slug_for_spheron_path(
    spheron_path: str, slugs: list[str], default: str | None
) -> str | None:
    options_lower = {slug.lower(): slug for slug in slugs}

    while True:
        label = f"Map Spheron path '{spheron_path}' to llm.json model"
        raw = prompt_select_or_new(label, slugs, default=default)
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

    origin = spheron_paths_by_origin(doc["models"])
    spheron_paths = list(origin)
    reviewed_names = load_reviewed_spheron_names()
    unreviewed_paths = [path for path in spheron_paths if path not in reviewed_names]

    print(f"models in {llm_path}: {len(llm_names)}")
    print(f"spheron paths derived from HuggingFace URLs: {len(spheron_paths)}")
    print(f"already reviewed spheron paths: {len(reviewed_names)}")
    print(f"unreviewed spheron paths: {len(unreviewed_paths)}")
    print()

    matched = 0
    without_candidates = 0
    written = 0
    skipped = 0
    interactive = sys.stdin.isatty()

    for spheron_path in unreviewed_paths:
        default = origin.get(spheron_path)
        candidates = find_matches(spheron_path, llm_names, limit=5)
        # The path was derived from a specific llm.json model's URL; surface that
        # model as the top suggestion (the fuzzy match rarely finds it, since the
        # full org/model path is longer than the model name).
        if default and default not in candidates:
            candidates = [default, *candidates][:5]
        print(f"{spheron_path}")
        if candidates:
            matched += 1
            for idx, candidate in enumerate(candidates, start=1):
                print(f"  {idx}. {candidate}")
        else:
            without_candidates += 1
            print("  no candidate matches")

        if not (interactive and args.write):
            continue

        slug = prompt_slug_for_spheron_path(spheron_path, llm_names, default)
        if not slug:
            add_spheron_unmappable(spheron_path)
            skipped += 1
            print("  -> recorded as unmappable (won't ask again)")
            continue

        add_spheron_mapping(spheron_path, slug)
        written += 1
        print(f"  -> wrote mapping to {slug}")

    print()
    print(f"spheron paths with candidate matches: {matched}")
    print(f"spheron paths without candidate matches: {without_candidates}")
    print(f"new mappings written: {written}")
    print(f"recorded as unmappable: {skipped}")
    if not args.write:
        print("dry-run only, pass --write to persist changes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
