#!/usr/bin/env python3
"""Review and update Hugging Face benchmark label -> llm.json key mappings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from add import find_matches, prompt_select_or_new
from _huggingface_mapping import (
    HF_MAPPING,
    add_hf_mapping,
    add_hf_unmappable,
    fetch_huggingface_benchmark_names,
    load_reviewed_hf_labels,
)

DEFAULT_LLM_JSON = Path(__file__).resolve().with_name("llm.json")


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\nError: {message}\n")


def parse_args() -> argparse.Namespace:
    parser = HelpOnErrorArgumentParser(
        description="Check current Hugging Face benchmark labels for mappings to llm.json benchmark keys."
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
        help=f"Write selected mappings back to {HF_MAPPING.name}.",
    )
    return parser.parse_args()


def load_doc(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict):
        raise ValueError("Top-level JSON value must be an object.")
    if not isinstance(doc.get("benchmarks"), dict):
        raise ValueError("JSON must contain a benchmarks object.")
    return doc


def prompt_key_for_label(label: str, keys: list[str]) -> str | None:
    options_lower = {k.lower(): k for k in keys}
    while True:
        prompt_label = f"Map HF label '{label}' to llm.json benchmark"
        raw = prompt_select_or_new(prompt_label, keys)
        if raw is None:
            return None
        canonical = options_lower.get(raw.lower())
        if canonical is not None:
            return canonical
        print("Selection must match an existing llm.json benchmark key. Press Enter to skip.")


def main() -> int:
    args = parse_args()
    llm_path = Path(args.json_file)
    doc = load_doc(llm_path)
    benchmark_keys = sorted(doc["benchmarks"].keys())

    hf_labels = fetch_huggingface_benchmark_names()
    reviewed_labels = load_reviewed_hf_labels()
    unmapped_labels = [name for name in hf_labels if name not in reviewed_labels]

    print(f"benchmarks in {llm_path}: {len(benchmark_keys)}")
    print(f"distinct HF labels found: {len(hf_labels)}")
    print(f"already reviewed: {len(reviewed_labels)}")
    print(f"unreviewed: {len(unmapped_labels)}")
    print()

    matched = 0
    without_candidates = 0
    written = 0
    skipped = 0
    interactive = sys.stdin.isatty()

    for label in unmapped_labels:
        candidates = find_matches(label, benchmark_keys, limit=5)
        print(f"{label}")
        if candidates:
            matched += 1
            for idx, candidate in enumerate(candidates, start=1):
                print(f"  {idx}. {candidate}")
        else:
            without_candidates += 1
            print("  no candidate matches")

        if not (interactive and args.write):
            continue

        key = prompt_key_for_label(label, benchmark_keys)
        if not key:
            add_hf_unmappable(label)
            skipped += 1
            print("  -> recorded as unmappable (won't ask again)")
            continue

        add_hf_mapping(label, key)
        written += 1
        print(f"  -> wrote mapping to {key}")

    print()
    print(f"HF labels with candidate matches: {matched}")
    print(f"HF labels without candidate matches: {without_candidates}")
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
