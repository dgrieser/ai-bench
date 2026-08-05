#!/usr/bin/env python3
"""Stamp hand-researched scores_source URLs from score-source-proposals.json.

update.py can only attribute a score to a page whose freshly fetched value still
equals the stored one (see its --fill-source-urls mode). Scores typed in by hand,
and scores whose leaderboard row has since changed or disappeared, are therefore
left with a null source forever. This script closes that gap from a reviewed
proposals file: each entry names the page the value was actually found on, with
the quote that was read off it.

A proposal is applied only if the stored score still equals the value the
proposal was researched against and no source is stored yet, so a proposals file
that has gone stale against llm.json reports skips instead of stamping a page
that never published the current number.

Default is a dry-run; pass -w/--write to persist changes (same convention as
update.py and prune.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _scores import editable_benchmarks, stamp_score_source

HERE = Path(__file__).resolve().parent
DEFAULT_LLM_JSON = HERE / "llm.json"
DEFAULT_PROPOSALS = HERE / "score-source-proposals.json"
JSON_DUMP_KWARGS = {"indent": 2, "ensure_ascii": False}

# Weakest first; --min-confidence keeps this level and everything above it.
CONFIDENCE_ORDER = ["weak", "probable", "confirmed-by-provenance", "confirmed"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(DEFAULT_LLM_JSON),
        help='Path to JSON file to read/update (default: "./llm.json" next to this script)',
    )
    parser.add_argument(
        "--proposals",
        "-p",
        default=str(DEFAULT_PROPOSALS),
        help='Proposals file (default: "./score-source-proposals.json")',
    )
    parser.add_argument(
        "--min-confidence",
        "-c",
        choices=CONFIDENCE_ORDER,
        default="probable",
        help="Skip proposals below this confidence level (default: probable).",
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
    proposals_doc = json.loads(Path(args.proposals).read_text(encoding="utf-8"))

    benchmarks = editable_benchmarks(doc)
    by_name = {m.get("name"): m for m in doc.get("models", [])}
    floor = CONFIDENCE_ORDER.index(args.min_confidence)

    apply_now: list[tuple[dict, dict]] = []
    below, skipped = [], []
    for proposal in proposals_doc.get("proposals", []):
        name = proposal["model"]
        key = proposal["benchmark"]
        label = f"{name}.{key}"

        confidence = proposal.get("confidence", "weak")
        if confidence not in CONFIDENCE_ORDER:
            skipped.append((label, f"unknown confidence {confidence!r}"))
            continue
        if CONFIDENCE_ORDER.index(confidence) < floor:
            below.append((label, confidence))
            continue

        model = by_name.get(name)
        if model is None:
            skipped.append((label, "model not in llm.json"))
            continue
        if key not in benchmarks:
            skipped.append((label, "not an editable benchmark key"))
            continue

        stored = (model.get("scores") or {}).get(key)
        if stored != proposal["value"]:
            skipped.append((label, f"score moved: {proposal['value']!r} -> {stored!r}"))
            continue

        existing = (model.get("scores_source") or {}).get(key)
        if isinstance(existing, str) and existing.strip():
            skipped.append((label, f"already sourced: {existing}"))
            continue

        apply_now.append((model, proposal))

    for label, confidence in below:
        print(f"  below --min-confidence ({confidence}): {label}")
    for label, reason in skipped:
        print(f"  skip {label}: {reason}")
    if below or skipped:
        print()

    if not apply_now:
        print("Nothing to apply.")
        return 0

    print(f"{len(apply_now)} source URL(s) to stamp:")
    for _, proposal in apply_now:
        print(
            f"  {proposal['model']}.{proposal['benchmark']} = {proposal['value']!r}"
            f"  [{proposal['confidence']}]\n      {proposal['url']}"
        )

    unresolved = proposals_doc.get("unresolved", [])
    if unresolved:
        print(f"\n{len(unresolved)} score(s) left unsourced on purpose:")
        for entry in unresolved:
            print(f"  {entry['model']}.{entry['benchmark']} = {entry['value']!r}: {entry['reason']}")

    if not args.write:
        print("\ndry-run only, pass --write to persist changes")
        return 0

    for model, proposal in apply_now:
        stamp_score_source(model, proposal["benchmark"], proposal["url"])

    path.write_text(json.dumps(doc, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
