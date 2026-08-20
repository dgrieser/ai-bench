#!/usr/bin/env python3
"""Drop models from llm.json that have too few non-null benchmark scores.

Default is a dry-run; pass -w/--write to persist changes (same convention as
update.py). A model's score count is the number of non-null values in its
"scores" object; a missing "scores" object counts as zero.
"""

import argparse
import json
import sys
from pathlib import Path

import derive_indexes

DEFAULT_LLM_JSON = Path(__file__).resolve().parent / "llm.json"
JSON_DUMP_KWARGS = {"indent": 2, "ensure_ascii": False}


def score_count(model: dict) -> int:
    scores = model.get("scores") or {}
    return sum(1 for v in scores.values() if v is not None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(DEFAULT_LLM_JSON),
        help='Path to JSON file to read/update (default: "./llm.json" next to this script)',
    )
    parser.add_argument(
        "--keep-min-scores",
        "-s",
        type=int,
        required=True,
        help="Keep only models with at least this many non-null scores; drop the rest.",
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
    models = doc.get("models", [])

    keep, drop = [], []
    for m in models:
        (keep if score_count(m) >= args.keep_min_scores else drop).append(m)

    if not drop:
        print(f"No models with < {args.keep_min_scores} scores. Nothing to do.")
        return 0

    print(f"Dropping {len(drop)} model(s) with < {args.keep_min_scores} non-null scores:")
    for m in sorted(drop, key=score_count):
        creator = (m.get("creator") or {}).get("name", "?")
        print(f"  [{score_count(m)}]  {m.get('name', '?'):40s} {creator}")
    print(f"\n{len(keep)} model(s) remain (was {len(models)}).")

    if not args.write:
        print("\ndry-run only, pass --write to persist changes")
        return 0

    doc["models"] = keep
    # Percentile ranks are computed over the models in the file, so dropping one
    # that carried a contributing score re-ranks the survivors even though none
    # their own scores moved. Refreshed after the drop, before the write.
    derive_indexes.refresh_and_report(doc)
    path.write_text(json.dumps(doc, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
