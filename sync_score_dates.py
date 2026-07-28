#!/usr/bin/env python3
"""Keep scores_updated in sync with scores in llm.json.

A benchmark whose score is null must have a null date; a stale date left behind
after a score is reverted makes the model look freshly evaluated. Also reports
the reverse case (score set, date missing), which needs a real date and is not
something this script can invent.

Default is a dry-run; pass -w/--write to persist changes (same convention as
prune.py and update.py).
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_LLM_JSON = Path(__file__).resolve().parent / "llm.json"
JSON_DUMP_KWARGS = {"indent": 2, "ensure_ascii": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(DEFAULT_LLM_JSON),
        help='Path to JSON file to read/update (default: "./llm.json" next to this script)',
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

    cleared, missing = [], []
    for model in doc.get("models", []):
        scores = model.get("scores") or {}
        updated = model.get("scores_updated")
        if not isinstance(updated, dict):
            continue
        for key, value in scores.items():
            date = updated.get(key)
            if value is None and date is not None:
                cleared.append((model.get("name", "?"), key, date))
                updated[key] = None
            elif value is not None and date is None:
                missing.append((model.get("name", "?"), key, value))

    for name, key, date in cleared:
        print(f"clear {name:36s} {key:20s} date {date} -> null (score is null)")
    for name, key, value in missing:
        print(f"WARN  {name:36s} {key:20s} score {value} has no date", file=sys.stderr)

    print(f"\n{len(cleared)} stale date(s) cleared, {len(missing)} score(s) missing a date")

    if not cleared:
        return 0
    if not args.write:
        print("\ndry-run only, pass --write to persist changes")
        return 0

    path.write_text(json.dumps(doc, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
