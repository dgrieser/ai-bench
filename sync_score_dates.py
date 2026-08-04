#!/usr/bin/env python3
"""Keep scores_updated and scores_source in sync with scores in llm.json.

A benchmark whose score is null must have a null date and a null source URL; a
stale date left behind after a score is reverted makes the model look freshly
evaluated, and a stale URL attributes a score that no longer exists. Also
reports the reverse cases (score set but date or URL missing), which need real
values this script cannot invent -- a missing URL is what
`update.py --fill-source-urls` backfills.

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
    cleared_urls, missing_urls = [], []
    for model in doc.get("models", []):
        name = model.get("name", "?")
        scores = model.get("scores") or {}
        updated = model.get("scores_updated")
        sources = model.get("scores_source")
        if isinstance(updated, dict):
            for key, value in scores.items():
                date = updated.get(key)
                if value is None and date is not None:
                    cleared.append((name, key, date))
                    updated[key] = None
                elif value is not None and date is None:
                    missing.append((name, key, value))
        if isinstance(sources, dict):
            for key, value in scores.items():
                url = sources.get(key)
                if value is None and url is not None:
                    cleared_urls.append((name, key, url))
                    sources[key] = None
                elif value is not None and url is None:
                    missing_urls.append((name, key, value))

    for name, key, date in cleared:
        print(f"clear {name:36s} {key:20s} date {date} -> null (score is null)")
    for name, key, url in cleared_urls:
        print(f"clear {name:36s} {key:20s} source {url} -> null (score is null)")
    for name, key, value in missing:
        print(f"WARN  {name:36s} {key:20s} score {value} has no date", file=sys.stderr)
    for name, key, value in missing_urls:
        print(f"WARN  {name:36s} {key:20s} score {value} has no source URL", file=sys.stderr)

    print(
        f"\n{len(cleared)} stale date(s) cleared, {len(cleared_urls)} stale URL(s) cleared, "
        f"{len(missing)} score(s) missing a date, {len(missing_urls)} score(s) missing a source URL"
    )

    if not cleared and not cleared_urls:
        return 0
    if not args.write:
        print("\ndry-run only, pass --write to persist changes")
        return 0

    path.write_text(json.dumps(doc, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
