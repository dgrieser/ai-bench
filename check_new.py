#!/usr/bin/env python3
"""Report open-weights models on Artificial Analysis not yet in llm.json.

"New" is decided by AA release_date: by default, models released within the
last 30 days (--days). Slugs already present in llm.json, and AA slugs listed
in the ignored-mapping file, are filtered out. Read-only: prints candidates,
does not modify llm.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_LLM_JSON = HERE / "llm.json"
AA_SCRIPT = HERE / "artificialanalysis.py"
IGNORED_MAPPING = HERE / "model-name-mapping-llm-to-artificialanalysis-ignored.json"


def existing_slugs(llm_path: Path) -> set[str]:
    doc = json.loads(llm_path.read_text(encoding="utf-8"))
    return {m.get("name") for m in doc.get("models", []) if m.get("name")}


def ignored_slugs() -> set[str]:
    if not IGNORED_MAPPING.exists():
        return set()
    mapping = json.loads(IGNORED_MAPPING.read_text(encoding="utf-8"))
    slugs: set[str] = set()
    for variants in mapping.values():
        slugs.update(variants)
    return slugs


def fetch_aa_models(cutoff: date, include_closed: bool) -> list[dict]:
    cmd = [
        sys.executable,
        str(AA_SCRIPT),
        "--release-date",
        cutoff.isoformat(),
        "--output",
        "json",
        "--no-mmmu-pro",
    ]
    if not include_closed:
        cmd.append("--open")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"artificialanalysis.py failed (exit {proc.returncode})")
    return json.loads(proc.stdout).get("data", [])


def creator_name(m: dict) -> str:
    c = m.get("model_creator")
    if isinstance(c, dict):
        return c.get("name") or c.get("slug") or "?"
    return "?"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(DEFAULT_LLM_JSON),
        help='Path to llm.json to compare against (default: "./llm.json").',
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Look back this many days from today for release_date (default: 30).",
    )
    parser.add_argument(
        "--date",
        help="Explicit cutoff YYYY-MM-DD (release_date on/after); overrides --days.",
    )
    parser.add_argument(
        "--include-closed",
        action="store_true",
        help="Also include closed-weights models (default: open weights only).",
    )
    args = parser.parse_args()

    if args.date:
        try:
            cutoff = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit("invalid --date, expected YYYY-MM-DD")
    else:
        cutoff = date.today() - timedelta(days=args.days)

    known = existing_slugs(Path(args.json_file)) | ignored_slugs()
    models = fetch_aa_models(cutoff, args.include_closed)

    new = [m for m in models if m.get("slug") and m["slug"] not in known]
    new.sort(key=lambda m: m.get("release_date") or "", reverse=True)

    kind = "models" if args.include_closed else "open-weights models"
    if not new:
        print(f"No new {kind} on AA since {cutoff.isoformat()}.")
        return 0

    print(f"{len(new)} new {kind} on AA since {cutoff.isoformat()} (not in llm.json):\n")
    print(f"  {'RELEASED':10s}  {'SLUG':32s}  {'CREATOR':22s}  URL")
    for m in new:
        print(
            f"  {m.get('release_date') or '?':10s}  {m['slug']:32s}  "
            f"{creator_name(m):22s}  {m.get('url') or ''}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
