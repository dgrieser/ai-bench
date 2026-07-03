#!/usr/bin/env python3
"""Find open-weights models on Artificial Analysis not yet in llm.json.

"New" is decided by AA release_date: by default, models released within the
last 30 days (--days). Slugs already in llm.json, slugs listed in the
ignored-mapping file, and slugs previously dismissed here are filtered out.

The candidates are printed, then (when run interactively) each is offered for
addition:
  [y] add        -> hands off to add.py, which adds the model to llm.json and
                    walks the per-source mapping prompts (scores are filled by
                    a subsequent update.py run).
  [n] never      -> record the slug in the dismiss file so it is not offered
                    again.
  [q] quit       -> stop asking.

With no tty (e.g. inside a batch script), it only prints the candidates and
makes no changes.
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
ADD_SCRIPT = HERE / "add.py"
IGNORED_MAPPING = HERE / "model-name-mapping-llm-to-artificialanalysis-ignored.json"
DISMISSED_FILE = HERE / "check_new-dismissed.json"


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


def load_dismissed() -> set[str]:
    if not DISMISSED_FILE.exists():
        return set()
    return set(json.loads(DISMISSED_FILE.read_text(encoding="utf-8")))


def dismiss(slug: str) -> None:
    slugs = load_dismissed()
    slugs.add(slug)
    DISMISSED_FILE.write_text(
        json.dumps(sorted(slugs), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


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


def add_model(slug: str, json_file: str) -> bool:
    """Run add.py for a slug, inheriting the terminal for its prompts."""
    proc = subprocess.run([sys.executable, str(ADD_SCRIPT), "--name", slug, json_file])
    return proc.returncode == 0


def prompt_choice(slug: str) -> str:
    """Return 'add', 'skip', or 'quit' for one candidate."""
    while True:
        try:
            ans = input(f"Add '{slug}'?  [y] add  [n] never ask again  [q] quit: ").strip().lower()
        except EOFError:
            return "quit"
        if ans in ("y", "yes"):
            return "add"
        if ans in ("n", "no"):
            return "skip"
        if ans in ("q", "quit"):
            return "quit"
        print("  answer y, n, or q")


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
    parser.add_argument(
        "--no-add",
        action="store_true",
        help="Only print candidates; never prompt to add (report only).",
    )
    args = parser.parse_args()

    if args.date:
        try:
            cutoff = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit("invalid --date, expected YYYY-MM-DD")
    else:
        cutoff = date.today() - timedelta(days=args.days)

    known = existing_slugs(Path(args.json_file)) | ignored_slugs() | load_dismissed()
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

    if args.no_add or not (sys.stdin.isatty() and sys.stdout.isatty()):
        return 0

    print()
    added = dismissed = 0
    for m in new:
        slug = m["slug"]
        choice = prompt_choice(slug)
        if choice == "quit":
            break
        if choice == "skip":
            dismiss(slug)
            dismissed += 1
            continue
        if add_model(slug, args.json_file):
            added += 1
        else:
            print(f"  add.py failed for '{slug}'; leaving it for next time.", file=sys.stderr)

    print(f"\nAdded {added}, dismissed {dismissed}.")
    if added:
        print("Run ./update.py -w (or ./update-all) to fill in scores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
