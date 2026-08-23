#!/usr/bin/env python3
"""Find open-weights models on Artificial Analysis not yet in llm.json.

"New" is decided by AA release_date: by default, models released within the
last 30 days (--days). Slugs already in llm.json, slugs an llm.json model
already reads on AA, slugs listed in the ignored-mapping file, and slugs
previously dismissed here are filtered out.

The candidates are printed, then (when run interactively) each is offered for
addition:
  [y] add        -> hands off to add.py, which adds the model to llm.json and
                    walks the per-source mapping prompts (scores are filled by
                    a subsequent update.py run).
  [n] never      -> record the slug in the dismiss file so it is not offered
                    again.
  [q] quit       -> stop asking.

With no tty (e.g. inside a batch script), it offers nothing and only prints the
candidates. Under --collect-prompts each candidate is queued for review instead
of being offered, and nothing is dismissed.

Both answers also exist unattended: propose.py turns a queued candidate into an
llm.json entry plus a line in check_new-decisions.json, and the reviewer flips
that line to __ignored__ to take the other branch. Every run except --no-add
starts by carrying out whatever those lines already say -- before any score is
fetched, so an ignored model is gone before anything can attach a mapping or a
score to it. See _new_models.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import _prompts
from _artificialanalysis_mapping import mapped_aa_slugs
from _new_models import apply_decisions, dismiss, load_decisions, load_dismissed

HERE = Path(__file__).resolve().parent
DEFAULT_LLM_JSON = HERE / "llm.json"
AA_SCRIPT = HERE / "artificialanalysis.py"
ADD_SCRIPT = HERE / "add.py"
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


def add_model(slug: str, json_file: str) -> bool:
    """Run add.py for a slug, inheriting the terminal for its prompts."""
    proc = subprocess.run([sys.executable, str(ADD_SCRIPT), "--name", slug, json_file])
    return proc.returncode == 0


def prompt_choice(model: dict) -> str:
    """Return 'add', 'skip', 'quit' or 'defer' for one candidate."""
    slug = model["slug"]
    if _prompts.collecting():
        note = " - ".join(
            part
            for part in (
                f"released {model.get('release_date') or '?'}",
                creator_name(model),
                model.get("url") or "",
            )
            if part
        )
        _prompts.record(
            kind="new-model",
            subject=slug,
            question=f"Add newly released model '{slug}' to llm.json?",
            note=note,
            command="./check_new.py",
        )
        return "defer"

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
        help="Only print candidates: never prompt to add, and never apply the "
        "decisions recorded in check_new-decisions.json (report only).",
    )
    _prompts.add_cli_flag(parser)
    args = parser.parse_args()
    _prompts.apply_cli_flag(args)

    if args.date:
        try:
            cutoff = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit("invalid --date, expected YYYY-MM-DD")
    else:
        cutoff = date.today() - timedelta(days=args.days)

    # A decision recorded in a merged proposal PR is carried out here, before
    # any score is fetched: dismiss() is frozen under collect mode, this is not.
    # --no-add promises to change nothing, so it skips this too.
    if not args.no_add:
        for line in apply_decisions(Path(args.json_file)):
            print(line)

    known = (
        existing_slugs(Path(args.json_file))
        | mapped_aa_slugs()
        | ignored_slugs()
        # A slug still awaiting its decision in the open proposal PR is not a
        # candidate: it is already in front of a person.
        | set(load_decisions())
        | load_dismissed()
    )
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

    if args.no_add:
        return 0
    if not _prompts.collecting() and not (sys.stdin.isatty() and sys.stdout.isatty()):
        return 0

    print()
    added = dismissed = deferred = 0
    for m in new:
        slug = m["slug"]
        choice = prompt_choice(m)
        if choice == "quit":
            break
        if choice == "defer":
            deferred += 1
            continue
        if choice == "skip":
            dismiss(slug)
            dismissed += 1
            continue
        if add_model(slug, args.json_file):
            added += 1
        else:
            print(f"  add.py failed for '{slug}'; leaving it for next time.", file=sys.stderr)

    if deferred:
        print(f"\nQueued {deferred} for manual review; nothing dismissed.")
        return 0
    print(f"\nAdded {added}, dismissed {dismissed}.")
    if added:
        print("Run ./update.py -w (or ./update-all) to fill in scores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
