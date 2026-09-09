#!/usr/bin/env python3
"""Rename a model's slug in llm.json and in every mapping file that names it.

  ./rename.py glm-5.3 glm-5-3        # what it would change
  ./rename.py glm-5.3 glm-5-3 -w     # change it

Default is a dry-run; pass -w/--write to persist changes (same convention as
update.py and prune.py).

The point of a rename is usually Artificial Analysis: update.py reads AA
directly for a model whose name *is* an AA slug, so renaming a hand-added entry
to the slug AA later published is what connects the two -- no mapping entry, and
nothing to keep in step afterwards. The alternative is to keep the name and map
it (`./answer.py update_artificialanalysis_mapping.py <name> <slug> -w`), which
is the right move when the name is one this site wants to keep.

Why this is not an edit of llm.json: a name left behind in a mapping file does
not fail loudly, it silently stops matching, and the scores that source used to
write simply stop arriving. _rename.py lists every place a name is written down.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _rename
from _rename import RenameError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rename a model slug everywhere it is written down.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("old", help="The model's current name in llm.json.")
    parser.add_argument("new", help="The name to give it, e.g. an Artificial Analysis slug.")
    parser.add_argument(
        "--llm-json",
        default=str(_rename.DEFAULT_LLM_JSON),
        help='Path to llm.json (default: "./llm.json" next to this script).',
    )
    parser.add_argument(
        "--write",
        "-w",
        action="store_true",
        help="Actually write. Without it, nothing changes.",
    )
    args = parser.parse_args(argv)

    llm_path = Path(args.llm_json)
    try:
        changes = _rename.plan(args.old, args.new, llm_path)
    except RenameError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Renaming {args.old!r} to {args.new!r} touches {len(changes)} file(s):")
    for change in changes:
        print(f"  {change}")

    if not args.write:
        print("\nNothing written. Pass -w to apply.")
        return 0

    try:
        log = _rename.rename(args.old, args.new, llm_path)
    except RenameError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print()
    for line in log:
        print(f"  {line}")
    print(
        "\nScores follow the entry; nothing is refetched here. The next refresh "
        "reads Artificial Analysis under the new name if it is a slug there."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
