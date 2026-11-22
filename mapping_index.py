#!/usr/bin/env python3
"""Publish every model mapping, per model, for the admin page's Models tab.

  ./mapping_index.py              a readable summary per route
  ./mapping_index.py --json       the document, as the admin page reads it
  ./mapping_index.py --publish    write it to _pending/mappings.json
  ./mapping_index.py --model hy4  what feeds one model

Reads only. Edits go through answer.py, as `mapping-edit` and
`rejection-edit` records:

  echo '[{"kind": "mapping-edit", "route": "update_openrouter_mapping.py",
          "route_kind": "*", "subject": "tencent/hy4-preview",
          "answer": "hy4", "if_previous": ["__unmappable__"]}]' \\
    | ./answer.py --stdin --no-queue-check -w

See _mapping_index.py for what is published and why.
"""

from __future__ import annotations

import argparse
import json
import sys

import _mapping_index as index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--json", action="store_true", help="print the document")
    group.add_argument("--publish", action="store_true", help=f"write {index.PUBLISHED.relative_to(index.HERE)}")
    group.add_argument("--model", metavar="NAME", help="list what feeds one model")
    args = parser.parse_args(argv)

    if args.publish:
        changed = index.publish()
        print(f"{index.PUBLISHED.relative_to(index.HERE)}: {'written' if changed else 'unchanged'}")
        return 0

    doc = index.build()
    if args.json:
        json.dump(doc, sys.stdout, indent=1, ensure_ascii=False, sort_keys=True)
        print()
        return 0

    if args.model:
        found = False
        for route in doc["routes"]:
            rid = route["id"]
            names = [n for n, v in doc["files"][rid].items()
                     if (v if isinstance(v, list) else [v]).count(args.model)
                     or (route["universe"] == "aa-slugs" and n == args.model)]
            supplied = doc["contributions"].get(args.model, {}).get(rid, [])
            if names or supplied:
                found = True
                print(f"{route['route']} ({route['route_kind']})")
                for name in names:
                    print(f"  {name} -> {doc['files'][rid][name]}")
                if supplied:
                    print(f"  supplies: {', '.join(supplied)}")
        for extra in (index.CARD, index.OTHER):
            supplied = doc["contributions"].get(args.model, {}).get(extra, [])
            if supplied:
                found = True
                print(f"{extra}: {', '.join(supplied)}")
        if not found:
            print(f"nothing maps onto {args.model!r}")
        return 0

    for route in doc["routes"]:
        rid = route["id"]
        print(f"{route['route']:<40} {route['route_kind']:<16} {len(doc['files'][rid]):>4} names  {route['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
