"""Where each name a source publishes was read: the link on an admin-page card.

A mapping question names a model the way a board writes it ("Fable 5.1",
"xingchen-agi/xing4.0-29b-a4b"), and answering it usually means looking at
that board. The asking script knows which page it read the name from, so it
says so: `prompt_select_or_new(..., source_url=...)` puts the URL on the queued
question, and the page links it.

Three shapes, depending on the source:

  * one board, one page -- the URL is the fetcher's own constant, and every
    name gets it (`same_page`);
  * a model page per name -- OpenRouter, Spheron, llm-stats -- built from the
    name itself;
  * several boards -- Terminal-Bench versions, Vals or evals.report benchmarks,
    SWE Atlas tracks -- where the name's page is the board its row came from.
    The fetcher knows that and nobody else does, so it prints it: `--format
    pages` emits {name: url} from the same rows `--format names` lists
    (`emit`), and the mapping module reads it back (`run`).

A name on more than one board gets the first board its rows list, which is the
fetcher's own board order.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


def same_page(names: Iterable[str], url: str | None) -> dict[str, str | None]:
    """Every name read from one page."""
    return {name: url for name in names}


def pages(rows: Iterable[dict[str, Any]], url_of: Callable[[dict[str, Any]], str | None]
          ) -> dict[str, str | None]:
    """Name -> the page its first row was read from, in the order the rows list them."""
    found: dict[str, str | None] = {}
    for row in rows:
        name = row.get("model")
        if isinstance(name, str) and name and found.get(name) is None:
            found[name] = url_of(row)
    return found


def emit(rows: Iterable[dict[str, Any]], url_of: Callable[[dict[str, Any]], str | None]) -> None:
    """`--format pages`: the mapping JSON on stdout."""
    print(json.dumps(pages(rows, url_of), ensure_ascii=False, indent=1))


def run(script: Path, *args: str) -> dict[str, str | None]:
    """Run `script --format pages` and read the mapping back."""
    proc = subprocess.run(
        [sys.executable, str(script), "--format", "pages", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{script.name} failed ({proc.returncode}): {proc.stderr.strip()}")
    raw: Any = json.loads(proc.stdout)
    if not isinstance(raw, dict):
        raise RuntimeError(f"{script.name} --format pages did not print an object")
    return {
        name: url if isinstance(url, str) and url else None
        for name, url in raw.items()
        if isinstance(name, str) and name
    }
