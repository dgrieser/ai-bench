"""Leaderboard rows out of a labs.scale.com page, with the board's identity checked.

The Scale Labs leaderboards (MCP-Atlas, the three SWE Atlas tracks) are one
Next.js App Router app: the rows travel in the page's self.__next_f flight
payload as an `"entries":[{...}, ...]` array of flat objects. The fetchers used
to pick up *any* flat object carrying "model" and "score" anywhere in that
payload, which would silently merge a second table -- a sidebar of another
board, a "related leaderboards" strip -- into this one.

This reader is stricter in two ways:

  * the payload has to say which board it is: the router records the slug it
    rendered as `"query":{"slug":"<slug>"}`, and a page whose slug is not the
    one asked for (a redirect to another board, a renamed route) is refused;
  * rows come only from `"entries"` arrays, and every such array on the page
    has to be the same table -- the flight payload may repeat one, but two
    different ones cannot be told apart and are refused rather than merged.
"""

from __future__ import annotations

import json
import re
from typing import Any

_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,(".*?")\]\)', re.DOTALL)
_ENTRIES_RE = re.compile(r'"entries":\s*\[')
_DECODER = json.JSONDecoder()


def decode_flight(html: str) -> str:
    """Concatenate the decoded self.__next_f flight chunks into one string."""
    decoded = ""
    for chunk in _PUSH_RE.findall(html):
        try:
            decoded += json.loads(chunk)
        except json.JSONDecodeError:
            continue
    return decoded


def _is_row(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("model"), str)
        and isinstance(obj.get("score"), (int, float))
        and not isinstance(obj.get("score"), bool)
    )


def board_slugs(decoded: str) -> set[str]:
    """Every route slug the payload says it rendered."""
    return set(re.findall(r'"query":\{"slug":"([^"]*)"\}', decoded))


def extract_board_rows(html: str, slug: str, source: str) -> list[dict]:
    """The rows of board `slug`, de-duplicated, in page order.

    Raises ValueError when the page is another board, carries no row table,
    or carries more than one distinct one.
    """
    decoded = decode_flight(html)
    slugs = board_slugs(decoded)
    if slug not in slugs:
        raise ValueError(
            f"{source}: the page does not identify itself as board {slug!r} "
            f"(found {sorted(slugs) or 'no slug'}) -- refusing to read another board's rows."
        )

    tables: list[list[dict]] = []
    for match in _ENTRIES_RE.finditer(decoded):
        try:
            value, _ = _DECODER.raw_decode(decoded, match.end() - 1)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, list) or not value or not all(_is_row(v) for v in value):
            continue
        if value not in tables:
            tables.append(value)

    if not tables:
        raise ValueError(
            f"{source}: no leaderboard rows in the flight payload -- the page layout "
            "changed; refusing to report an empty leaderboard."
        )
    if len(tables) > 1:
        sizes = ", ".join(str(len(t)) for t in tables)
        raise ValueError(
            f"{source}: the flight payload carries {len(tables)} different row tables "
            f"({sizes} rows) -- refusing to guess which one is board {slug!r}."
        )

    rows: list[dict] = []
    seen: set[tuple[str, float]] = set()
    for obj in tables[0]:
        ident = (obj["model"], obj["score"])
        if ident in seen:
            continue
        seen.add(ident)
        rows.append(obj)
    return rows
