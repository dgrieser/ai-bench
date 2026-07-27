#!/usr/bin/env python3
"""
Fetch SWE-Marathon resolution rates from https://www.swe-marathon.org/

The site is a client-rendered Vite SPA: the leaderboard is not in the served
HTML and there is no JSON endpoint. The data ships as an object-literal array
inside the hashed /assets/*.js bundle, so this script resolves the bundle from
the page's <script src>, finds the array holding the ``pass1`` field and parses
that JS literal directly (the minifier emits backtick strings, bare keys and
``!0``/``!1`` booleans, so json.loads cannot be used).

Each row is one (model, scaffold) pair -- the same model appears more than once
when it was run under several harnesses (e.g. Kimi K2.6 under both Kimi Code CLI
and Terminus 2), and the reported score is scaffold-dependent. ``score`` is the
site's headline "Resolution rate (pass@1)" in percent (binary reward: a task
counts only when every verifier test passes).

The Oracle (held-out solution) and NOP (no actions) rows are reference bounds,
not models, and are dropped unless --include-reference is passed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request


URL = "https://www.swe-marathon.org/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Field that identifies the leaderboard array inside the bundle.
_MARKER = "pass1:"
_SCRIPT_SRC_RE = re.compile(r'<script[^>]+src="([^"]+\.js)"')
_KEY_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_NUMBER_RE = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


class ParseError(ValueError):
    """Raised when the bundle does not hold the expected JS literal."""


def fetch_text(url: str, retries: int = 3, delay: float = 2.0) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            if attempt == retries:
                raise
            wait = delay * attempt
            print(
                f"  attempt {attempt}/{retries} failed ({exc}); retrying in {wait:.0f}s ...",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def bundle_urls(page_html: str, base: str = URL) -> list[str]:
    """Absolute URLs of every <script src> .js asset on the page."""
    urls: list[str] = []
    for src in _SCRIPT_SRC_RE.findall(page_html):
        if src.startswith("http"):
            urls.append(src)
        else:
            urls.append(base.rstrip("/") + "/" + src.lstrip("/"))
    return urls


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def _parse_string(text: str, i: int) -> tuple[str, int]:
    quote = text[i]
    i += 1
    out: list[str] = []
    while i < len(text):
        c = text[i]
        if c == "\\":
            nxt = text[i + 1]
            out.append({"n": "\n", "t": "\t", "r": "\r"}.get(nxt, nxt))
            i += 2
            continue
        if c == quote:
            return "".join(out), i + 1
        if quote == "`" and c == "$" and text[i + 1 : i + 2] == "{":
            # Interpolated template: not data, refuse rather than guess.
            raise ParseError("template interpolation in string literal")
        out.append(c)
        i += 1
    raise ParseError("unterminated string literal")


def _parse_value(text: str, i: int) -> tuple[object, int]:
    i = _skip_ws(text, i)
    if i >= len(text):
        raise ParseError("unexpected end of input")
    c = text[i]

    if c == "{":
        obj: dict[str, object] = {}
        i = _skip_ws(text, i + 1)
        if text[i] == "}":
            return obj, i + 1
        while True:
            i = _skip_ws(text, i)
            if text[i] in "\"'`":
                key, i = _parse_string(text, i)
            else:
                match = _KEY_RE.match(text, i)
                if not match:
                    raise ParseError(f"bad object key at {i}")
                key, i = match.group(0), match.end()
            i = _skip_ws(text, i)
            if text[i] != ":":
                raise ParseError(f"expected ':' at {i}")
            value, i = _parse_value(text, i + 1)
            obj[key] = value
            i = _skip_ws(text, i)
            if text[i] == ",":
                i += 1
                i = _skip_ws(text, i)
                if text[i] == "}":
                    return obj, i + 1
                continue
            if text[i] == "}":
                return obj, i + 1
            raise ParseError(f"expected ',' or '}}' at {i}")

    if c == "[":
        arr: list[object] = []
        i = _skip_ws(text, i + 1)
        if text[i] == "]":
            return arr, i + 1
        while True:
            value, i = _parse_value(text, i)
            arr.append(value)
            i = _skip_ws(text, i)
            if text[i] == ",":
                i += 1
                i = _skip_ws(text, i)
                if text[i] == "]":
                    return arr, i + 1
                continue
            if text[i] == "]":
                return arr, i + 1
            raise ParseError(f"expected ',' or ']' at {i}")

    if c in "\"'`":
        return _parse_string(text, i)

    # Minified booleans: !0 -> true, !1 -> false.
    if c == "!":
        if text[i + 1] == "0":
            return True, i + 2
        if text[i + 1] == "1":
            return False, i + 2
        raise ParseError(f"unsupported '!' expression at {i}")

    for literal, value in (("true", True), ("false", False), ("null", None), ("undefined", None)):
        if text.startswith(literal, i):
            return value, i + len(literal)

    match = _NUMBER_RE.match(text, i)
    if match:
        raw = match.group(0)
        number: object = float(raw) if ("." in raw or "e" in raw or "E" in raw) else int(raw)
        return number, match.end()

    # An identifier here means the array references a variable rather than
    # holding inline data; the shape changed and guessing would be wrong.
    raise ParseError(f"unsupported token at {i}: {text[i : i + 24]!r}")


def extract_rows(bundle: str) -> list[dict]:
    """Parse the leaderboard array (the one carrying ``pass1``) out of a bundle.

    Walks back from the marker to each preceding '[' and returns the first
    candidate that parses as an array of row objects, so an extra wrapping
    array or a reordered minification does not break extraction.
    """
    marker = bundle.find(_MARKER)
    if marker == -1:
        raise ParseError(f"{_MARKER!r} not found in bundle")

    search_from = marker
    for _ in range(10):
        start = bundle.rfind("[", 0, search_from)
        if start == -1:
            break
        try:
            value, _end = _parse_value(bundle, start)
        except (ParseError, IndexError):
            search_from = start
            continue
        if isinstance(value, list):
            rows = [r for r in value if isinstance(r, dict) and "pass1" in r]
            if rows:
                return rows
        search_from = start

    raise ParseError("could not parse a leaderboard array around the pass1 marker")


def get_scores(include_reference: bool = False) -> list[dict]:
    """Return a list of score dicts, best first.

    Keys: model, scaffold, score (pass@1 %), partial_avg, cost_avg, tok_avg,
    trials, incomplete, reference, id, site_rank, rank.
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    page = fetch_text(URL)

    bundle = None
    for asset in bundle_urls(page):
        print(f"Fetching {asset} ...", file=sys.stderr)
        text = fetch_text(asset)
        if _MARKER in text:
            bundle = text
            break
    if bundle is None:
        raise ParseError(f"no page script contained {_MARKER!r}")

    print("Parsing leaderboard data...", file=sys.stderr)
    raw_rows = extract_rows(bundle)

    results: list[dict] = []
    dropped = 0
    for row in raw_rows:
        name = row.get("name")
        pass1 = row.get("pass1")
        if not isinstance(name, str) or not name:
            continue
        if isinstance(pass1, bool) or not isinstance(pass1, (int, float)):
            continue
        reference = bool(row.get("ref"))
        if reference and not include_reference:
            dropped += 1
            continue
        results.append(
            {
                "model": name,
                "scaffold": row.get("scaffold"),
                "score": round(float(pass1), 2),
                "partial_avg": row.get("partialAvg"),
                "cost_avg": row.get("costAvg"),
                "tok_avg": row.get("tokAvg"),
                "trials": row.get("nLoggedTrials"),
                "incomplete": bool(row.get("incomplete")),
                "reference": reference,
                "id": row.get("id"),
                "site_rank": row.get("rank"),
            }
        )

    print(
        f"  parsed {len(results)} rows"
        + (f" ({dropped} reference rows dropped)" if dropped else ""),
        file=sys.stderr,
    )

    results.sort(key=lambda r: -r["score"])
    for i, r in enumerate(results, 1):
        r["rank"] = i
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch SWE-Marathon leaderboard scores.")
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Also keep the Oracle / NOP reference rows (not real models).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = get_scores(include_reference=args.include_reference)

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for name in sorted({entry["model"] for entry in scores}):
            print(name)
    else:
        col_widths = [
            max(len("MODEL"), max((len(e["model"]) for e in scores), default=0)),
            6,
            max(len("SCAFFOLD"), max((len(e.get("scaffold") or "") for e in scores), default=0)),
            6,
        ]
        fmt = (
            f"{{:<{col_widths[0]}}}  {{:>{col_widths[1]}}}  "
            f"{{:<{col_widths[2]}}}  {{:>{col_widths[3]}}}"
        )
        print(fmt.format("MODEL", "PASS@1", "SCAFFOLD", "TRIALS"))
        for entry in scores:
            print(
                fmt.format(
                    entry["model"],
                    str(entry["score"]),
                    entry.get("scaffold") or "",
                    str(entry.get("trials") if entry.get("trials") is not None else ""),
                )
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
