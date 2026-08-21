#!/usr/bin/env python3
"""
Fetch MCP-Atlas (pass rate %) scores from the Scale Labs leaderboard.

MCP-Atlas (Scale AI) grades an agent on real MCP servers; the leaderboard at
https://labs.scale.com/leaderboard/mcp_atlas runs the benchmark's 500-task
public subset and reports Scale's own runs, so it is the benchmark's first-party
source. Labs republishing a self-reported MCP-Atlas number -- llm-stats and
evals.report do, and so do model cards -- disagree with it by a point or two;
update.py therefore runs this scraper after those.

The page is the same Next.js App Router app as the SWE Atlas boards, with the
same flight-payload row shape, so the extraction mirrors fetch_swe_atlas.py.

Row labels mix display names and API ids, with the harness or reasoning effort
in a parenthetical and decimal points sometimes spelled "p": "Muse Spark 1.1",
"Inkling (xHigh)", "glm-5p2", "kimi-k2p5". The reported `model` is therefore a
normalized base name -- parenthetical stripped, "5p2" restored to "5.2",
separators unified, lowercased -- so one name -> slug mapping covers a model's
variants and the key reads like the other Scale Labs mapping files' keys;
`raw` and `harness` keep the original label.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request


URL = "https://labs.scale.com/leaderboard/mcp_atlas"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,(".*?")\]\)', re.DOTALL)
# Flat row objects in the flight payload; entries carry no nested braces.
_ROW_RE = re.compile(r'\{[^{}]*"score":[^{}]*\}')
# Reasoning-effort modifiers that trail a model name (case-insensitive).
_EFFORT_RE = re.compile(r"\b(?:xhigh|x-high|high|medium|low|max)\b", re.IGNORECASE)
# Scale spells a version's decimal point as "p" in its API ids ("glm-5p2").
_DECIMAL_P_RE = re.compile(r"(?<=[0-9])p(?=[0-9])")


def fetch_html(url: str = URL, retries: int = 3, delay: float = 2.0) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
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


def decode_flight(html: str) -> str:
    """Concatenate the decoded self.__next_f flight chunks into one string."""
    decoded = ""
    for chunk in _PUSH_RE.findall(html):
        try:
            decoded += json.loads(chunk)
        except json.JSONDecodeError:
            continue
    return decoded


def extract_rows(html: str) -> list[dict]:
    """Extract leaderboard row objects from the flight payload."""
    decoded = decode_flight(html)
    rows: list[dict] = []
    seen: set[int] = set()
    for match in _ROW_RE.finditer(decoded):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if not isinstance(obj.get("model"), str) or not isinstance(obj.get("score"), (int, float)):
            continue
        # Flight payloads can repeat the array; de-dup on identity of (model, score).
        ident = hash((obj["model"], obj["score"]))
        if ident in seen:
            continue
        seen.add(ident)
        rows.append(obj)
    if not rows:
        raise ValueError(
            f"No leaderboard rows in the flight payload of {URL} — the page layout "
            "changed; refusing to report an empty leaderboard."
        )
    return rows


def split_harness(raw: str) -> tuple[str, str | None]:
    """Split a leaderboard label into (label_without_parenthetical, modifier).

    "Inkling (xHigh)"          -> ("Inkling", "xHigh")
    "gpt-5.6 (sol)"            -> ("gpt-5.6", "sol")
    "Muse Spark 1.1"           -> ("Muse Spark 1.1", None)
    """
    match = re.search(r"\(([^)]*)\)", raw)
    harness = match.group(1).strip() if match else None
    without = re.sub(r"\s*\([^)]*\)\s*", " ", raw).strip()
    return without, harness


def normalize_model(raw: str) -> str:
    """Normalize a leaderboard label to a base model name for mapping.

    Strips the parenthetical and reasoning-effort modifiers, restores Scale's
    "p" decimal spelling, unifies separators and lowercases, so the key matches
    the shape the other Scale Labs mapping files use: "glm-5p2" -> "glm 5.2",
    "Inkling (xHigh)" -> "inkling".
    """
    without, _ = split_harness(raw)
    without = _EFFORT_RE.sub(" ", without)
    without = _DECIMAL_P_RE.sub(".", without)
    without = without.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", without).strip().lower()


def get_scores(include_deprecated: bool = False) -> list[dict]:
    """Return one dict per leaderboard row.

    Keys: model (normalized base), raw, harness, score (pass rate %), ci,
    company, contamination, rank (rank within the leaderboard, 1 = best).
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    rows = extract_rows(fetch_html())

    kept: list[dict] = []
    dropped = 0
    for obj in rows:
        if obj.get("deprecated") and not include_deprecated:
            dropped += 1
            continue
        raw = obj["model"]
        _, harness = split_harness(raw)
        kept.append(
            {
                "model": normalize_model(raw),
                "raw": raw,
                "harness": harness,
                "score": round(float(obj["score"]), 2),
                "ci": obj.get("confidenceInterval_upper"),
                "company": obj.get("company"),
                "contamination": obj.get("contaminationMessage") or None,
            }
        )
    print(
        f"  parsed {len(kept)} leaderboard rows"
        + (f" ({dropped} deprecated dropped)" if dropped else ""),
        file=sys.stderr,
    )

    kept.sort(key=lambda r: -r["score"])
    for i, entry in enumerate(kept, 1):
        entry["rank"] = i
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch MCP-Atlas leaderboard scores from Scale Labs."
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--include-deprecated",
        action="store_true",
        help="Also keep rows Scale has flagged as deprecated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = get_scores(args.include_deprecated)

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for name in sorted({entry["model"] for entry in scores}):
            print(name)
    else:
        col_widths = [
            max(len("MODEL"), max((len(e["model"]) for e in scores), default=0)),
            6,
            max(len("RAW"), max((len(e["raw"]) for e in scores), default=0)),
        ]
        fmt = f"{{:<{col_widths[0]}}}  {{:>{col_widths[1]}}}  {{:<{col_widths[2]}}}"
        print(fmt.format("MODEL", "PASS", "RAW"))
        for entry in scores:
            print(fmt.format(entry["model"], str(entry["score"]), entry["raw"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch the MCP-Atlas leaderboard: {exc}", file=sys.stderr)
        raise SystemExit(1)
