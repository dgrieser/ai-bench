#!/usr/bin/env python3
"""
Fetch Artificial Analysis Coding Agent Index benchmark scores.

https://artificialanalysis.ai/agents/coding-agents publishes AA's own runs of
the coding-agent benchmarks — DeepSWE, SWE-Atlas-QnA and Terminal-Bench v2.1 —
one row per (agent harness, model) pair, e.g. "Claude Code - Opus 5 (xhigh)".
This is the only AA surface carrying DeepSWE and SWE-Atlas numbers: the model
pages and the v2 API report neither.

The page is a Next.js App Router app; requesting it with an "RSC: 1" header
returns the flight payload, in which every leaderboard row is a plain JSON
object ({"id":"<32 hex>", ..., "evals":[...]}) that json.JSONDecoder can
decode in place — no headless browser, mirroring fetch_swe_atlas.py.

Row labels embed the agent and modifiers ("Opus 5 (xhigh)", "Fable 5 (max)
(with fallback)"), so the reported `model` is a normalized base name with
every parenthetical stripped, and one name -> slug mapping covers all agent
and effort variants; `raw` and `agent` are kept for transparency.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request


URL = "https://artificialanalysis.ai/agents/coding-agents"

# datasetIndexName in the page's eval records -> llm.json benchmark key.
DATASETS: dict[str, str] = {
    "deep-swe": "deepswe",
    "swe-atlas-qna": "swe_atlas_qna",
    "terminal-bench-v2.1": "terminal_bench_2_1",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    # Ask for the React Server Component payload instead of the HTML shell.
    "RSC": "1",
}

_ROW_START_RE = re.compile(r'\{"id":"[0-9a-f]{32}"')


def fetch_payload(url: str, retries: int = 3, delay: float = 2.0) -> str:
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


def extract_rows(payload: str) -> list[dict]:
    """Decode every leaderboard row object out of the flight payload.

    Rows appear both in the highlight sections and the full table, so they are
    de-duplicated on the row id AA assigns to each (agent, model) pair.
    """
    decoder = json.JSONDecoder()
    rows: list[dict] = []
    seen: set[str] = set()
    for match in _ROW_START_RE.finditer(payload):
        try:
            obj, _ = decoder.raw_decode(payload, match.start())
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("evals"), list):
            continue
        display = obj.get("display")
        if not isinstance(display, dict) or not isinstance(display.get("model"), str):
            continue
        row_id = obj.get("id")
        if row_id in seen:
            continue
        seen.add(row_id)
        rows.append(obj)
    return rows


def normalize_model(raw: str) -> str:
    """Normalize a row's model label to a base name for mapping.

    Every parenthetical is a modifier — reasoning effort ("(xhigh)"), a mode
    ("(thinking)"), a routing note ("(with fallback)") — never the model, so
    all of them are stripped, separators unified and the result lowercased:
    "Fable 5 (max) (with fallback)" -> "fable 5", "GLM-5.2" -> "glm 5.2".
    """
    without = re.sub(r"\s*\([^)]*\)\s*", " ", raw)
    without = without.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", without).strip().lower()


def get_scores() -> list[dict]:
    """Return one row dict per (agent, model, benchmark) measurement.

    Keys: model (normalized base), raw (the row's display label), agent,
    key (llm.json benchmark key), score (percentage).
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    rows = extract_rows(fetch_payload(URL))
    print(f"  parsed {len(rows)} leaderboard rows", file=sys.stderr)

    results: list[dict] = []
    ignored: set[str] = set()
    for obj in rows:
        display = obj["display"]
        raw_model = display["model"]
        agent = obj.get("agentName")
        label = obj.get("displayLabel") or raw_model
        for entry in obj["evals"]:
            if not isinstance(entry, dict):
                continue
            dataset = entry.get("datasetIndexName")
            mean = entry.get("mean")
            reward = mean.get("reward") if isinstance(mean, dict) else None
            if not isinstance(reward, (int, float)) or isinstance(reward, bool):
                continue
            key = DATASETS.get(dataset)
            if key is None:
                if isinstance(dataset, str):
                    ignored.add(dataset)
                continue
            results.append(
                {
                    "model": normalize_model(raw_model),
                    "raw": label,
                    "agent": agent,
                    "key": key,
                    "score": round(float(reward) * 100.0, 2),
                }
            )
    # A new index revision can add a benchmark; surface it instead of silence.
    for dataset in sorted(ignored):
        print(f"  ignoring unmapped dataset: {dataset}", file=sys.stderr)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Artificial Analysis Coding Agent Index benchmark scores."
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = get_scores()

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for name in sorted({entry["model"] for entry in scores}):
            print(name)
    else:
        col_widths = [
            max(len("MODEL"), max((len(e["model"]) for e in scores), default=0)),
            6,
            max(len("KEY"), max((len(e["key"]) for e in scores), default=0)),
            max(len("RAW"), max((len(e["raw"]) for e in scores), default=0)),
        ]
        fmt = f"{{:<{col_widths[0]}}}  {{:>{col_widths[1]}}}  {{:<{col_widths[2]}}}  {{:<{col_widths[3]}}}"
        print(fmt.format("MODEL", "SCORE", "KEY", "RAW"))
        for entry in scores:
            print(fmt.format(entry["model"], str(entry["score"]), entry["key"], entry["raw"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch the Coding Agent Index: {exc}", file=sys.stderr)
        raise SystemExit(1)
