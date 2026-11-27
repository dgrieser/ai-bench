#!/usr/bin/env python3
"""
Fetch Artificial Analysis Coding Agent Index benchmark scores.

https://artificialanalysis.ai/agents/coding-agents publishes AA's own runs of
the coding-agent benchmarks -- since Coding Agent Index v2.1 (2026-11)
FrontierCode-AA 2.0, Terminal-Bench 4.1 and SWE-Atlas-QnA-AA, before that
DeepSWE v1.1, SWE-Atlas-QnA and Terminal-Bench 4.0 -- one row per (agent
harness, model) pair, e.g. "Claude Code - Opus 5 (xhigh)", and the index
itself, the plain average of the three.

Each model runs under its own vendor's agent (Claude Code, Codex, Opencode)
rather than AA's single harness on the model pages, so the two AA surfaces
disagree by design where they share a column (FrontierCode 2.0). The model
pages outrank this one there (_precedence.RANK_AA); everywhere else this page
outranks the benchmark sites, tbench.ai's 4.1 board included.

AA's dataset ids do not always name the revision they run: Terminal-Bench
stayed "terminal-bench-v4" when the page moved from 4.0 to 4.1. So a dataset
whose id names no minor revision is read only while the page's own label for it
(the benchmark list's "Terminal-Bench 4.1") is the one LABEL_GATES expects, and
the index only while the page names the version INDEX_VERSION expects
("Artificial Analysis Coding Agent Index v2.1"). Anything else is skipped and
reported on stderr: a new revision needs a column of its own.

The page is a Next.js App Router app; requesting it with an "RSC: 1" header
returns the flight payload, in which every leaderboard row is a plain JSON
object ({"id":"<32 hex>", ..., "evals":[...]}) that json.JSONDecoder can
decode in place -- no headless browser, mirroring fetch_swe_atlas.py. The
index's version is rendered only into the HTML, which is fetched once more
without the header.

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

from _fetch_checks import check_fractions, require_rows


URL = "https://artificialanalysis.ai/agents/coding-agents"

# datasetIndexName in the page's eval records -> llm.json benchmark key.
#
# AA re-cut the index on both versioned datasets at once: "terminal-bench-v2.1"
# became "terminal-bench-v4", and "deep-swe" -- left out before because a
# dataset id naming no revision cannot pick between llm.json's per-revision
# columns -- now names one, so it is ingested as the comment here always said
# it would be. An id this file does not know costs one line on stderr and
# nothing else, so both columns went unfed between the rename and this change
# while the page kept publishing them. Index v2.1 then dropped DeepSWE for
# FrontierCode 2.0 (whose id names its revision) and moved Terminal-Bench to
# 4.1 under the unchanged "terminal-bench-v4", which is why LABEL_GATES exists.
DATASETS: dict[str, str] = {
    "deep-swe-v1.1": "deepswe_1_1",
    "frontiercode-v2.0": "frontiercode_2_0",
    # AA's own run of SWE Atlas Codebase Q&A, which it labels SWE-Atlas-QnA-AA
    # since Coding Agent Index v2.0: its own judge (GPT-6 Luna, 5 votes per
    # criterion) and 127 rubric criteria reworded. That is neither Scale's
    # v1.0 nor its v1.1, so it has a column of its own rather than one of
    # Scale's per-revision columns.
    "swe-atlas-qna": "swe_atlas_qna_aa",
    "terminal-bench-v4": "terminal_bench_4_1",
}

# Datasets whose id names no minor revision -> the label the page's benchmark
# list must give them for DATASETS' column to be right. A different label (or
# none) skips the dataset. FrontierCode's and DeepSWE's ids carry the revision.
# "swe-atlas-qna" kept its id when AA changed the grading under it, so the label
# is what says it is still the -AA variant swe_atlas_qna_aa holds.
LABEL_GATES: dict[str, str] = {
    "swe-atlas-qna": "SWE-Atlas-QnA-AA",
    "terminal-bench-v4": "Terminal-Bench 4.1",
}

# The index itself: each row's indexScore (0-1), read while the page names
# this version of it.
INDEX_KEY = "aa_coding_agent_index_2_1"
INDEX_VERSION = "2.1"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    # Ask for the React Server Component payload instead of the HTML shell.
    "RSC": "1",
}

_ROW_START_RE = re.compile(r'\{"id":"[0-9a-f]{32}"')
# The benchmark list in the flight payload, keyed by dataset id:
# ["$","li","terminal-bench-v4",{"children":[["$","span",null,{...,"children":"Terminal-Bench 4.1"}]
_DATASET_LABEL_RE = re.compile(
    r'\["\$","li","(?P<dataset>[^"]+)",\{"children":\[\["\$","span",null,'
    r'\{"className":"font-medium","children":"(?P<label>[^"]+)"'
)
_INDEX_VERSION_RE = re.compile(r"Coding Agent Index v(\d+(?:\.\d+)*)")


def fetch_payload(
    url: str, retries: int = 3, delay: float = 2.0, rsc: bool = True
) -> str:
    headers = HEADERS if rsc else {k: v for k, v in HEADERS.items() if k != "RSC"}
    req = urllib.request.Request(url, headers=headers)
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


def dataset_labels(payload: str) -> dict[str, str]:
    """{dataset id: the label the page's benchmark list gives it}."""
    return {m["dataset"]: m["label"] for m in _DATASET_LABEL_RE.finditer(payload)}


def index_version(html: str) -> str | None:
    """The Coding Agent Index version the page names, the most-cited one."""
    found = _INDEX_VERSION_RE.findall(html)
    if not found:
        return None
    return max(set(found), key=found.count)


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
    payload = fetch_payload(URL)
    rows = extract_rows(payload)
    print(f"  parsed {len(rows)} leaderboard rows", file=sys.stderr)

    labels = dataset_labels(payload)
    gated_out: set[str] = set()
    for dataset, expected in LABEL_GATES.items():
        if labels.get(dataset) != expected:
            gated_out.add(dataset)
            print(
                f"  not reading {dataset}: the page labels it "
                f"{labels.get(dataset)!r}, {DATASETS[dataset]} holds {expected!r}",
                file=sys.stderr,
            )
    version = index_version(fetch_payload(URL, rsc=False))
    read_index = version == INDEX_VERSION
    if not read_index:
        print(
            f"  not reading the index: the page names "
            f"{'Coding Agent Index v' + version if version else 'no index version'}, "
            f"{INDEX_KEY} holds v{INDEX_VERSION}",
            file=sys.stderr,
        )

    results: list[dict] = []
    ignored: set[str] = set()
    for obj in rows:
        display = obj["display"]
        raw_model = display["model"]
        agent = obj.get("agentName")
        label = obj.get("displayLabel") or raw_model
        base = {"model": normalize_model(raw_model), "raw": label, "agent": agent}
        index_score = obj.get("indexScore")
        components, evals = obj.get("indexComponentCount"), obj.get("evalCount")
        # An index over fewer than all its components is not the index.
        complete = not (isinstance(components, int) and isinstance(evals, int)) or evals >= components
        if (
            read_index and complete
            and isinstance(index_score, (int, float)) and not isinstance(index_score, bool)
        ):
            check_fractions([index_score], URL, "indexScore")
            results.append({**base, "key": INDEX_KEY, "score": round(float(index_score) * 100.0, 2)})
        for entry in obj["evals"]:
            if not isinstance(entry, dict):
                continue
            dataset = entry.get("datasetIndexName")
            mean = entry.get("mean")
            reward = mean.get("reward") if isinstance(mean, dict) else None
            if not isinstance(reward, (int, float)) or isinstance(reward, bool):
                continue
            # reward is a 0-1 fraction, multiplied into a percentage below.
            check_fractions([reward], URL, f"{dataset} reward")
            key = DATASETS.get(dataset)
            if key is None:
                if isinstance(dataset, str):
                    ignored.add(dataset)
                continue
            if dataset in gated_out:
                continue
            results.append({**base, "key": key, "score": round(float(reward) * 100.0, 2)})
    # A new index revision can add a benchmark; surface it instead of silence.
    for dataset in sorted(ignored):
        print(f"  ignoring unmapped dataset: {dataset}", file=sys.stderr)
    # A renamed "reward" or dataset index drops every row one at a time.
    require_rows(results, URL, "rows for a mapped dataset")
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
