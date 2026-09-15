#!/usr/bin/env python3
"""Fetch benchmark scores from the llm-stats.com Open LLM Leaderboard.

The leaderboard page is backed by the zeroeval API. A single endpoint returns
every model as a flat record with one column per benchmark (keys ending in
``_score``, values on a 0-1 scale). The source benchmark label is that key with
the ``_score`` suffix stripped (e.g. ``gpqa_score`` -> ``gpqa``).

Records also carry a ``license`` ("proprietary" or an open licence name), which
is reported as ``open_weights`` so closed models can be recognised without a
lookup elsewhere.

**HLE is the one column the flat endpoint cannot be read off directly.**
``hle_score`` is a single field over a mixed population: llm-stats republishes
whatever headline each lab printed, and the per-model endpoint's
``analysis_method`` string splits the 104 models carrying one into roughly 38
run *with* tools, 29 *without*, and 37 that never say. Our ``hle`` column is the
no-tools column (Artificial Analysis' text-only, no-tools run supplies 93% of
it), and tools are worth a median +11.5 points on this benchmark -- so taking
the flat field would blend two measurements a third of the way apart. See
docs/hle-tool-mode-audit-2026-09.md.

``resolve_hle_no_tools()`` therefore re-reads HLE per model from
``MODEL_URL``, where ``analysis_method`` says what was run, and emits it under
the label ``hle (no tools)``:

  * a note stating both modes ("With tools: 57.4%. Without tools: 43.2%") has
    the without-tools figure parsed out of it and used;
  * a note stating only no-tools keeps the headline score;
  * a note stating tools, or saying nothing about them, drops the score
    entirely. An unverified number is not a no-tools number.

The bare ``hle`` label is never emitted, so it cannot be ingested by accident.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

from _openness import license_open

URL = "https://api.zeroeval.com/leaderboard/models/full"
# Per-model endpoint. Same records as URL plus a ``benchmarks`` list, where each
# entry carries the ``analysis_method`` prose the flat endpoint drops.
MODEL_URL = "https://api.zeroeval.com/leaderboard/models/{model_id}"
# Human-facing page publishing the same data; stored as the per-score source
# URL because the API host is not a page a reader can open.
LEADERBOARD_URL = "https://llm-stats.com/leaderboards/open-llm-leaderboard"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) ai-bench-fetcher/1.0"

_SCORE_SUFFIX = "_score"

# The flat-endpoint label we refuse to publish as-is, and the qualified label
# resolve_hle_no_tools() puts in its place.
HLE_LABEL = "hle"
HLE_NO_TOOLS_LABEL = "hle (no tools)"
# llm-stats' benchmark_id for the full 2,500-question board. Its sibling
# "hle-verified" is a different question set and never feeds hle_score.
HLE_BENCHMARK_ID = "humanity\u2019s-last-exam".replace("\u2019", "'")

_WITH_TOOLS_RE = re.compile(
    r"\bwith(?:\s+|-)(?:tools?|search|browsing|retrieval)\b"
    r"|\bw/\s*tools?\b|\bsearch agent\b|\btool[- ]augmented\b",
    re.IGNORECASE,
)
_NO_TOOLS_RE = re.compile(
    r"\bno\s+tools?\b|\bwithout\s+tools?\b|\bw/o\s+tools?\b|\btool[- ]free\b",
    re.IGNORECASE,
)
# "Without tools: 43.2%" / "no tools: 43.2 %" -- the figure a both-modes note
# names for the run we actually want.
_NO_TOOLS_VALUE_RE = re.compile(
    r"(?:without\s+tools?|no\s+tools?|w/o\s+tools?)[^0-9%]{0,20}?(\d{1,3}(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


def fetch_json(url: str, timeout: int = 60) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _hle_analysis_method(model_id: str, timeout: int = 30) -> str | None:
    """The ``analysis_method`` llm-stats records for this model's HLE run.

    Returns None when the model has no HLE entry, when the entry carries no
    method, or when the request fails -- all of which end the same way at the
    call site: the score is dropped rather than guessed at.
    """
    url = MODEL_URL.format(model_id=urllib.parse.quote(model_id, safe=""))
    try:
        payload = fetch_json(url, timeout=timeout)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"warning: {model_id}: HLE detail fetch failed: {exc}", file=sys.stderr)
        return None
    if not isinstance(payload, dict):
        return None
    for entry in payload.get("benchmarks") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("benchmark_id") != HLE_BENCHMARK_ID:
            continue
        method = entry.get("analysis_method")
        return method if isinstance(method, str) and method.strip() else None
    return None


def hle_no_tools_score(headline: float, method: str | None) -> float | None:
    """The no-tools HLE score this record supports, or None if it supports none.

    ``headline`` is the flat endpoint's 0-1 ``hle_score``; ``method`` the prose
    from the per-model endpoint. The three cases are spelled out in the module
    docstring. A note naming both modes wins over one naming neither, so the
    parse is tried before the with/without classification.
    """
    if not isinstance(method, str) or not method.strip():
        return None
    with_tools = bool(_WITH_TOOLS_RE.search(method))
    no_tools = bool(_NO_TOOLS_RE.search(method))
    if with_tools and no_tools:
        # Both runs described. Use the figure the note gives for the no-tools
        # one; without a parsable figure the headline is the tools run, so
        # there is nothing here to keep.
        match = _NO_TOOLS_VALUE_RE.search(method)
        if match is None:
            return None
        value = float(match.group(1))
        return value / 100.0 if value > 1.0 else value
    if no_tools:
        return headline
    return None


def resolve_hle_no_tools(results: list[dict], timeout: int = 30) -> None:
    """Replace every record's raw ``hle`` label with a verified ``hle (no tools)``.

    Mutates ``results`` in place. Only models carrying an HLE score are looked
    up, so this costs one extra request per scored model rather than per model.
    """
    scored = [r for r in results if r["scores"].get(HLE_LABEL) is not None]
    if scored:
        print(
            f"Resolving HLE tool mode for {len(scored)} model(s) ...",
            file=sys.stderr,
        )
    kept = 0
    for record in results:
        headline = record["scores"].pop(HLE_LABEL, None)
        if headline is None:
            continue
        method = _hle_analysis_method(record["model"], timeout=timeout)
        value = hle_no_tools_score(headline, method)
        if value is None:
            continue
        record["scores"][HLE_NO_TOOLS_LABEL] = value
        kept += 1
    if scored:
        print(
            f"  kept {kept} of {len(scored)} as no-tools runs; "
            f"dropped {len(scored) - kept} run with tools or not stated",
            file=sys.stderr,
        )


def get_scores(resolve_hle: bool = True) -> list[dict]:
    """Return a list of dicts with keys: model, name, license, open_weights, scores.

    ``model`` is the llm-stats model_id, ``scores`` maps the source benchmark
    label (``_score`` suffix stripped) to its raw 0-1 value. ``open_weights`` is
    None when the record carries no licence. Records without any non-null score
    are dropped.

    ``resolve_hle`` re-reads HLE per model so only verified no-tools runs are
    published, under the label ``hle (no tools)``; see the module docstring. It
    costs one request per model carrying an HLE score, so callers that only
    want model ids or licences turn it off.
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    payload = fetch_json(URL)
    if not isinstance(payload, list):
        raise ValueError("Unexpected response: expected a JSON list")

    results: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        model_id = item.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            continue
        scores: dict[str, float] = {}
        for key, value in item.items():
            if not key.endswith(_SCORE_SUFFIX):
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            label = key[: -len(_SCORE_SUFFIX)]
            scores[label] = value
        if not scores:
            continue
        results.append(
            {
                "model": model_id,
                "name": item.get("name"),
                "license": item.get("license"),
                "open_weights": license_open(item.get("license")),
                "scores": scores,
            }
        )

    if resolve_hle:
        resolve_hle_no_tools(results)
        results = [r for r in results if r["scores"]]

    results.sort(key=lambda r: r["model"])
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch benchmark scores from the llm-stats.com Open LLM Leaderboard."
    )
    parser.add_argument(
        "--format",
        choices=["json", "table", "names"],
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--names",
        choices=["models", "benchmarks"],
        default="models",
        help="With --format names, list model ids or benchmark labels (default: models).",
    )
    parser.add_argument(
        "--no-hle-detail",
        action="store_true",
        help=(
            "Skip the per-model HLE tool-mode lookup and drop the column instead. "
            "For callers that only need model ids or licences."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = get_scores(resolve_hle=not args.no_hle_detail)

    if args.format == "json":
        print(json.dumps(results, ensure_ascii=False))
    elif args.format == "names":
        if args.names == "models":
            for entry in results:
                print(entry["model"])
        else:
            labels: set[str] = set()
            for entry in results:
                labels.update(entry["scores"].keys())
            for label in sorted(labels):
                print(label)
    else:
        for entry in results:
            title = entry["model"]
            if entry.get("name"):
                title = f"{entry['model']} ({entry['name']})"
            print(f"\n## {title}")
            if not entry["scores"]:
                print("  (no scores)")
                continue
            width = max(len(k) for k in entry["scores"])
            for label, value in sorted(entry["scores"].items()):
                print(f"  {label:<{width}}  {value}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
