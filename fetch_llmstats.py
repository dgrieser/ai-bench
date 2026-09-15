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

``resolve_tool_modes()`` therefore re-reads HLE per model from
``MODEL_URL``, where ``analysis_method`` says what was run, and emits it under
the label ``hle (no tools)``:

  * a note stating both modes ("With tools: 57.4%. Without tools: 43.2%") has
    the without-tools figure parsed out of it and used;
  * a note stating only no-tools keeps the headline score;
  * a note stating tools, or saying nothing about them, drops the score
    entirely. An unverified number is not a no-tools number.

The bare ``hle`` label is never emitted, so it cannot be ingested by accident.
The same pass holds the other no-tools columns llm-stats feeds -- AIME 2025,
GPQA, MMMU-Pro, SciCode -- to a no-tools run, dropping only the entries whose
``analysis_method`` says tools were used; see ``NO_TOOL_FIELDS``.
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

# The other flat fields whose llm.json column is a no-tools column, mapped to
# the detail board their analysis_method lives on. HLE gets the stricter
# treatment above -- verify a no-tools run or publish nothing -- because its
# population really is a coin flip: labs lead with the tools number and 38 of
# 104 entries say so. On these four the no-tools run is the default and tools
# are the exception that gets labelled, so rejecting the labelled exceptions is
# proportionate and keeps the coverage. Today that rejects three AIME 2025
# entries (nemotron-3-nano at 99.2 against a 89.1 no-tools run, and both Sarvam
# models at 96.7 against a card reporting 88.3 "w/ Tools") and one MMMU-Pro
# ("w/ python"). See docs/hle-tool-mode-audit-2026-09.md.
NO_TOOL_FIELDS = {
    "aime_2025": "aime-2025",
    "gpqa": "gpqa-diamond",
    "mmmu_pro": "mmmu-pro",
    "scicode": "scicode",
}
# llm-stats' benchmark_id for the full 2,500-question board, which is what
# hle_score is drawn from. Its sibling "hle-verified" is a different question
# set and never feeds it.
HLE_BENCHMARK_ID = "humanity\u2019s-last-exam".replace("\u2019", "'")
# And the board that is our column exactly: no tools, on the text-only subset,
# which is the question set Artificial Analysis runs and 93% of the column is.
# Where llm-stats has a model on this board its score is preferred outright --
# the flat field is the full multimodal set, so a no-tools headline from it is
# still a different question set, worth 2-3 points on the cards reporting both
# (DeepSeek-V4.1-Flash: 36.8 full against 39.1 text-only).
HLE_TEXT_ONLY_BENCHMARK_ID = HLE_BENCHMARK_ID + "-(no-tools,-text-only)"

# A code interpreter is a tool, so the python spellings belong here too: it is
# what separates MMMU-Pro "w/ python" from the run AA publishes.
_WITH_TOOLS_RE = re.compile(
    r"\bwith(?:\s+|-)(?:tools?|search|browsing|retrieval|python)\b"
    r"|\bw/\s*(?:tools?|python)\b|\bsearch agent\b|\btool[- ]augmented\b"
    r"|\bcode (?:execution|interpreter)\b",
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


def _benchmark_entries(model_id: str, timeout: int = 30) -> dict[str, dict]:
    """benchmark_id -> that board's entry, from one per-model request.

    The flat leaderboard endpoint drops this prose; it is the only place
    llm-stats says how a score was produced. An empty result -- no entry, no
    method, or a failed request -- ends the same way at every call site: the
    score is dropped or kept on its own merits, never guessed at.
    """
    url = MODEL_URL.format(model_id=urllib.parse.quote(model_id, safe=""))
    try:
        payload = fetch_json(url, timeout=timeout)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"warning: {model_id}: detail fetch failed: {exc}", file=sys.stderr)
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict] = {}
    for entry in payload.get("benchmarks") or []:
        if not isinstance(entry, dict):
            continue
        bid = entry.get("benchmark_id")
        if isinstance(bid, str):
            out.setdefault(bid, entry)
    return out


def _method(entries: dict[str, dict], bid: str) -> str | None:
    m = (entries.get(bid) or {}).get("analysis_method")
    return m if isinstance(m, str) and m.strip() else None


def used_tools(method: str | None) -> bool:
    """True when the note says tools were used and does not also deny it.

    A note naming both modes describes two runs; it is handled where the
    headline's mode matters, not here.
    """
    if not isinstance(method, str) or not method.strip():
        return False
    return bool(_WITH_TOOLS_RE.search(method)) and not bool(_NO_TOOLS_RE.search(method))


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


def resolve_tool_modes(results: list[dict], timeout: int = 30) -> None:
    """Hold every no-tools column to a no-tools run. Mutates ``results``.

    HLE is republished as a verified ``hle (no tools)`` or not at all; the other
    no-tools fields keep their label and lose only the entries whose note says
    tools were used. One request per model that carries any gated score, not one
    per model.
    """
    gated = (HLE_LABEL, *NO_TOOL_FIELDS)
    scored = [r for r in results if any(r["scores"].get(f) is not None for f in gated)]
    if scored:
        print(f"Resolving tool mode for {len(scored)} model(s) ...", file=sys.stderr)
    hle_seen = hle_kept = rejected = 0
    for record in results:
        if not any(record["scores"].get(f) is not None for f in gated):
            continue
        entries = _benchmark_entries(record["model"], timeout=timeout)

        headline = record["scores"].pop(HLE_LABEL, None)
        exact = (entries.get(HLE_TEXT_ONLY_BENCHMARK_ID) or {}).get("score")
        if not isinstance(exact, (int, float)) or isinstance(exact, bool):
            exact = None
        # The exact board is worth publishing even where the flat field is null,
        # which costs nothing: these details are already fetched.
        if headline is not None or exact is not None:
            hle_seen += 1
            value = float(exact) if exact is not None else hle_no_tools_score(
                headline, _method(entries, HLE_BENCHMARK_ID)
            )
            if value is not None:
                record["scores"][HLE_NO_TOOLS_LABEL] = value
                hle_kept += 1

        for field, board in NO_TOOL_FIELDS.items():
            if record["scores"].get(field) is None:
                continue
            if used_tools(_method(entries, board)):
                del record["scores"][field]
                rejected += 1
    if scored:
        print(
            f"  HLE: kept {hle_kept} of {hle_seen} as no-tools runs; "
            f"dropped {hle_seen - hle_kept} run with tools or not stated",
            file=sys.stderr,
        )
        print(f"  other no-tools columns: dropped {rejected} run with tools", file=sys.stderr)


def get_scores(resolve_hle: bool = True) -> list[dict]:
    """Return a list of dicts with keys: model, name, license, open_weights, scores.

    ``model`` is the llm-stats model_id, ``scores`` maps the source benchmark
    label (``_score`` suffix stripped) to its raw 0-1 value. ``open_weights`` is
    None when the record carries no licence. Records without any non-null score
    are dropped.

    ``resolve_hle`` re-reads the tool-mode-sensitive columns per model so only
    no-tools runs are published; see the module docstring. It costs one request
    per model carrying a gated score, so callers that only want model ids or
    licences turn it off.
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
        resolve_tool_modes(results)
    else:
        # The gated columns only exist in a publishable form once the tool mode
        # is known, so the cheap path drops them rather than letting an
        # unverified number out by a different door.
        for record in results:
            for field in (HLE_LABEL, *NO_TOOL_FIELDS):
                record["scores"].pop(field, None)
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
            "Skip the per-model tool-mode lookup, dropping the gated columns "
            "instead. For callers that only need model ids or licences."
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
