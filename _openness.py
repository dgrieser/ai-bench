#!/usr/bin/env python3
"""Decide whether a leaderboard model ships open weights.

llm.json only tracks open-weight models, so a source model that is provably
closed can never map to one -- update_*_mapping.py records those without
prompting.

Three of the scraped sources publish the fact directly:

  llm-stats.com / zeroeval   ``license``     "proprietary" vs an open licence
  benchlm.ai (DeepSWE)       ``sourceType``  "Proprietary" / "Open Weight"
  evals.report               an " Open" suffix on the model label

The rest (SWE-Rebench, FrontierSWE, SWE Atlas, SWE-Marathon, OSWorld) publish
only a creator, so their names are resolved against a pooled index built from
the three sources above and keyed by a normalized model name. The index is
cached under ~/.cache/ai-bench/openness.json for CACHE_TTL_SECONDS.

Only a *closed* verdict is ever acted on, and only when the pooled sources
agree: an open model may still be unmappable (a duplicate reasoning variant, a
model llm.json does not track), which stays a human decision. Two guards keep a
mislabelled source from silently dropping a model:

  * names that look like an llm.json model (see is_closed_weights) are never
    auto-recorded, and
  * auto-recorded entries get their own CLOSED_WEIGHTS sentinel, so
    ``--recheck-closed`` can put them back in front of a human.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Iterable

# Sentinel stored for source names a human reviewed and deliberately left
# unmapped. Never used as a real llm.json model slug.
UNMAPPABLE = "__unmappable__"

# Sentinel stored for source names skipped automatically because the source
# says the model has closed weights. Kept distinct from UNMAPPABLE so the
# machine-made decisions can be re-reviewed.
CLOSED_WEIGHTS = "__closed_weights__"

# Mapping values that are markers rather than llm.json model slugs.
SENTINELS = frozenset({UNMAPPABLE, CLOSED_WEIGHTS})

CACHE_PATH = os.path.expanduser("~/.cache/ai-bench/openness.json")
CACHE_TTL_SECONDS = 12 * 3600

# Trailing reasoning-effort / thinking-mode modifiers, in any of the bracket
# styles the leaderboards use ("-high", " [medium]", " (xhigh)", " thinking").
_EFFORT_RE = re.compile(
    r"[-_ \[(]+(?:xhigh|x-high|very-high|high|medium|low|minimal|max"
    r"|no[- ]?thinking|thinking|reasoning)[\])]*\s*$",
    re.IGNORECASE,
)
# Release-date suffixes: "gpt-5.2-2025-12-11" -> "gpt-5.2".
_DATE_RE = re.compile(r"[-_/](?:19|20)\d\d[-_/]\d\d[-_/]\d\d\b")
_PARENS_RE = re.compile(r"\([^)]*\)")

_INDEX: dict[str, bool] | None = None


def normalize(name: str) -> str:
    """Reduce a model name to a comparison key shared across sources.

    Drops harness parentheticals, release dates and reasoning-effort suffixes,
    then flattens every separator to a single space so the ".", "-" and " "
    spellings of a version collapse together:

    "GPT-5.2-2025-12-11-xhigh"       -> "gpt 5 2"
    "Qwen3.5-27B"                    -> "qwen3 5 27b"
    "Opus 4.8 (Claude Code) xHigh"   -> "opus 4 8"
    """
    text = _PARENS_RE.sub(" ", name.strip().lower())
    text = _DATE_RE.sub(" ", text)
    while True:
        stripped = _EFFORT_RE.sub("", text).strip()
        if stripped == text:
            break
        text = stripped
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def license_open(value: Any) -> bool | None:
    """Read llm-stats' ``license`` field: "proprietary" is the only closed value."""
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower() != "proprietary"


def source_type_open(value: Any) -> bool | None:
    """Read a source's weights-availability label.

    Covers benchlm.ai's ``sourceType`` ("Open Weight" / "Proprietary") and
    Toolathlon's model type ("Open-Weights" / "Open-Source" / "Proprietary");
    separators are normalised so both spellings land on the same verdict.

    The field is also left null or "Pending" for models awaiting review, which
    carries no verdict.
    """
    if not isinstance(value, str):
        return None
    text = re.sub(r"[-_]+", " ", value).strip().lower()
    if text in {"open weight", "open weights", "open source", "open"}:
        return True
    if text in {"proprietary", "closed", "closed weight", "closed weights"}:
        return False
    return None


def _pool(verdicts: dict[str, bool], name: str, is_open: bool | None) -> None:
    """Add one verdict to the index, dropping keys the sources disagree on.

    A disagreement means at least one source is wrong about this name, so the
    key is poisoned to open (never auto-skipped) rather than guessed at.
    """
    if is_open is None:
        return
    key = normalize(name)
    if not key:
        return
    previous = verdicts.get(key)
    if previous is None:
        verdicts[key] = is_open
    elif previous is not is_open:
        verdicts[key] = True


def _pool_llmstats(verdicts: dict[str, bool]) -> None:
    import fetch_llmstats

    for entry in fetch_llmstats.get_scores():
        is_open = entry.get("open_weights")
        for name in (entry.get("model"), entry.get("name")):
            if isinstance(name, str) and name:
                _pool(verdicts, name, is_open)


def _pool_deepswe(verdicts: dict[str, bool]) -> None:
    import fetch_deepswe

    for entry in fetch_deepswe.get_scores():
        is_open = entry.get("open_weights")
        for name in (entry.get("model"), entry.get("slug")):
            if isinstance(name, str) and name:
                _pool(verdicts, name, is_open)


def _pool_evals_report(verdicts: dict[str, bool]) -> None:
    import fetch_evals_report

    entries = fetch_evals_report.get_scores(
        list(fetch_evals_report.BENCHMARKS), include_unverified=True
    )
    for entry in entries:
        _pool(verdicts, entry.get("model") or "", entry.get("open_weights"))


def build_index() -> dict[str, bool]:
    """Fetch every source that publishes weight availability and pool it."""
    verdicts: dict[str, bool] = {}
    for label, pool in (
        ("llm-stats", _pool_llmstats),
        ("evals.report", _pool_evals_report),
        ("DeepSWE", _pool_deepswe),
    ):
        try:
            pool(verdicts)
        except Exception as exc:  # noqa: BLE001 - a dead source must not block review
            print(f"  openness: skipping {label} ({exc})", file=sys.stderr)
    return verdicts


def _load_cached() -> dict[str, bool] | None:
    try:
        with open(CACHE_PATH, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if time.time() - payload.get("fetched", 0) > CACHE_TTL_SECONDS:
        return None
    verdicts = payload.get("verdicts")
    if not isinstance(verdicts, dict):
        return None
    return {k: bool(v) for k, v in verdicts.items() if isinstance(k, str)}


def _save_cached(verdicts: dict[str, bool]) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as handle:
            json.dump({"fetched": time.time(), "verdicts": verdicts}, handle)
    except OSError:
        pass


def open_index(refresh: bool = False) -> dict[str, bool]:
    """Normalized model name -> has open weights, pooled across sources."""
    global _INDEX
    if _INDEX is not None and not refresh:
        return _INDEX
    verdicts = None if refresh else _load_cached()
    if verdicts is None:
        verdicts = build_index()
        if verdicts:
            _save_cached(verdicts)
    _INDEX = verdicts
    return _INDEX


def verdict(name: str) -> bool | None:
    """True if the pooled sources call `name` open, False if closed, else None."""
    return open_index().get(normalize(name))


def _resembles_tracked_model(name: str, tracked: Iterable[str]) -> str | None:
    """Return the llm.json name `name` looks like a shorter/longer spelling of.

    llm.json holds only open-weight models, so any resemblance to one outweighs
    a source calling the model closed -- sources do get this wrong (llm-stats
    labels Grok 2 and ERNIE 4.5 proprietary although both shipped weights).
    """
    key = normalize(name)
    if not key:
        return None
    for candidate in tracked:
        other = normalize(candidate)
        if not other:
            continue
        if other == key or other.startswith(key + " ") or key.startswith(other + " "):
            return candidate
    return None


def is_closed_weights(
    name: str,
    *,
    open_weights: bool | None = None,
    guard_names: Iterable[str] = (),
) -> bool:
    """Whether `name` can be skipped as a closed-weight model without asking.

    `open_weights` is the verdict the source published for this row, when it
    publishes one; it wins over the pooled index. `guard_names` are the
    llm.json model names -- a name resembling one of those is never skipped.
    Callers should warm the index with open_index() first, so its fetches do
    not interleave with per-name output.
    """
    if _resembles_tracked_model(name, guard_names):
        return False
    if open_weights is not None:
        return open_weights is False
    return verdict(name) is False


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect the pooled open-weight index used to skip closed models."
    )
    parser.add_argument("name", nargs="*", help="Model names to classify.")
    parser.add_argument(
        "--refresh", action="store_true", help="Rebuild the index, ignoring the cache."
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table).",
    )
    args = parser.parse_args()

    index = open_index(refresh=args.refresh)

    if args.name:
        rows = [(name, index.get(normalize(name))) for name in args.name]
    else:
        rows = sorted((key, value) for key, value in index.items())

    if args.format == "json":
        print(json.dumps({name: value for name, value in rows}, ensure_ascii=False))
        return 0

    closed = sum(1 for _, value in rows if value is False)
    width = max((len(name) for name, _ in rows), default=len("MODEL"))
    print(f"{'MODEL':<{width}}  WEIGHTS")
    for name, value in rows:
        label = {True: "open", False: "closed", None: "unknown"}[value]
        print(f"{name:<{width}}  {label}")
    print(f"\n{len(rows)} names, {closed} closed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
