#!/usr/bin/env python3
"""Ask a person for the provenance llm.json is missing: dates and source URLs.

Every score carries the date it was read (``scores_updated``) and the page it
was read from (``scores_source``); every VRAM figure carries its own page
(``vram_source``). The scrapers stamp both as they write, so a gap only appears
where no scraper was involved -- a score typed in by hand (add.py, edit.py), a
value whose leaderboard row has since moved, a VRAM figure taken from a
calculator by hand -- plus the model- and benchmark-level URLs a hand-edited
entry can be left without.

sync_score_dates.py reports those gaps but cannot invent a value, and
`update.py --fill-source-urls` only attributes a score whose freshly fetched
value still matches. This script asks instead, one gap at a time, offering as
candidates the URLs and dates the rest of the file already uses for the same
benchmark, model or creator, so the common answer is a single keystroke.

Default is a dry-run; pass -w/--write to persist changes (same convention as
update.py and prune.py).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

import _prompts
import fetch_spheron
from _scores import editable_benchmarks, stamp_score_source, stamp_score_updated
from _spheron_mapping import hf_path_from_url

DEFAULT_LLM_JSON = Path(__file__).resolve().parent / "llm.json"
JSON_DUMP_KWARGS = {"indent": 2, "ensure_ascii": False}

# Gap kinds, in the order they are asked. Model-level URLs first so one answered
# there can be offered as a candidate for the scores below it. A model's own
# date_added is add.py's business, not this script's.
KINDS = (
    "model-url",
    "creator-url",
    "score-date",
    "score-source",
    "vram-source",
    "benchmark-urls",
)

# Candidates offered per gap. Enough to cover the usual answers without turning
# the prompt into a page to read.
MAX_CANDIDATES = 6


class Quit(Exception):
    """The user asked to stop; everything answered so far is kept."""


@dataclass
class Gap:
    """One missing value, with how to fill it and what to suggest."""

    kind: str
    label: str
    detail: str
    value_kind: str  # "url" or "date"
    apply: Callable[[str], None]
    candidates: list[tuple[str, str]] = field(default_factory=list)


def is_missing(value: Any) -> bool:
    """True for null and for a string that is empty or only whitespace."""
    return not (isinstance(value, str) and value.strip())


def parse_url(raw: str) -> str:
    text = raw.strip()
    if not text.startswith(("http://", "https://")):
        raise ValueError("expected a URL starting with http:// or https://")
    return text


def parse_date(raw: str) -> str:
    text = raw.strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        raise ValueError("expected a date like 2026-08-06") from None


PARSERS = {"url": parse_url, "date": parse_date}


def dedupe(candidates: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop repeated values, keeping the first (highest-ranked) reason for each."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for value, why in candidates:
        if is_missing(value) or value in seen:
            continue
        seen.add(value)
        out.append((value, why))
    return out[:MAX_CANDIDATES]


def top(counter: Counter[str], limit: int) -> list[str]:
    return [value for value, _ in counter.most_common(limit)]


def benchmark_urls(entry: dict[str, Any]) -> list[str]:
    """Existing URLs of one benchmark entry, across both supported shapes."""
    urls = entry.get("urls")
    if isinstance(urls, list):
        return [u for u in urls if isinstance(u, str) and u.strip()]
    url = entry.get("url")
    return [url] if isinstance(url, str) and url.strip() else []


@dataclass
class Corpus:
    """What the rest of llm.json already says, for ranking candidates."""

    source_by_benchmark: dict[str, Counter[str]]
    date_by_benchmark: dict[str, Counter[str]]
    url_by_creator: dict[str, Counter[str]]
    owner_by_url: dict[str, str]

    @classmethod
    def build(cls, doc: dict[str, Any]) -> "Corpus":
        sources: dict[str, Counter[str]] = {}
        dates: dict[str, Counter[str]] = {}
        creators: dict[str, Counter[str]] = {}
        owners: dict[str, str] = {}
        for model in doc.get("models", []):
            name = model.get("name")
            urls: list[str] = []
            for key, url in (model.get("scores_source") or {}).items():
                if not is_missing(url):
                    sources.setdefault(key, Counter())[url] += 1
                    urls.append(url)
            for key, when in (model.get("scores_updated") or {}).items():
                if not is_missing(when):
                    dates.setdefault(key, Counter())[when] += 1
            creator = model.get("creator") or {}
            creator_name, creator_url = creator.get("name"), creator.get("url")
            if not is_missing(creator_name) and not is_missing(creator_url):
                creators.setdefault(creator_name, Counter())[creator_url] += 1
            if not is_missing(name):
                page = model.get("url")
                if not is_missing(page):
                    urls.append(page)
                for url in urls:
                    if names_a_model(url, model):
                        owners[url] = name
        return cls(sources, dates, creators, owners)

    def common_sources(self, key: str, model: dict[str, Any], limit: int) -> list[str]:
        """The URLs other models cite most for one benchmark, own pages excluded."""
        name = model.get("name")
        counts = self.source_by_benchmark.get(key, Counter())
        return [
            url
            for url, _ in counts.most_common()
            if self.owner_by_url.get(url, name) == name
        ][:limit]


def mentions(url: str, needle: str) -> bool:
    """True when needle appears in the URL as a whole token.

    Bare substring matching would read the "b" of a model named "b" out of
    "leaderboard"; a per-model page carries the name delimited by / . - _ or the
    end of the URL.
    """
    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(needle.lower())}(?![a-z0-9])")
    return bool(pattern.search(url.lower()))


def names_a_model(url: str, model: dict[str, Any]) -> bool:
    """True when the URL is a page about this model rather than a leaderboard.

    A per-model page -- a HuggingFace card, an Artificial Analysis model page --
    publishes only that model's numbers, so it must never be suggested for
    another model just because it is the most-cited source for the benchmark.
    Recognised by the model's own name or HuggingFace path appearing in the URL,
    which is what a per-model page has and a leaderboard does not.
    """
    if url == model.get("url"):
        return True
    needles = [model.get("name"), hf_path_from_url(model.get("url"))]
    return any(mentions(url, n) for n in needles if isinstance(n, str) and n.strip())


def model_source_counts(model: dict[str, Any]) -> Counter[str]:
    """Source URLs this model already uses, by how many scores cite them."""
    counts: Counter[str] = Counter()
    for url in (model.get("scores_source") or {}).values():
        if not is_missing(url):
            counts[url] += 1
    return counts


def spheron_url(model: dict[str, Any]) -> str | None:
    """The GPU recommender page for this model, from its HuggingFace URL."""
    path = hf_path_from_url(model.get("url"))
    return fetch_spheron.BASE_URL.format(model=path) if path else None


def model_gaps(doc: dict[str, Any], model: dict[str, Any], corpus: Corpus) -> list[Gap]:
    """Every missing date and URL on one model."""
    name = model.get("name", "?")
    gaps: list[Gap] = []

    if is_missing(model.get("url")):
        gaps.append(
            Gap(
                kind="model-url",
                label=name,
                detail="model page URL is missing",
                value_kind="url",
                apply=lambda value, m=model: m.__setitem__("url", value),
            )
        )

    creator = model.get("creator")
    if isinstance(creator, dict) and is_missing(creator.get("url")):
        peers = corpus.url_by_creator.get(creator.get("name") or "", Counter())
        gaps.append(
            Gap(
                kind="creator-url",
                label=f"{name}.creator",
                detail=f"creator {creator.get('name') or '?'} has no URL",
                value_kind="url",
                apply=lambda value, c=creator: c.__setitem__("url", value),
                candidates=dedupe(
                    [(url, "used by other models of this creator") for url in top(peers, 3)]
                ),
            )
        )

    benchmarks = editable_benchmarks(doc)
    scores = model.get("scores") or {}
    updated = model.get("scores_updated") or {}
    sources = model.get("scores_source") or {}
    own_sources = model_source_counts(model)

    for key, score in scores.items():
        if score is None or key not in benchmarks:
            continue
        label = f"{name}.{key}"
        if is_missing(updated.get(key)):
            common = corpus.date_by_benchmark.get(key, Counter())
            gaps.append(
                Gap(
                    kind="score-date",
                    label=label,
                    detail=f"score {score} has no date",
                    value_kind="date",
                    apply=lambda value, m=model, k=key: stamp_score_updated(m, k, when=value),
                    candidates=dedupe(
                        [
                            (date.today().isoformat(), "today"),
                            *((d, f"most common date for {key}") for d in top(common, 2)),
                            (model.get("date_added"), "this model's date_added"),
                        ]
                    ),
                )
            )
        if is_missing(sources.get(key)):
            gaps.append(
                Gap(
                    kind="score-source",
                    label=label,
                    detail=f"score {score} has no source URL",
                    value_kind="url",
                    apply=lambda value, m=model, k=key: stamp_score_source(m, k, value),
                    candidates=dedupe(
                        [
                            *((u, f"{key} benchmark page") for u in benchmark_urls(benchmarks[key])),
                            *((u, "used by other scores of this model") for u in top(own_sources, 2)),
                            *(
                                (u, f"most common source for {key}")
                                for u in corpus.common_sources(key, model, 2)
                            ),
                            (model.get("url"), "this model's page"),
                        ]
                    ),
                )
            )

    vram = model.get("vram")
    if isinstance(vram, dict) and vram:
        vram_sources = model.get("vram_source")
        vram_sources = vram_sources if isinstance(vram_sources, dict) else {}
        own_vram = Counter(u for u in vram_sources.values() if not is_missing(u))
        for quant, size in vram.items():
            if size is None or not is_missing(vram_sources.get(quant)):
                continue
            gaps.append(
                Gap(
                    kind="vram-source",
                    label=f"{name}.vram.{quant}",
                    detail=f"{quant} {size} GB has no source URL",
                    value_kind="url",
                    apply=lambda value, m=model, q=quant: set_vram_source(m, q, value),
                    candidates=dedupe(
                        [
                            *((u, "used by another quant of this model") for u in top(own_vram, 2)),
                            (spheron_url(model), "Spheron GPU recommender for this model"),
                            (model.get("url"), "this model's page"),
                        ]
                    ),
                )
            )

    return gaps


def set_vram_source(model: dict[str, Any], quant: str, url: str) -> None:
    """Record the page one VRAM figure was read from.

    Mirrors _scores.stamp_score_source: a non-dict "vram_source" is an error
    rather than something to overwrite, so corrupt data surfaces loudly instead
    of the attribution being silently dropped.
    """
    sources = model.setdefault("vram_source", {})
    if not isinstance(sources, dict):
        name = model.get("name", "<unknown>")
        raise TypeError(
            f"model {name!r} has a non-dict 'vram_source' "
            f"({type(sources).__name__}); cannot stamp {quant!r}"
        )
    sources[quant] = url


def benchmark_gaps(doc: dict[str, Any]) -> list[Gap]:
    """Benchmark entries with no page linked at all."""
    gaps: list[Gap] = []
    for key, entry in (doc.get("benchmarks") or {}).items():
        if not isinstance(entry, dict) or benchmark_urls(entry):
            continue
        gaps.append(
            Gap(
                kind="benchmark-urls",
                label=f"benchmarks.{key}",
                detail=f"{entry.get('name', key)} has no URL",
                value_kind="url",
                apply=lambda value, e=entry: set_benchmark_url(e, value),
            )
        )
    return gaps


def set_benchmark_url(entry: dict[str, Any], url: str) -> None:
    """Store one benchmark URL in the list shape, dropping the legacy scalar."""
    entry.pop("url", None)
    entry["urls"] = [url]


def collect_gaps(doc: dict[str, Any], names: list[str], kinds: set[str]) -> list[Gap]:
    corpus = Corpus.build(doc)
    gaps: list[Gap] = []
    for model in doc.get("models", []):
        if names and model.get("name") not in names:
            continue
        gaps.extend(model_gaps(doc, model, corpus))
    if not names:
        gaps.extend(benchmark_gaps(doc))
    order = {kind: index for index, kind in enumerate(KINDS)}
    return sorted(
        (gap for gap in gaps if gap.kind in kinds),
        key=lambda gap: (order[gap.kind], gap.label),
    )


def prompt_gap(gap: Gap, position: str) -> str | None:
    """Ask for one value. Returns it, or None when the gap is skipped."""
    parse = PARSERS[gap.value_kind]
    print(f"\n{position} {gap.label}\n  {gap.detail}")
    for index, (value, why) in enumerate(gap.candidates, start=1):
        print(f"  {index}  {value}\n       ({why})")

    hint = "date" if gap.value_kind == "date" else "URL"
    keys = "number, " if gap.candidates else ""
    today = "[t] today, " if gap.value_kind == "date" else ""
    while True:
        try:
            raw = input(f"  {hint}, {keys}{today}[Enter] skip, [q] quit: ")
        except EOFError:
            raise Quit from None

        text = raw.strip()
        if not text:
            return None
        if text.lower() in {"q", "quit"}:
            raise Quit
        if text.lower() in {"t", "today"} and gap.value_kind == "date":
            return date.today().isoformat()
        if text.isdigit() and 1 <= int(text) <= len(gap.candidates):
            return gap.candidates[int(text) - 1][0]
        try:
            return parse(text)
        except ValueError as exc:
            print(f"  invalid: {exc}")


def record_gap(gap: Gap) -> None:
    """Queue one gap as an unanswered question (collect mode)."""
    _prompts.record(
        kind=gap.kind,
        subject=gap.label,
        question=f"{gap.label}: {gap.detail}. Which {gap.value_kind}?",
        candidates=[value for value, _ in gap.candidates],
        command="./fill_missing_source_urls.py",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(DEFAULT_LLM_JSON),
        help='Path to JSON file to read/update (default: "./llm.json" next to this script)',
    )
    parser.add_argument(
        "--model",
        "-m",
        action="append",
        default=[],
        metavar="NAME",
        help="Only ask about this model. Repeat for several; benchmark-level "
        "gaps are skipped when set.",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=KINDS,
        default=[],
        metavar="KIND",
        help=f"Only ask about this kind of gap ({', '.join(KINDS)}). Repeat for several.",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="Report the gaps and exit without asking anything.",
    )
    parser.add_argument(
        "--write",
        "-w",
        action="store_true",
        help="Write changes back to the input JSON file (default is dry-run).",
    )
    _prompts.add_cli_flag(parser)
    args = parser.parse_args()
    _prompts.apply_cli_flag(args)

    path = Path(args.json_file)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc.get("models"), list):
        print(f'Error: "models" in {path} is missing or not a list', file=sys.stderr)
        return 1

    known = {model.get("name") for model in doc["models"]}
    unknown = [name for name in args.model if name not in known]
    if unknown:
        print(f"Error: no such model in {path}: {', '.join(unknown)}", file=sys.stderr)
        return 1

    gaps = collect_gaps(doc, args.model, set(args.only or KINDS))
    if not gaps:
        print("Nothing missing.")
        return 0

    counts = Counter(gap.kind for gap in gaps)
    print(f"{len(gaps)} missing value(s):")
    for kind in KINDS:
        if counts[kind]:
            print(f"  {counts[kind]:4d}  {kind}")
    for gap in gaps:
        print(f"    {gap.label:44s} {gap.detail}")

    if args.list:
        return 0

    if _prompts.collecting():
        for gap in gaps:
            record_gap(gap)
        print(f"\nQueued {len(gaps)} question(s); nothing filled in.")
        return 0

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("\nnot a terminal, nothing asked; run interactively to fill these in")
        return 0

    filled = 0
    try:
        for index, gap in enumerate(gaps, start=1):
            value = prompt_gap(gap, f"[{index}/{len(gaps)}]")
            if value is None:
                continue
            gap.apply(value)
            filled += 1
            print(f"  set {gap.label} = {value}")
    except (Quit, KeyboardInterrupt):
        print("\nstopped; keeping what was answered")

    print(f"\n{filled} filled, {len(gaps) - filled} left")
    if not filled:
        return 0
    if not args.write:
        print("dry-run only, pass --write to persist changes")
        return 0

    path.write_text(json.dumps(doc, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
