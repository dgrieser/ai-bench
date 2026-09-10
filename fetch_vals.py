#!/usr/bin/env python3
"""
Fetch benchmark scores from Vals AI (vals.ai).

Vals AI is an independent evaluator: it runs the models itself, on its own
harness, and publishes accuracy alongside a standard error, a latency and a
cost per test for every cell. That is the property this ingest is here for --
the columns below already have a first-party publisher, and what Vals adds is a
second, uniform run of the same field.

Each leaderboard lives at https://www.vals.ai/benchmarks/<slug> as an Astro
page. The table is client-rendered, but the whole payload is already in the
document: Astro serializes an island's props into a `props` attribute on its
`<astro-island>` element, and the BenchmarkView island carries the complete
score matrix. So one GET per benchmark is enough, with no HTML table parsing --
`deserialize()` below undoes Astro's `[type, value]` encoding and the rest is
plain JSON.

Under `benchmarkView.default` sit:

  metadata   the benchmark's identity: `slug`, `benchmark` (display name),
             `version`, `updated` (the day the board last moved) and `tasks`
             (the task ids the board is split into)
  tasks      task id -> {model key -> {accuracy, stderr, cost_per_test, ...}}

Model keys are `<provider>/<model>` paths ("zai/glm-5.3-flash"), which is what
this scraper reports and what the name mapping is keyed on: Vals publishes no
display name in the payload, and the path is unique and stable where a rendered
label is neither. `model_label()` gives the model half, which is the part that
resembles a model name -- the openness index and the review candidates are
matched on it.

BENCHMARKS maps the Vals slug to our llm.json benchmark key. TASKS overrides
which task within a board is read, for the one board whose "overall" is not the
column we track: Vals' AIME leaderboard pools the 2024 and 2025 exams, while
llm.json keeps a column per exam year, so `aime_2025` is read instead.

Boards Vals runs that llm.json has no column for (its private industry suites --
Finance Agent, LegalBench, MedQA, the Vals Index -- plus MATH 500 and MGSM) are
deliberately absent; add an entry to BENCHMARKS to ingest one.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any


BASE_URL = "https://www.vals.ai/benchmarks/{slug}"

# Vals AI benchmark slug -> llm.json benchmark key.
BENCHMARKS: dict[str, str] = {
    # Vals' board is titled "SWE-bench Verified" and its metadata slug is the
    # bare "swebench"; the numbers are the Verified split, which is the only
    # SWE-bench column llm.json keeps.
    "swebench": "swe_bench_verified",
    # Two revisions of Terminal-Bench, each on its own page and each stamping
    # its version into the metadata ("2.1" / "2.0"), so neither can be mistaken
    # for the other -- the version trap that keeps SWE-Marathon out of the
    # evals.report ingest does not apply here. tbench.ai owns both columns and
    # outranks this source, so these fill gaps rather than displace the board.
    "terminal-bench-2-1": "terminal_bench_2_1",
    "terminal-bench-2": "terminal_bench_2_0",
    # Vals' own implementation of LiveCodeBench, run over the same field as the
    # rest of its academic boards.
    "lcb": "livecodebench",
    "gpqa": "gpqa_diamond",
    "mmlu_pro": "mmlu_pro",
    # Titled "MMMU Pro" on the site; the metadata slug is the bare "mmmu".
    "mmmu": "mmmu_pro",
    # Read through TASKS: see below.
    "aime": "aime_2025",
}

# The task each board is read at. "overall" everywhere except where the board
# pools several test sets that llm.json splits into columns of their own.
DEFAULT_TASK = "overall"
TASKS: dict[str, str] = {
    # Vals' AIME board carries aime_2024 and aime_2025 side by side and its
    # "overall" is the two exams pooled. llm.json has a column per exam year,
    # and pooling two years into one of them would be the blend those columns
    # exist to prevent, so the 2025 task is read and the 2024 half dropped.
    # The board is archived upstream (last moved 2026-04-16), so it fills the
    # models it measured and never moves again.
    "aime": "aime_2025",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# One <astro-island> element, with the component it hydrates and its props.
_ISLAND_RE = re.compile(r"<astro-island\b([^>]*)>")
_ATTR_RE = re.compile(r'([a-zA-Z-]+)="([^"]*)"')

# The island holding the score matrix, matched on the component-url filename:
# Astro content-hashes the bundle ("BenchmarkView.Bv3BoWAp.js"), so the hash
# cannot be part of the match.
BENCHMARK_VIEW_COMPONENT = "BenchmarkView"


def benchmark_url(slug: str) -> str:
    return BASE_URL.format(slug=slug)


def task_of(slug: str) -> str:
    return TASKS.get(slug, DEFAULT_TASK)


def model_label(model_key: str) -> str:
    """The model half of a "<provider>/<model>" key.

    Vals namespaces every model by the provider it called ("fireworks/gpt-oss-120b",
    "openai/gpt-5.6-sol"), which is what makes the key unique but is not part of
    the model's name anywhere else. Everything that compares a Vals name against
    another source's -- the openness index, the review candidates -- compares
    this.
    """
    return model_key.rsplit("/", 1)[-1]


def fetch_html(url: str, retries: int = 3, delay: float = 2.0) -> str:
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


def deserialize(node: Any) -> Any:
    """Undo Astro's island-props encoding.

    Astro writes every value as a ``[type, value]`` pair: 0 is the value itself
    (recursing through an object's own values), 1 an array whose elements are
    themselves encoded. The rest of the type table (RegExp, Date, Map, Set,
    BigInt, URL, typed arrays) does not appear anywhere in these payloads, so an
    unknown type yields its raw value rather than raising -- a Date appearing
    someday should not take the whole ingest down.
    """
    if not (isinstance(node, list) and len(node) == 2 and isinstance(node[0], int)):
        return node
    kind, value = node
    if kind == 0:
        if isinstance(value, dict):
            return {key: deserialize(item) for key, item in value.items()}
        return value
    if kind == 1 and isinstance(value, list):
        return [deserialize(item) for item in value]
    return value


def island_props(page_html: str, component: str) -> dict[str, Any] | None:
    """Props of the first island hydrating `component`, decoded."""
    for match in _ISLAND_RE.finditer(page_html):
        attrs = dict(_ATTR_RE.findall(match.group(1)))
        if component not in attrs.get("component-url", ""):
            continue
        raw = json.loads(html.unescape(attrs.get("props", "{}")))
        if not isinstance(raw, dict):
            return None
        return {key: deserialize(value) for key, value in raw.items()}
    return None


def parse_board(page_html: str, slug: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(metadata, {model key -> cell}) for one benchmark page.

    Raises ValueError when the page is not the board that was asked for, which
    is what a silent redirect to a renamed slug looks like.
    """
    props = island_props(page_html, BENCHMARK_VIEW_COMPONENT)
    if not props:
        raise ValueError(f"no {BENCHMARK_VIEW_COMPONENT} island on the {slug} page")
    view = props.get("benchmarkView")
    if not isinstance(view, dict) or not isinstance(view.get("default"), dict):
        raise ValueError(f"unexpected benchmarkView shape for {slug}")

    board = view["default"]
    metadata = board.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"no metadata for {slug}")
    if metadata.get("slug") != slug:
        raise ValueError(
            f"asked for {slug} but the page reports {metadata.get('slug')!r}"
        )

    task = task_of(slug)
    tasks = board.get("tasks")
    if not isinstance(tasks, dict) or not isinstance(tasks.get(task), dict):
        raise ValueError(f"{slug} has no {task!r} task")
    return metadata, tasks[task]


def get_scores(slugs: list[str]) -> list[dict]:
    """Return a flat list of score dicts across the requested benchmarks.

    Keys: benchmark (Vals slug), key (llm.json benchmark key), task, model (the
    "<provider>/<model>" key), label (the model half), provider, score, stderr,
    cost_per_test, date (the day the board last moved), rank (within the
    benchmark, 1 = best).
    """
    results: list[dict] = []
    for slug in slugs:
        key = BENCHMARKS[slug]
        url = benchmark_url(slug)
        print(f"Fetching {url} ...", file=sys.stderr)
        metadata, cells = parse_board(fetch_html(url), slug)
        updated = metadata.get("updated") or None

        rows: list[dict] = []
        for model_key, cell in cells.items():
            if not isinstance(model_key, str) or not isinstance(cell, dict):
                continue
            accuracy = cell.get("accuracy")
            if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
                continue
            stderr = cell.get("stderr")
            if isinstance(stderr, bool) or not isinstance(stderr, (int, float)):
                stderr = None
            cost = cell.get("cost_per_test")
            if isinstance(cost, bool) or not isinstance(cost, (int, float)):
                cost = None
            rows.append(
                {
                    "benchmark": slug,
                    "key": key,
                    "task": task_of(slug),
                    "model": model_key,
                    "label": model_label(model_key),
                    "provider": cell.get("provider"),
                    "score": round(float(accuracy), 2),
                    "stderr": round(float(stderr), 3) if stderr is not None else None,
                    "cost_per_test": cost,
                    "date": updated,
                }
            )
        print(f"  parsed {len(rows)} rows for {slug}", file=sys.stderr)
        rows.sort(key=lambda row: -row["score"])
        for position, row in enumerate(rows, 1):
            row["rank"] = position
        results.extend(rows)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Vals AI benchmark scores.")
    parser.add_argument(
        "--benchmark",
        choices=[*BENCHMARKS.keys(), "all"],
        default="all",
        help="Which benchmark to fetch (default: all).",
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
    slugs = list(BENCHMARKS.keys()) if args.benchmark == "all" else [args.benchmark]
    scores = get_scores(slugs)

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for name in sorted({entry["model"] for entry in scores}):
            print(name)
    else:
        widths = [
            max(len("MODEL"), max((len(e["model"]) for e in scores), default=0)),
            6,
            max(len("BENCHMARK"), max((len(e["benchmark"]) for e in scores), default=0)),
        ]
        fmt = f"{{:<{widths[0]}}}  {{:>{widths[1]}}}  {{:<{widths[2]}}}"
        print(fmt.format("MODEL", "SCORE", "BENCHMARK"))
        for entry in scores:
            print(fmt.format(entry["model"], str(entry["score"]), entry["benchmark"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch vals.ai: {exc}", file=sys.stderr)
        raise SystemExit(1)
