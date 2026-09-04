#!/usr/bin/env python3
"""
Fetch Agents' Last Exam (ALE) scores from https://agents-last-exam.org/

The leaderboard page is a Next.js app that hydrates from a plain JSON endpoint,
so this reads the endpoint directly: one flat list of rows, each a
(split, harness, model, harness variant) measurement.

ALE publishes several views of the same runs and they are NOT interchangeable:

  * ``full/overall``        every public task, the site's default view. This is
                            the one we read.
  * ``full/near-term``      the subset today's agents partially solve
  * ``full/full-spectrum``  one task per sub-industry
  * ``full/last-exam``      the hardest workflows, where pass rates are ~0-3%
  * ``unlicensed/*``        the same cuts restricted to unlicensed tasks
  * ``linux_only``          the ALE-CLI subset

Reading the wrong one would look like every model collapsing or jumping, so the
split is selected by name and a missing one is an error rather than a fallback.

The metric is Pass Rate -- the share of tasks fully passed -- which is what the
site sorts by and what model cards quote. ``avgScore`` (partial credit against
the hidden reference) is carried through for transparency but is not the stored
score. The "Best-per-task" row the site shows on top is synthesised in the
browser from the rows below it and is not in this payload; nothing to filter.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


URL = "https://agents-last-exam.org/api/demo/leaderboard"
# Human-facing page publishing the same data; stored as the per-score source
# URL because the API path is not a page a reader can open.
LEADERBOARD_URL = "https://agents-last-exam.org/leaderboard"

# The site's default view: every public task, license filter off.
SPLIT = "full/overall"

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) ai-bench-fetcher/1.0"


def fetch_json(url: str = URL, timeout: int = 60) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def select_split(payload: object, split: str) -> list[dict]:
    """Rows of one leaderboard view.

    Raises when the split is gone rather than falling back to another one: the
    tiers differ by an order of magnitude in pass rate, so a silent switch would
    be indistinguishable from every model regressing at once.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError(f"Unexpected response from {URL}: expected an object with 'rows'")

    rows = [row for row in payload["rows"] if isinstance(row, dict)]
    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        available = sorted({row.get("split") for row in rows if isinstance(row.get("split"), str)})
        raise ValueError(f"Split {split!r} not found; available: {available}")
    return selected


def get_scores(split: str = SPLIT) -> list[dict]:
    """Return a list of dicts: model, raw, harness, harness_variant, score,
    avg_score, passes, tasks, runs, rank.

    ``score`` is Pass Rate as a percentage, the metric the leaderboard sorts by.
    One entry per (harness, model, variant) row; rank is within this list.
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    rows = select_split(fetch_json(), split)
    print(f"  parsed {len(rows)} rows for split {split!r}", file=sys.stderr)

    results: list[dict] = []
    for row in rows:
        model = row.get("model")
        pass_rate = row.get("passRate")
        if not isinstance(model, str) or not model.strip():
            continue
        if not isinstance(pass_rate, (int, float)) or isinstance(pass_rate, bool):
            continue
        avg_score = row.get("avgScore")
        results.append(
            {
                "model": model.strip(),
                "raw": model.strip(),
                "harness": row.get("harness") or None,
                "harness_variant": row.get("harnessVariant") or None,
                "score": round(float(pass_rate) * 100.0, 2),
                "avg_score": (
                    round(float(avg_score) * 100.0, 2)
                    if isinstance(avg_score, (int, float)) and not isinstance(avg_score, bool)
                    else None
                ),
                "passes": row.get("passes"),
                "tasks": row.get("tasks"),
                "runs": row.get("runs"),
            }
        )

    results.sort(key=lambda r: -r["score"])
    for i, entry in enumerate(results, 1):
        entry["rank"] = i
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Agents' Last Exam leaderboard scores.")
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--split",
        default=SPLIT,
        help=f"Leaderboard view to read (default: {SPLIT}). Other tiers are on a "
        "different scale and are not what llm.json stores.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = get_scores(split=args.split)

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for name in sorted({entry["model"] for entry in scores}):
            print(name)
    else:
        widths = [
            max([len("MODEL")] + [len(e["model"]) for e in scores]),
            8,
            6,
            max([len("HARNESS")] + [len(e["harness"] or "") for e in scores]),
            max([len("VARIANT")] + [len(e["harness_variant"] or "") for e in scores]),
        ]
        fmt = (
            f"{{:<{widths[0]}}}  {{:>{widths[1]}}}  {{:>{widths[2]}}}  "
            f"{{:<{widths[3]}}}  {{:<{widths[4]}}}"
        )
        print(fmt.format("MODEL", "PASSRATE", "SCORE", "HARNESS", "VARIANT"))
        for entry in scores:
            print(
                fmt.format(
                    entry["model"],
                    str(entry["score"]),
                    "" if entry["avg_score"] is None else str(entry["avg_score"]),
                    entry["harness"] or "",
                    entry["harness_variant"] or "",
                )
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch the Agents' Last Exam leaderboard: {exc}", file=sys.stderr)
        raise SystemExit(1)
