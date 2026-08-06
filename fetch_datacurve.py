#!/usr/bin/env python3
"""
Fetch DeepSWE scores from Datacurve's own leaderboard, https://deepswe.datacurve.ai/

The repo already reads DeepSWE numbers off benchlm.ai (fetch_deepswe.py), which
republishes them; this reads the benchmark's own site, so it is the source that
decides when the two disagree.

The page ships no table -- it hydrates from a versioned JSON artifact
("/artifacts/v1.1/leaderboard-live.json") whose path is embedded in the page. The
path is discovered from the HTML rather than hardcoded, so a bump to v1.2 keeps
working; DEFAULT_ARTIFACT_PATH is the fallback when the page cannot be read.

Every row is one *configuration* -- harness + model + reasoning effort -- with
pass@1 (attempt pass rate over scored rollouts, the number the leaderboard
ranks by and what llm.json stores), pass@4, a run-to-run confidence interval and
cost/token/duration aggregates. Models are named as slugs already
("glm-5-2"), and a row is labelled "<model>[<effort>]" -- the same spelling
benchlm.ai uses, so both DeepSWE sources share one name mapping.

Only the best-scoring configuration per model is reported by default;
--all-configs emits every row instead.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from urllib.parse import urljoin

SITE_URL = "https://deepswe.datacurve.ai/"
# Fallback when the page cannot be read; the live path is discovered from it.
DEFAULT_ARTIFACT_PATH = "/artifacts/v1.1/leaderboard-live.json"
# What this scraper requests, for the fill_source_urls.py inventory.
URL = urljoin(SITE_URL, DEFAULT_ARTIFACT_PATH)

_ARTIFACT_RE = re.compile(r"/artifacts/[^\"'\\\s]*leaderboard-live\.json")
_VERSION_NUM_RE = re.compile(r"\d+")

# Metric read off each row. pass@1 is what the leaderboard ranks by and what
# llm.json stores; pass@4 is offered for comparison.
METRICS = {"pass@1": "pass_at_1", "pass@4": "pass_at_4"}
DEFAULT_METRIC = "pass@1"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def artifact_version(path: str) -> tuple[int, ...]:
    """Sort key ordering artifact paths by the version numbers in them."""
    return tuple(int(n) for n in _VERSION_NUM_RE.findall(path)) or (-1,)


def discover_artifact_url() -> str:
    """The artifact URL the live page points at, or the pinned fallback.

    Several paths can appear in one page (a preloaded query plus a router
    manifest); the highest version wins. A page that cannot be fetched or parsed
    falls back to DEFAULT_ARTIFACT_PATH with a warning rather than failing, so a
    site redesign degrades to the last known-good path instead of losing the
    source entirely.
    """
    try:
        html = fetch_text(SITE_URL)
    except (urllib.error.URLError, OSError) as exc:
        print(f"  warning: {SITE_URL} unreachable ({exc}); using {DEFAULT_ARTIFACT_PATH}",
              file=sys.stderr)
        return URL

    paths = sorted(set(_ARTIFACT_RE.findall(html)), key=artifact_version)
    if not paths:
        print(f"  warning: no artifact path in {SITE_URL}; using {DEFAULT_ARTIFACT_PATH}",
              file=sys.stderr)
        return URL
    return urljoin(SITE_URL, paths[-1])


def config_label(row: dict) -> str | None:
    """"<model>[<effort>]", or "<model>" when the model has no effort setting.

    Matches how benchlm.ai spells the same configuration, which is what lets
    both DeepSWE sources share model-name-mapping-deepswe-to-artificialanalysis.json.
    """
    model = row.get("model")
    if not isinstance(model, str) or not model:
        return None
    effort = row.get("reasoning_effort")
    return f"{model}[{effort}]" if isinstance(effort, str) and effort else model


def parse_rows(payload: dict, metric: str, all_configs: bool) -> list[dict]:
    """Score rows from the artifact, best configuration per model by default."""
    field = METRICS.get(metric, metric)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Artifact has no 'rows' array")

    parsed: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = config_label(row)
        value = row.get(field)
        if label is None or not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        parsed.append(
            {
                "model": label,
                "base_model": row.get("model"),
                "harness": row.get("harness"),
                "reasoning_effort": row.get("reasoning_effort"),
                "score": round(float(value) * 100, 2),
                "pass_at_1": row.get("pass_at_1"),
                "pass_at_4": row.get("pass_at_4"),
                "n_runs": row.get("n_runs"),
                "ci_lo": row.get("ci_lo"),
                "ci_hi": row.get("ci_hi"),
                "n_tasks_attempted": row.get("n_tasks_attempted"),
                "mean_cost_usd": row.get("mean_cost_usd"),
            }
        )

    if not all_configs:
        best: dict[str, dict] = {}
        for row in parsed:
            key = row["base_model"] or row["model"]
            if key not in best or row["score"] > best[key]["score"]:
                best[key] = row
        parsed = list(best.values())

    parsed.sort(key=lambda r: -r["score"])
    for i, r in enumerate(parsed, 1):
        r["rank"] = i
    return parsed


def get_scores(
    metric: str = DEFAULT_METRIC,
    all_configs: bool = False,
    artifact_url: str | None = None,
) -> list[dict]:
    """Return one dict per configuration: model, score, harness, effort, ..."""
    url = artifact_url or discover_artifact_url()
    print(f"Fetching {url} ...", file=sys.stderr)
    payload = json.loads(fetch_text(url))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected payload from {url}: expected an object")

    scores = parse_rows(payload, metric, all_configs)
    scope = "configuration(s)" if all_configs else "model(s), best configuration each"
    print(
        f"  parsed {len(scores)} {scope} ({metric}, "
        f"{payload.get('n_tasks_in_set', '?')} tasks, generated "
        f"{payload.get('generated_at', '?')})",
        file=sys.stderr,
    )
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch DeepSWE scores from Datacurve's leaderboard."
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRICS),
        default=DEFAULT_METRIC,
        help=f"Metric to report as score (default: {DEFAULT_METRIC}, what the board ranks by).",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Report every harness/effort configuration instead of the best per model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = get_scores(metric=args.metric, all_configs=args.all_configs)

    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    elif args.format == "names":
        for entry in scores:
            print(entry["model"])
    else:
        model_width = max([len("MODEL"), *(len(e["model"]) for e in scores)])
        harness_width = max([len("HARNESS"), *(len(e.get("harness") or "") for e in scores)])
        fmt = f"{{:<{model_width}}}  {{:>6}}  {{:>6}}  {{:<{harness_width}}}  {{:>4}}"
        print(fmt.format("MODEL", "SCORE", "PASS@4", "HARNESS", "RUNS"))
        for entry in scores:
            pass_at_4 = entry.get("pass_at_4")
            print(
                fmt.format(
                    entry["model"],
                    f"{entry['score']:.2f}",
                    f"{pass_at_4 * 100:.2f}" if isinstance(pass_at_4, (int, float)) else "",
                    entry.get("harness") or "",
                    entry.get("n_runs") if entry.get("n_runs") is not None else "",
                )
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
