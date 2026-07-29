#!/usr/bin/env python3
"""
Fetch Resolved Rate (%) scores from https://swe-rebench.com/

The site is a Next.js app that embeds leaderboard data as React Server Component
flight data in a self.__next_f.push() script tag. This script extracts it directly
without needing a headless browser.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from datetime import datetime, timezone


URL = "https://swe-rebench.com/"
# rangeStats language bucket to read; "all" aggregates every language.
DEFAULT_LANGUAGE = "all"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def extract_items(html: str) -> list[dict]:
    """Extract the model items array from the Next.js RSC flight data."""
    idx = html.find("modelId")
    if idx == -1:
        raise ValueError("Could not find model data in page HTML")

    script_start = html.rfind("<script>", 0, idx)
    script_end = html.find("</script>", idx)
    script_content = html[script_start + 8 : script_end].strip()

    prefix = "self.__next_f.push([1,"
    if not script_content.startswith(prefix):
        raise ValueError(f"Unexpected script format: {script_content[:80]!r}")

    raw_json_str = script_content[len(prefix) : -2]  # strip prefix and trailing ])
    decoded = json.loads(raw_json_str)  # unescape the double-encoded JSON string

    items_idx = decoded.find('"items":')
    if items_idx == -1:
        raise ValueError("Could not find 'items' array in decoded data")

    arr_start = decoded.index("[", items_idx)
    depth, end = 0, arr_start
    for i, c in enumerate(decoded[arr_start:], arr_start):
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i
                break

    return json.loads(decoded[arr_start : end + 1])


def parse_range_key(key: str) -> tuple[int, int] | None:
    """
    Parse a rangeStats key of the form "from_ms:to_ms".

    The leaderboard also emits unbounded keys such as "all", which carry no
    time range; those return None so callers can skip them.
    """
    parts = key.split(":")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def range_stats(item: dict, language: str = DEFAULT_LANGUAGE) -> dict:
    """
    Return the {"from_ms:to_ms": stats} mapping for one model.

    The leaderboard buckets rangeStats by language ("all", "python", "go", ...).
    Older payloads stored the window keys directly, so a flat mapping is passed
    through unchanged.
    """
    stats_by_key = item["rangeStats"]
    if any(isinstance(v, dict) and "resolvedRate" in v for v in stats_by_key.values()):
        return stats_by_key

    if language not in stats_by_key:
        raise ValueError(
            f"Unknown language {language!r} for model {item.get('modelName')!r}; "
            f"available: {', '.join(sorted(stats_by_key))}"
        )
    return stats_by_key[language]


def find_current_window(items: list[dict], language: str = DEFAULT_LANGUAGE) -> str:
    """
    Find the rangeStats key for the most recent 1-month window.
    Keys have the format "from_ms:to_ms". We pick the key where
    to - from ≈ 1 month and 'to' is the latest available.
    """
    all_keys: set[str] = set()
    for item in items:
        all_keys.update(range_stats(item, language).keys())

    month_ms_min = 25 * 24 * 3600 * 1000   # 25 days in ms
    month_ms_max = 35 * 24 * 3600 * 1000   # 35 days in ms

    monthly_keys = []
    for k in all_keys:
        parsed = parse_range_key(k)
        if parsed is None:
            continue
        f, t = parsed
        duration = t - f
        if month_ms_min <= duration <= month_ms_max:
            monthly_keys.append((t, k))

    if not monthly_keys:
        raise ValueError("Could not find any 1-month range keys in rangeStats")

    _, best_key = max(monthly_keys)
    return best_key


def find_latest_monthly_window(item: dict, language: str = DEFAULT_LANGUAGE) -> str | None:
    """Return the most recent 1-month rangeStats key with a non-zero score for a model."""
    month_ms_min = 25 * 24 * 3600 * 1000
    month_ms_max = 35 * 24 * 3600 * 1000

    candidates = []
    for k, stats in range_stats(item, language).items():
        parsed = parse_range_key(k)
        if parsed is None:
            continue
        f, t = parsed
        duration = t - f
        if month_ms_min <= duration <= month_ms_max and stats["resolvedRate"] > 0:
            candidates.append((t, k))

    if not candidates:
        return None
    _, best_key = max(candidates)
    return best_key


def get_scores(
    window: str | None = None,
    all_models: bool = False,
    language: str = DEFAULT_LANGUAGE,
) -> list[dict]:
    """
    Returns a sorted list of dicts with keys: rank, model, resolved_rate, sem.

    window=None uses the most recent 1-month window (only models evaluated then).
    all_models=True picks each model's most recent monthly window individually,
    returning all models but from potentially different task sets.
    language selects the leaderboard's per-language bucket ("all" = aggregate).
    """
    print(f"Fetching {URL} ...", file=sys.stderr)
    html = fetch_html(URL)

    print("Parsing leaderboard data...", file=sys.stderr)
    items = extract_items(html)

    if all_models:
        results = []
        for item in items:
            key = find_latest_monthly_window(item, language)
            if key is None:
                continue
            stats = range_stats(item, language)[key]
            f_ts, t_ts = parse_range_key(key)
            results.append(
                {
                    "model": item["modelName"],
                    "resolved_rate": round(stats["resolvedRate"], 2),
                    "sem": round(stats.get("sem", 0), 4),
                    "pass_at_5": round(stats.get("passN", 0), 2),
                    "window_from": datetime.fromtimestamp(f_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                    "window_to": datetime.fromtimestamp(t_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                }
            )
        results.sort(key=lambda x: -x["resolved_rate"])
        for i, r in enumerate(results, 1):
            r["rank"] = i
        return results

    if window is None:
        window = find_current_window(items, language)

    parsed = parse_range_key(window)
    if parsed is None:
        raise ValueError(f"Window must have the format 'from_ms:to_ms', got {window!r}")
    f_ts, t_ts = parsed
    from_date =datetime.fromtimestamp(f_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    to_date = datetime.fromtimestamp(t_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"Window: {from_date} → {to_date}", file=sys.stderr)

    results = []
    for item in items:
        stats = range_stats(item, language).get(window)
        if stats is None:
            continue
        resolved_rate = stats["resolvedRate"]
        if resolved_rate == 0:
            continue
        results.append(
            {
                "model": item["modelName"],
                "resolved_rate": round(resolved_rate, 2),
                "sem": round(stats.get("sem", 0), 4),
                "pass_at_5": round(stats.get("passN", 0), 2),
                "window_from": from_date,
                "window_to": to_date,
            }
        )

    results.sort(key=lambda x: -x["resolved_rate"])
    for i, r in enumerate(results, 1):
        r["rank"] = i

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch SWE-rebench leaderboard scores")
    parser.add_argument(
        "--window",
        help="Override time window as 'from_ms:to_ms' (e.g. 1769904000000:1772323200000)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Include all models using each one's most recent monthly window (different task sets)",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help="Language bucket to read, e.g. python/go/java/rust/typescript "
        f"(default: {DEFAULT_LANGUAGE}, the cross-language aggregate)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv", "names"],
        default="table",
        help="Output format (default: table)",
    )
    args = parser.parse_args()

    scores = get_scores(
        window=args.window, all_models=args.all_models, language=args.language
    )

    if args.format == "json":
        print(json.dumps(scores, indent=2))
    elif args.format == "names":
        # remove duplicates and print all unique names
        names = set(r["model"] for r in scores)
        print("\n".join(sorted(names)))
    elif args.format == "csv":
        print("rank,model,resolved_rate,sem,pass_at_5,window_from,window_to")
        for r in scores:
            print(
                f"{r['rank']},{r['model']!r},{r['resolved_rate']},"
                f"{r['sem']},{r['pass_at_5']},{r['window_from']},{r['window_to']}"
            )

    else:  # table
        mixed_windows = len({(r["window_from"], r["window_to"]) for r in scores}) > 1
        if mixed_windows:
            print(f"\n{'Rank':<5} {'Model':<45} {'Resolved Rate':>14}  {'SEM':>6}  {'Pass@5':>7}  {'Window'}")
            print("-" * 100)
            for r in scores:
                print(
                    f"{r['rank']:<5} {r['model']:<45} {r['resolved_rate']:>13.1f}%"
                    f"  {r['sem']:>6.2f}  {r['pass_at_5']:>6.1f}%  {r['window_from']} → {r['window_to']}"
                )
        else:
            print(f"\n{'Rank':<5} {'Model':<45} {'Resolved Rate':>14}  {'SEM':>6}  {'Pass@5':>7}")
            print("-" * 80)
            for r in scores:
                print(
                    f"{r['rank']:<5} {r['model']:<45} {r['resolved_rate']:>13.1f}%"
                    f"  {r['sem']:>6.2f}  {r['pass_at_5']:>6.1f}%"
                )
            print(f"\n{len(scores)} models, window {scores[0]['window_from']} → {scores[0]['window_to']}")
        if mixed_windows:
            print(f"\n{len(scores)} models (mixed windows — each model's most recent evaluation)")


if __name__ == "__main__":
    main()
