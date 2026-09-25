#!/usr/bin/env python3
"""
Fetch GPQA Diamond (%) scores from OpenRouter's model pages.

OpenRouter (https://openrouter.ai/{author}/{model}) runs a small set of
benchmarks itself against every provider endpoint that serves a model, and
shows the results on the model's page. It is a per-model source, like
Spheron: there is no board listing every model's score, so the catalogue
(``/api/v1/models``) supplies the names to map and each mapped model's page is
read on its own.

Only one of those benchmarks is read. The page carries three kinds of number:

  * ``gpqa_diamond``                 OpenRouter's own run, once per provider
                                     endpoint. This is the one we read.
  * ``tau_bench_verified_airline``   OpenRouter's own run too, but tau2-bench
                                     airline is not a column in llm.json.
  * an Artificial Analysis table     A mirror of AA's numbers under OpenRouter's
                                     roof. artificialanalysis.py reads those at
                                     the source, so the copy is ignored.

The scores come from the page's dehydrated React Query cache (the
``self.__next_f`` flight payload), under the query keyed
``["model-page", "benchmarkScores", {"permaslug": ...}]``, as 0-1 fractions
with one row per endpoint and one ``auto-routing`` row for OpenRouter's own
router.

One model's GPQA is many runs of the same 198 questions on different
deployments -- different quantizations, different inference stacks, and now
and then a broken endpoint that scores 0. The reported score is the **median
of the endpoint runs**: the router row is not an endpoint and is left out, and
a run of exactly 0 is a failed run on a four-option test, not a measurement.
The median is what keeps one broken or one unusually good deployment from
deciding the column, which a best-run rule (the one the leaderboards use,
where every row is the model's own run) would hand to the luckiest endpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from _fetch_checks import check_fractions


SITE_URL = "https://openrouter.ai"
# The catalogue: every model id OpenRouter serves, which is what gets mapped.
MODELS_API_URL = f"{SITE_URL}/api/v1/models"
# The human-facing equivalent of the catalogue, for the Sources panel.
MODELS_PAGE_URL = f"{SITE_URL}/models"
MODEL_PAGE_URL = SITE_URL + "/{model}"

BENCHMARK = "gpqa_diamond"
# OpenRouter's own router: a run through whichever endpoint it picked, so a
# repeat of one of the endpoint rows rather than a deployment of its own.
ROUTER_PROVIDER = "auto-routing"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Model pages read at once, as in fetch_spheron.py: one page per model.
MAX_WORKERS = 8

_FLIGHT_RE = re.compile(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)')
_QUERY_START_RE = re.compile(r'\{"dehydratedAt":')
_QUERY_KEY = ["model-page", "benchmarkScores"]


def fetch_text(url: str, retries: int = 3, delay: float = 2.0) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            # A 404 is a model OpenRouter no longer serves; asking again cannot help.
            if exc.code < 500 and exc.code != 429 or attempt == retries:
                raise
            time.sleep(delay * attempt)
        except (urllib.error.URLError, OSError):
            if attempt == retries:
                raise
            time.sleep(delay * attempt)
    raise AssertionError("unreachable")


def base_id(model_id: str) -> str:
    """A catalogue id without its variant: "x/y:free" -> "x/y".

    Variants (":free", ":batch", ...) are the same weights under another price,
    and the plain id's page carries the model's scores for all of them.
    """
    return model_id.split(":", 1)[0]


def fetch_model_names() -> list[str]:
    """Every model id in OpenRouter's catalogue, variants folded, sorted."""
    print(f"Fetching {MODELS_API_URL} ...", file=sys.stderr)
    payload = json.loads(fetch_text(MODELS_API_URL))
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise ValueError(
            f"{MODELS_API_URL} did not return a data list -- the API changed; "
            "refusing to report an empty catalogue."
        )
    names = sorted(
        {
            base_id(entry["id"])
            for entry in data
            if isinstance(entry, dict) and isinstance(entry.get("id"), str) and entry["id"]
        }
    )
    if not names:
        raise ValueError(f"No model ids in {MODELS_API_URL} -- refusing to report none.")
    return names


def flight_text(page_html: str) -> str:
    """The page's React flight payload, decoded and joined."""
    return "".join(json.loads(chunk) for chunk in _FLIGHT_RE.findall(page_html))


def benchmark_query(page_html: str) -> dict | None:
    """The dehydrated benchmarkScores query, or None when the page has none."""
    text = flight_text(page_html)
    decoder = json.JSONDecoder()
    for match in _QUERY_START_RE.finditer(text):
        try:
            query, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        key = query.get("queryKey") if isinstance(query, dict) else None
        if isinstance(key, list) and key[:2] == _QUERY_KEY:
            return query
    return None


def endpoint_runs(query: dict, benchmark: str = BENCHMARK) -> list[float]:
    """One 0-1 score per endpoint run of benchmark, router row and failed runs dropped."""
    data = (query.get("state") or {}).get("data") or {}
    rows = data.get("scores") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError("benchmarkScores query carries no scores list -- the layout changed.")
    runs: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("benchmark_type") != benchmark:
            continue
        if row.get("provider_name") == ROUTER_PROVIDER:
            continue
        endpoint = row.get("endpoint_id")
        score = row.get("score")
        if not isinstance(endpoint, str) or not endpoint:
            continue
        if not isinstance(score, (int, float)) or isinstance(score, bool) or score == 0:
            continue
        runs[endpoint] = float(score)
    return list(runs.values())


def parse_model(page_html: str, model: str) -> dict:
    """One row for a model page: the median endpoint run, or None without one."""
    query = benchmark_query(page_html)
    if query is None:
        raise ValueError(
            f"No benchmarkScores query on {MODEL_PAGE_URL.format(model=model)} -- "
            "the page layout changed; refusing to read it as a model without scores."
        )
    runs = endpoint_runs(query)
    check_fractions(runs, MODEL_PAGE_URL.format(model=model), field=BENCHMARK)
    permaslug = ((query.get("queryKey") or [None, None, {}])[2] or {}).get("permaslug")
    return {
        "model": model,
        "permaslug": permaslug,
        "score": round(statistics.median(runs) * 100, 2) if runs else None,
        "runs": len(runs),
        "low": round(min(runs) * 100, 2) if runs else None,
        "high": round(max(runs) * 100, 2) if runs else None,
        "source": MODEL_PAGE_URL.format(model=model),
    }


def get_scores(models: list[str]) -> list[dict]:
    """Return one row per requested model id.

    Keys: model (the requested catalogue id), permaslug (OpenRouter's dated
    id), score (median GPQA Diamond % over the endpoint runs, None without
    one), runs (endpoint runs the median is over), low/high (their range),
    source (the model page).
    """
    def fetch_one(model: str) -> dict | None:
        url = MODEL_PAGE_URL.format(model=model)
        print(f"Fetching {url} ...", file=sys.stderr)
        try:
            return parse_model(fetch_text(url), model)
        except Exception as exc:  # noqa: BLE001 - keep going across models
            print(f"  failed for {model}: {exc}", file=sys.stderr)
            return None

    wanted = list(dict.fromkeys(base_id(model) for model in models))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        rows = list(pool.map(fetch_one, wanted))
    results = [row for row in rows if row is not None]
    # One model failing is that model's page; every model failing is the
    # site's layout, and must not read as "no scores today".
    if wanted and not results:
        raise ValueError(
            f"Every one of {len(wanted)} OpenRouter page(s) failed to parse -- the "
            "page layout changed; refusing to report no scores."
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch GPQA Diamond scores from OpenRouter's model pages."
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="AUTHOR/NAME",
        help="OpenRouter model id to read (repeatable), e.g. openai/gpt-6-sol. "
        "Required unless --format names.",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "names"],
        default="table",
        help="Output format (default: table). names lists the whole catalogue.",
    )
    args = parser.parse_args()
    if args.format != "names" and not args.model:
        parser.error("--model is required unless --format names")
    return args


def main() -> int:
    args = parse_args()
    if args.format == "names":
        for name in fetch_model_names():
            print(name)
        return 0

    scores = get_scores(args.model)
    if args.format == "json":
        print(json.dumps(scores, ensure_ascii=False))
    else:
        width = max([len("MODEL"), *(len(e["model"]) for e in scores)])
        fmt = f"{{:<{width}}}  {{:>6}}  {{:>4}}  {{}}"
        print(fmt.format("MODEL", "GPQA", "RUNS", "RANGE"))

        def cell(value: object) -> str:
            return "-" if value is None else str(value)

        for entry in scores:
            span = f"{entry['low']}-{entry['high']}" if entry["runs"] else ""
            print(fmt.format(entry["model"], cell(entry["score"]), entry["runs"], span))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
    except (urllib.error.URLError, OSError) as exc:
        print(f"error: could not fetch OpenRouter: {exc}", file=sys.stderr)
        raise SystemExit(1)
