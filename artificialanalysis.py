#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse
import json
import os
import re
import sys
from datetime import datetime, date

import argcomplete
import requests
from tabulate import tabulate
import yaml

API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
FORMATS = {"json", "yaml", "md", "text"}
MODEL_PAGE_URL = "https://artificialanalysis.ai/models/{}"
_CONTEXT_CACHE = {}
_CONTEXT_ENABLED = True
_VERBOSE = False
CACHE_PATH = os.path.expanduser("~/.cache/artificialanalysis/models.json")
_CACHE_WARMED = False


def _is_open_source(model: dict):
    for key in ("open_source", "is_open_source", "open"):
        if key in model:
            return bool(model.get(key))
    return None


def _parse_release_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


def _format_headers(labels, style):
    headers = []
    for label in labels:
        if style == "text":
            headers.append(f"\x1b[1m{label.upper()}\x1b[0m")
        else:
            headers.append(label)
    return headers


def _extract_creator(m: dict) -> str:
    if isinstance(m.get("model_creator"), dict):
        return m["model_creator"].get("slug") or m["model_creator"].get("name", "")
    return ""


def _fetch_context_window(slug: str):
    if not _CONTEXT_ENABLED:
        return ""
    if not slug:
        return ""
    if slug in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[slug]
    try:
        if _VERBOSE:
            url = MODEL_PAGE_URL.format(slug)
            print(f"> GET {url}", file=sys.stderr)
        resp = requests.get(MODEL_PAGE_URL.format(slug), timeout=15)
        if _VERBOSE:
            print(f"< {resp.status_code} {url}", file=sys.stderr)
        if resp.status_code != 200:
            _CONTEXT_CACHE[slug] = ""
            return ""
        m = re.search(r"Context window.+?<span[^>]*>([0-9]+[kmb])", resp.text, re.IGNORECASE)
        if not m:
            _CONTEXT_CACHE[slug] = ""
            return ""
        _CONTEXT_CACHE[slug] = m.group(1)
        return _CONTEXT_CACHE[slug]
    except requests.RequestException:
        _CONTEXT_CACHE[slug] = ""
        return ""


def _extract_context_window(m: dict):
    return _fetch_context_window(m.get("slug", ""))


def _load_cache():
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(models):
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        creators = set()
        slugs = []
        for m in models:
            slug = m.get("slug")
            if slug:
                slugs.append(slug)
            if isinstance(m.get("model_creator"), dict):
                cslug = m["model_creator"].get("slug")
                if cslug:
                    creators.add(cslug)
        payload = {
            "slugs": sorted(set(slugs)),
            "creators": sorted(creators),
        }
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError:
        pass


def _ensure_cache():
    global _CACHE_WARMED
    if _CACHE_WARMED:
        return _load_cache()
    cache = _load_cache()
    if cache.get("slugs") or cache.get("creators"):
        _CACHE_WARMED = True
        return cache
    api_key = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY")
    if not api_key:
        _CACHE_WARMED = True
        return cache
    try:
        if _VERBOSE:
            print(f"> GET {API_URL}", file=sys.stderr)
        resp = requests.get(API_URL, headers={"x-api-key": api_key}, timeout=15)
        if _VERBOSE:
            print(f"< {resp.status_code} {API_URL}", file=sys.stderr)
        if resp.status_code != 200:
            _CACHE_WARMED = True
            return cache
        data = resp.json()
        models = data.get("data", [])
        _save_cache(models)
    except requests.RequestException:
        pass
    _CACHE_WARMED = True
    return _load_cache()


def _model_completer(prefix, _parsed_args=None, **_kwargs):
    cache = _ensure_cache()
    for slug in cache.get("slugs", []):
        if slug.startswith(prefix):
            yield slug


def _creator_completer(prefix, _parsed_args=None, **_kwargs):
    cache = _ensure_cache()
    for creator in cache.get("creators", []):
        if creator.startswith(prefix):
            yield creator


def _extract_eval_any(m: dict, keys):
    evals = m.get("evaluations") or {}
    for key in keys:
        if key in evals and evals.get(key) is not None:
            return evals.get(key)
    return None


def _print_table(models, output):
    columns = [
        ("Name", lambda m: m.get("slug", "")),
        ("Creator", _extract_creator),
        ("Context Window", _extract_context_window),
        ("Intellience Index", lambda m: _extract_eval_any(m, ["artificial_analysis_intelligence_index"])),
        ("Coding Index", lambda m: _extract_eval_any(m, ["artificial_analysis_coding_index"])),
        ("Math Index", lambda m: _extract_eval_any(m, ["artificial_analysis_math_index"])),
        ("Terminal-Bench Hard", lambda m: _extract_eval_any(m, ["terminalbench_hard"])),
        ("tau^2 Bench Telecom", lambda m: _extract_eval_any(m, ["tau2"])),
        ("AA-LCR", lambda m: _extract_eval_any(m, ["lcr"])),
        ("HLE", lambda m: _extract_eval_any(m, ["hle"])),
        ("GPQA Diamond", lambda m: _extract_eval_any(m, ["gpqa_diamond", "gpqa"])),
        ("LiveCodeBench", lambda m: _extract_eval_any(m, ["livecodebench"])),
        ("SciCode", lambda m: _extract_eval_any(m, ["scicode"])),
        ("IFBench", lambda m: _extract_eval_any(m, ["ifbench"])),
        ("AIME 2025", lambda m: _extract_eval_any(m, ["aime_25"])),
        ("MMLU Pro", lambda m: _extract_eval_any(m, ["mmlu_pro"])),
    ]

    headers = _format_headers([c[0] for c in columns], output)
    rows = []
    percent_cols = {
        "Terminal-Bench Hard",
        "tau^2 Bench Telecom",
        "AA-LCR",
        "HLE",
        "GPQA Diamond",
        "LiveCodeBench",
        "SciCode",
        "IFBench",
        "AIME 2025",
        "MMLU Pro",
    }
    for m in models:
        row = []
        for label, extractor in columns:
            val = extractor(m)
            if output in {"text", "md"} and label in percent_cols and isinstance(val, (int, float)):
                val = f"{val:.1%}"
            row.append(val)
        rows.append(row)

    if output == "text":
        colalign = [
            "right" if label in percent_cols else "left"
            for label, _ in columns
        ]
        print(tabulate(rows, headers=headers, tablefmt="plain", colalign=colalign))
    else:
        print(tabulate(rows, headers=headers, tablefmt="github"))


def main():
    parser = argparse.ArgumentParser(prog="artificialanalysis")
    parser.add_argument("--list-models", action="store_true", help="list all model slugs")
    model_arg = parser.add_argument(
        "--model",
        "-m",
        action="append",
        default=[],
        help="filter by model slug (can be repeated)",
    )
    parser.add_argument("--open", action="store_true", help="only open source")
    parser.add_argument("--closed", action="store_true", help="only closed source")
    parser.add_argument(
        "--no-context-window",
        action="store_true",
        help="skip context window retrieval from model pages",
    )
    parser.add_argument(
        "--creator",
        "-c",
        action="append",
        default=[],
        help="filter by model_creator.slug (can be repeated)",
    )
    parser.add_argument("--output", "-o", choices=sorted(FORMATS), default="text", help="output format")
    parser.add_argument("--release-date", "-d", help="release date on/after YYYY-mm-dd")
    parser.add_argument("--verbose", action="store_true", help="log requests to stderr")

    model_arg.completer = _model_completer
    # creator arg needs handle to set completer
    for action in parser._actions:
        if "--creator" in action.option_strings:
            action.completer = _creator_completer
            break
    
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    
    global _CONTEXT_ENABLED
    global _VERBOSE
    if args.verbose:
        _VERBOSE = True
    if args.no_context_window:
        _CONTEXT_ENABLED = False

    has_filters = any(
        [
            args.model,
            args.open,
            args.closed,
            args.creator,
            args.release_date,
        ]
    )
    if not args.list_models and not has_filters:
        parser.print_usage(sys.stderr)
        return 2

    api_key = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY")
    if not api_key:
        print("ARTIFICIAL_ANALYSIS_API_KEY is not set", file=sys.stderr)
        return 1

    try:
        if _VERBOSE:
            print(f"> GET {API_URL}", file=sys.stderr)
        resp = requests.get(API_URL, headers={"x-api-key": api_key}, timeout=30)
        if _VERBOSE:
            print(f"< {resp.status_code} {API_URL}", file=sys.stderr)
    except requests.RequestException as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1

    if resp.status_code != 200:
        print(f"request failed: status {resp.status_code}", file=sys.stderr)
        return 1

    data = resp.json()
    models = data.get("data", [])
    _save_cache(models)

    if args.list_models:
        for m in models:
            slug = m.get("slug")
            if slug:
                print(slug)
        return 0

    if args.model:
        wanted = set(args.model)
        models = [m for m in models if m.get("slug") in wanted]

    if args.creator:
        wanted = set(args.creator)
        def _creator_slug(m):
            if isinstance(m.get("model_creator"), dict):
                return m["model_creator"].get("slug")
            return None
        models = [m for m in models if _creator_slug(m) in wanted]

    if args.release_date:
        try:
            min_date = _parse_release_date(args.release_date)
        except ValueError:
            print("invalid --release-date, expected YYYY-mm-dd", file=sys.stderr)
            return 2
        filtered = []
        for m in models:
            rd = m.get("release_date")
            if not rd:
                continue
            try:
                if _parse_release_date(rd) >= min_date:
                    filtered.append(m)
            except ValueError:
                continue
        models = filtered

    if args.open or args.closed:
        filtered = []
        for m in models:
            is_open = _is_open_source(m)
            if args.open and is_open is True:
                filtered.append(m)
            elif args.closed and is_open is False:
                filtered.append(m)
        models = filtered

    if args.output == "json":
        data["data"] = models
        print(json.dumps(data, indent=2))
        return 0
    if args.output == "yaml":
        data["data"] = models
        print(yaml.safe_dump(data, sort_keys=False))
        return 0

    _print_table(models, args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
