#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse
import html
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
_PAGE_METRICS_CACHE = {}
_CONTEXT_ENABLED = True
_MMMU_PRO_ENABLED = True
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


def _normalize_page_text(text: str):
    normalized = html.unescape(text)
    normalized = normalized.replace("\\/", "/")
    normalized = re.sub(r"\\u002[fF]", "/", normalized)
    normalized = normalized.replace("\\u003a", ":").replace("\\u003A", ":")
    return normalized


def _parse_context_window(text: str):
    m = re.search(r"Context window.+?<span[^>]*>([0-9]+[kmb])", text, re.IGNORECASE)
    if m:
        return m.group(1)
    # RSC payloads expose raw token counts instead of the rendered label.
    m = re.search(r'"context_window_tokens"\s*:\s*([0-9]+)', text, re.IGNORECASE)
    if m:
        tokens = int(m.group(1))
        if tokens >= 1_000_000_000:
            return f"{tokens // 1_000_000_000}b"
        if tokens >= 1_000_000:
            return f"{tokens // 1_000_000}m"
        if tokens >= 1_000:
            return f"{tokens // 1_000}k"
        return str(tokens)
    return ""


def _parse_hugging_face_url(text: str):
    normalized = _normalize_page_text(text)
    match = re.search(r"https://huggingface\.co/[A-Za-z0-9][^\s\"'<>\\),]+", normalized)
    if not match:
        return ""

    return match.group(0).rstrip(".,;:")


def _parse_creator(text: str, expected_name: str = ""):
    normalized = _normalize_page_text(text)
    result = {"name": expected_name, "url": ""}

    if expected_name:
        name_pattern = re.escape(expected_name)
        patterns = [
            rf'"href":"(https?://[^"]+)","target":"_blank"[^{{}}]{{0,250}}"children":"{name_pattern}"',
            rf'"name":"{name_pattern}"[^{{}}]{{0,500}}"creator_url":"([^"]*)"',
        ]
    else:
        patterns = [
            r'"href":"(https?://[^"]+)","target":"_blank"[^{}]{0,250}"children":"([^"]+)"',
            r'"name":"([^"]+)"[^{}]{0,500}"creator_url":"([^"]*)"',
        ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        if expected_name:
            result["url"] = match.group(1)
        elif pattern.startswith('"href"'):
            result["url"] = match.group(1)
            result["name"] = match.group(2)
        else:
            result["name"] = match.group(1)
            result["url"] = match.group(2)
        if result["url"]:
            break

    return result


def _parse_mmmu_pro(text: str, slug: str):
    model_path = f"/models/{slug}"
    target = f'"model_url":"{model_path}"'
    pos = text.find(target)
    if pos == -1:
        return None

    # In the RSC stream, benchmark fields may appear before model_url for
    # the same model. Restrict to the chunk since the previous model_url.
    prev_model = text.rfind('"model_url":"/models/', 0, pos)
    start = prev_model if prev_model != -1 else 0
    block = text[start : pos + len(target)]
    match = re.search(r'"mmmu_pro":([0-9.]+)', block)
    if not match:
        return None
    return float(match.group(1))


def _fetch_page_metrics(slug: str, creator_name: str = ""):
    if not slug:
        return {"context_window": "", "hugging_face_url": "", "creator": {"name": creator_name, "url": ""}, "mmmu_pro": None}
    if slug in _PAGE_METRICS_CACHE:
        return _PAGE_METRICS_CACHE[slug]

    result = {"context_window": "", "hugging_face_url": "", "creator": {"name": creator_name, "url": ""}, "mmmu_pro": None}
    url = MODEL_PAGE_URL.format(slug)
    try:
        if _VERBOSE:
            print(f"> GET {url}", file=sys.stderr)
        resp = requests.get(url, headers={"RSC": "1"}, timeout=15)
        if _VERBOSE:
            print(f"< {resp.status_code} {url}", file=sys.stderr)
        if resp.status_code != 200:
            _PAGE_METRICS_CACHE[slug] = result
            return result
        result["context_window"] = _parse_context_window(resp.text)
        result["hugging_face_url"] = _parse_hugging_face_url(resp.text)
        result["creator"] = _parse_creator(resp.text, creator_name)
        result["mmmu_pro"] = _parse_mmmu_pro(resp.text, slug)
    except requests.RequestException:
        pass

    _PAGE_METRICS_CACHE[slug] = result
    return result


def _extract_context_window(m: dict):
    if not _CONTEXT_ENABLED:
        return ""
    return _fetch_page_metrics(m.get("slug", "")).get("context_window", "")


def _extract_creator_name(m: dict):
    creator = _extract_page_creator(m)
    return creator.get("name") or _extract_creator(m)


def _extract_creator_url(m: dict):
    return _extract_page_creator(m).get("url", "")


def _extract_page_creator(m: dict):
    creator_name = ""
    if isinstance(m.get("model_creator"), dict):
        creator_name = m["model_creator"].get("name", "")
    return _fetch_page_metrics(m.get("slug", ""), creator_name).get("creator", {})


def _extract_hugging_face_url(m: dict):
    return _fetch_page_metrics(m.get("slug", "")).get("hugging_face_url", "")


def _extract_mmmu_pro(m: dict):
    val = _extract_eval_any(m, ["mmmu_pro"])
    if val is not None:
        return val
    if not _MMMU_PRO_ENABLED:
        return None
    return _fetch_page_metrics(m.get("slug", "")).get("mmmu_pro")


def _enrich_structured_metrics(models):
    for m in models:
        evals = m.get("evaluations")
        if not isinstance(evals, dict):
            evals = {}
        evals["mmmu_pro"] = _extract_mmmu_pro(m)
        m["evaluations"] = evals

        context = _extract_context_window(m)
        hugging_face_url = _extract_hugging_face_url(m)
        page_creator = _extract_page_creator(m)
        if isinstance(m.get("model_creator"), dict):
            if page_creator.get("name"):
                m["model_creator"]["name"] = page_creator["name"]
            if page_creator.get("url"):
                m["model_creator"]["url"] = page_creator["url"]
        model_url = hugging_face_url or m.get("url")
        if "evaluations" in m:
            reordered = {}
            for key, value in m.items():
                if key == "url":
                    continue
                if key == "context":
                    continue
                if key == "evaluations":
                    if model_url:
                        reordered["url"] = model_url
                    reordered["context"] = context
                reordered[key] = value
            m.clear()
            m.update(reordered)
        else:
            if model_url:
                m["url"] = model_url
            m["context"] = context


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
        ("Creator", _extract_creator_name),
        ("Creator URL", _extract_creator_url),
        ("Context Window", _extract_context_window),
        ("Hugging Face", _extract_hugging_face_url),
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
        ("MMMU Pro", _extract_mmmu_pro),
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
        "MMMU Pro",
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
        "--no-mmmu-pro",
        action="store_true",
        help="skip MMMU Pro retrieval from model pages",
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
    global _MMMU_PRO_ENABLED
    global _VERBOSE
    if args.verbose:
        _VERBOSE = True
    if args.no_context_window:
        _CONTEXT_ENABLED = False
    if args.no_mmmu_pro:
        _MMMU_PRO_ENABLED = False

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
        _enrich_structured_metrics(models)
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
