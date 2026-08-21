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

from _context import format_context_tokens, snap_context_tokens
from _params import format_params

API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
FORMATS = {"json", "yaml", "md", "text"}
MODEL_PAGE_URL = "https://artificialanalysis.ai/models/{}"
_PAGE_METRICS_CACHE = {}
_CONTEXT_ENABLED = True
_PARAMS_ENABLED = True
_MMMU_PRO_ENABLED = True
_VERBOSE = False
CACHE_PATH = os.path.expanduser("~/.cache/artificialanalysis/models.json")
_CACHE_WARMED = False


def _is_open_source(model: dict):
    for key in ("open_source", "is_open_source", "open"):
        if key in model:
            return bool(model.get(key))
    # AA's v2 API no longer carries an open-source flag; fall back to the model
    # page, which links a weights repo ("Model weights" row) only for open
    # models. Page metrics are cached, so enrichment reuses this fetch.
    slug = model.get("slug", "")
    if not slug:
        return None
    return bool(_fetch_page_metrics(slug).get("hugging_face_url", ""))


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


def _current_model_chunk(text: str, slug: str):
    # The page's own record lives in the "currentModel" payload; the rest of the
    # page carries comparison models with the same keys, so anchor on that
    # payload and confirm the slug before reading anything out of it.
    anchor = text.find('"currentModel":')
    if anchor == -1 or f'"slug":"{slug}"' not in text[anchor : anchor + 2000]:
        anchor = text.find(f'"slug":"{slug}"')
    if anchor == -1:
        return ""
    return text[anchor : anchor + 2000]


def _parse_context_window(text: str, slug: str = ""):
    m = re.search(r"Context window.+?<span[^>]*>([0-9]+[kmb])", text, re.IGNORECASE)
    if m:
        return m.group(1)

    # RSC payloads expose raw token counts instead of the rendered label, as
    # "contextWindowTokens" on the model record and "context_window_tokens"
    # elsewhere. Snap them back to the advertised size (131072 -> 128k).
    tokens = None
    chunk = _current_model_chunk(text, slug) if slug else ""
    if chunk:
        m = re.search(r'"contextWindowTokens":([0-9]+)', chunk)
        if m:
            tokens = int(m.group(1))
    if tokens is None:
        m = re.search(r'"context_window_tokens"\s*:\s*([0-9]+)', text, re.IGNORECASE)
        if m:
            tokens = int(m.group(1))
    if tokens is None or tokens <= 0:
        return ""

    return format_context_tokens(snap_context_tokens(tokens))


def _parse_params(text: str, slug: str):
    chunk = _current_model_chunk(text, slug)
    if not chunk:
        return ""

    match = re.search(
        r'"parameters":(null|[0-9.]+),"inferenceParametersActiveBillions":(null|[0-9.]+)',
        chunk,
    )
    if not match:
        return ""

    total, active = (None if g == "null" else g for g in match.groups())
    return format_params(total, active)


def _canonical_hf_url(url: str) -> str:
    # Reduce to the canonical repo url (https://huggingface.co/<org>/<repo>),
    # dropping trailing paths like /tree/main, /blob/main/LICENSE, query, or
    # fragment that sometimes trail the href.
    match = re.match(r"(https://huggingface\.co/[^/\s\"'<>\\),?#]+/[^/\s\"'<>\\),?#]+)", url)
    if match:
        return match.group(1)
    return url.rstrip("/.,;:")


def _parse_hugging_face_url(text: str):
    normalized = _normalize_page_text(text)
    # The model's own weights repo lives in the page's "Model weights" row; the
    # rest of the page links a global model catalog, so grabbing the first
    # huggingface.co link picks up an unrelated repo on closed-weights models.
    # Anchor on the label and take the href that immediately follows it.
    match = re.search(
        r'"Model weights".{0,400}?"href":"(https://huggingface\.co/[^"]+)"',
        normalized,
        re.DOTALL,
    )
    if not match:
        return ""

    return _canonical_hf_url(match.group(1))


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


_PAGE_FLOAT_FIELDS = [
    ("agenticIndex", "agentic_index"),
    ("omniscience", "omniscience"),
    ("gdpval", "gdpval"),
    ("gdpvalNormalized", "gdpval_normalized"),
    ("itBenchSre", "it_bench_sre"),
    ("critpt", "critpt"),
    ("apexAgents", "apex_agents"),
    ("mmmuPro", "mmmu_pro"),
    ("terminalbenchV21", "terminalbench_v2_1"),
    ("terminalbenchHard", "terminalbench_hard"),
    ("ifbench", "ifbench"),
    ("harveyLabCriteriaPass", "harvey_lab_criteria_pass"),
    ("automationBenchPartialScore", "automation_bench_partial_score"),
    ("enterpriseOpsGym", "enterprise_ops_gym"),
    ("codingIndex", "coding_index"),
    ("intelligenceIndex", "intelligence_index"),
    ("livecodebench", "livecodebench"),
    ("scicode", "scicode"),
    ("aime25", "aime_25"),
]

_PAGE_BOOL_FIELDS = [
    ("microevalsEnabled", "microevals_enabled"),
    ("intelligenceIndexIsEstimated", "intelligence_index_is_estimated"),
]

_PAGE_OBJECT_FIELDS = {
    "omniscienceBreakdown": [
        ("accuracy", "omniscience_accuracy", float),
        ("hallucinationRate", "omniscience_hallucination_rate", float),
    ],
    # The capture stops at the first "}", i.e. inside the leading "overall"
    # sub-object, so "elo" is Briefcase's composite Elo rather than the
    # analytical-quality or presentation Elo that follow it.
    "briefcaseBreakdown": [
        ("elo", "briefcase", float),
    ],
    "openness": [
        ("opennessIndex", "openness_index", float),
        ("modelAvailability", "openness_model_availability", int),
        ("transparencyMethodology", "openness_transparency_methodology", int),
        ("transparencyPostTrainingData", "openness_transparency_post_training_data", int),
        ("transparencyPreTrainingData", "openness_transparency_pre_training_data", int),
    ],
    "outputSpeedVariance": [
        ("p05", "output_speed_p05", float),
        ("q25", "output_speed_q25", float),
        ("median", "output_speed_median", float),
        ("q75", "output_speed_q75", float),
        ("p95", "output_speed_p95", float),
    ],
    "timeToFirstChunkVariance": [
        ("p05", "ttft_p05", float),
        ("q25", "ttft_q25", float),
        ("median", "ttft_median", float),
        ("q75", "ttft_q75", float),
        ("p95", "ttft_p95", float),
    ],
}


def _parse_metrics_block(text: str, slug: str):
    slug_anchor = f'"slug":"{slug}"'
    slug_pos = text.find(slug_anchor)
    if slug_pos == -1:
        return {}
    me_pos = text.find('"microevalsEnabled"', slug_pos)
    if me_pos == -1:
        return {}
    chunk = text[me_pos : me_pos + 5000]

    result = {}

    for page_key, result_key in _PAGE_FLOAT_FIELDS:
        m = re.search(rf'"{page_key}":(null|-?[0-9.]+)', chunk)
        if m:
            val = m.group(1)
            result[result_key] = None if val == "null" else float(val)

    for page_key, result_key in _PAGE_BOOL_FIELDS:
        m = re.search(rf'"{page_key}":(true|false)', chunk)
        if m:
            result[result_key] = m.group(1) == "true"

    for obj_name, sub_fields in _PAGE_OBJECT_FIELDS.items():
        pattern = r'"' + obj_name + r'":\{([^}]+)\}'
        m = re.search(pattern, chunk)
        if not m:
            continue
        obj_chunk = m.group(1)
        for sub_key, result_key, caster in sub_fields:
            sm = re.search(rf'"{sub_key}":(null|-?[0-9.]+)', obj_chunk)
            if sm:
                val = sm.group(1)
                if val == "null":
                    result[result_key] = None
                else:
                    result[result_key] = caster(float(val))

    return result


def _fetch_page_metrics(slug: str, creator_name: str = ""):
    if not slug:
        return {"context_window": "", "params": "", "hugging_face_url": "", "creator": {"name": creator_name, "url": ""}, "mmmu_pro": None}
    if slug in _PAGE_METRICS_CACHE:
        return _PAGE_METRICS_CACHE[slug]

    result = {"context_window": "", "params": "", "hugging_face_url": "", "creator": {"name": creator_name, "url": ""}, "mmmu_pro": None}
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
        result["context_window"] = _parse_context_window(resp.text, slug)
        result["params"] = _parse_params(resp.text, slug)
        result["hugging_face_url"] = _parse_hugging_face_url(resp.text)
        result["creator"] = _parse_creator(resp.text, creator_name)
        metrics = _parse_metrics_block(resp.text, slug)
        result.update(metrics)
        if "mmmu_pro" not in result:
            result["mmmu_pro"] = None
    except requests.RequestException:
        pass

    _PAGE_METRICS_CACHE[slug] = result
    return result


def _extract_context_window(m: dict):
    if not _CONTEXT_ENABLED:
        return ""
    return _fetch_page_metrics(m.get("slug", "")).get("context_window", "")


def _extract_params(m: dict):
    if not _PARAMS_ENABLED:
        return ""
    return _fetch_page_metrics(m.get("slug", "")).get("params", "")


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


_PAGE_ONLY_EVALS = [
    "agentic_index",
    "omniscience",
    "omniscience_accuracy",
    "omniscience_hallucination_rate",
    "gdpval",
    "gdpval_normalized",
    "it_bench_sre",
    "briefcase",
    "critpt",
    "apex_agents",
    "harvey_lab_criteria_pass",
    "automation_bench_partial_score",
    "enterprise_ops_gym",
]

_PAGE_FALLBACK_EVALS = [
    "terminalbench_v2_1",
    "terminalbench_hard",
    "ifbench",
    "livecodebench",
    "scicode",
    "aime_25",
]

_PAGE_META_KEYS = [
    "microevals_enabled",
    "intelligence_index_is_estimated",
    "openness_index",
    "openness_model_availability",
    "openness_transparency_methodology",
    "openness_transparency_post_training_data",
    "openness_transparency_pre_training_data",
    "output_speed_p05",
    "output_speed_q25",
    "output_speed_median",
    "output_speed_q75",
    "output_speed_p95",
    "ttft_p05",
    "ttft_q25",
    "ttft_median",
    "ttft_q75",
    "ttft_p95",
]


def _extract_page_eval(m: dict, key: str):
    return _fetch_page_metrics(m.get("slug", "")).get(key)


def _extract_eval_or_page(m: dict, api_keys, page_key: str):
    val = _extract_eval_any(m, api_keys)
    if val is not None:
        return val
    return _fetch_page_metrics(m.get("slug", "")).get(page_key)


def _enrich_structured_metrics(models):
    for m in models:
        evals = m.get("evaluations")
        if not isinstance(evals, dict):
            evals = {}
        evals["mmmu_pro"] = _extract_mmmu_pro(m)

        page_metrics = _fetch_page_metrics(m.get("slug", ""))
        for key in _PAGE_ONLY_EVALS:
            val = page_metrics.get(key)
            if val is not None:
                evals[key] = val
        for key in _PAGE_FALLBACK_EVALS:
            if evals.get(key) is None:
                val = page_metrics.get(key)
                if val is not None:
                    evals[key] = val
        m["evaluations"] = evals

        for key in _PAGE_META_KEYS:
            val = page_metrics.get(key)
            if val is not None:
                m[key] = val

        context = _extract_context_window(m)
        params = _extract_params(m)
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
                if key in {"url", "params", "context"}:
                    continue
                if key == "evaluations":
                    if model_url:
                        reordered["url"] = model_url
                    reordered["params"] = params
                    reordered["context"] = context
                reordered[key] = value
            m.clear()
            m.update(reordered)
        else:
            if model_url:
                m["url"] = model_url
            m["params"] = params
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
        ("Parameters", _extract_params),
        ("Context Window", _extract_context_window),
        ("Hugging Face", _extract_hugging_face_url),
        ("Intelligence Index", lambda m: _extract_eval_any(m, ["artificial_analysis_intelligence_index"])),
        ("Coding Index", lambda m: _extract_eval_any(m, ["artificial_analysis_coding_index"])),
        ("Math Index", lambda m: _extract_eval_any(m, ["artificial_analysis_math_index"])),
        ("Agentic Index", lambda m: _extract_page_eval(m, "agentic_index")),
        ("AA-Omniscience", lambda m: _extract_page_eval(m, "omniscience")),
        ("Terminal-Bench v2.1", lambda m: _extract_eval_or_page(m, ["terminalbench_v2_1"], "terminalbench_v2_1")),
        ("tau^2 Bench Telecom", lambda m: _extract_eval_any(m, ["tau2"])),
        ("AA-LCR", lambda m: _extract_eval_any(m, ["lcr"])),
        ("HLE", lambda m: _extract_eval_any(m, ["hle"])),
        ("GPQA Diamond", lambda m: _extract_eval_any(m, ["gpqa_diamond", "gpqa"])),
        ("LiveCodeBench", lambda m: _extract_eval_or_page(m, ["livecodebench"], "livecodebench")),
        ("SciCode", lambda m: _extract_eval_or_page(m, ["scicode"], "scicode")),
        ("IFBench", lambda m: _extract_eval_or_page(m, ["ifbench"], "ifbench")),
        ("AIME 2025", lambda m: _extract_eval_or_page(m, ["aime_25"], "aime_25")),
        ("MMMU Pro", _extract_mmmu_pro),
        ("GDPval", lambda m: _extract_page_eval(m, "gdpval_normalized")),
        ("IT-Bench SRE", lambda m: _extract_page_eval(m, "it_bench_sre")),
        ("Briefcase", lambda m: _extract_page_eval(m, "briefcase")),
        ("Crit-Pt", lambda m: _extract_page_eval(m, "critpt")),
        ("Apex Agents", lambda m: _extract_page_eval(m, "apex_agents")),
        ("Openness", lambda m: _extract_page_eval(m, "openness_index")),
        ("Out Speed p05", lambda m: _extract_page_eval(m, "output_speed_p05")),
        ("Out Speed p95", lambda m: _extract_page_eval(m, "output_speed_p95")),
        ("TTFT p05", lambda m: _extract_page_eval(m, "ttft_p05")),
        ("TTFT p95", lambda m: _extract_page_eval(m, "ttft_p95")),
    ]

    headers = _format_headers([c[0] for c in columns], output)
    rows = []
    percent_cols = {
        "Terminal-Bench v2.1",
        "tau^2 Bench Telecom",
        "AA-LCR",
        "HLE",
        "GPQA Diamond",
        "LiveCodeBench",
        "SciCode",
        "IFBench",
        "AIME 2025",
        "MMMU Pro",
        "GDPval",
        "IT-Bench SRE",
        "Crit-Pt",
        "Apex Agents",
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
        "--no-params",
        action="store_true",
        help="skip parameter count retrieval from model pages",
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
    global _PARAMS_ENABLED
    global _MMMU_PRO_ENABLED
    global _VERBOSE
    if args.verbose:
        _VERBOSE = True
    if args.no_context_window:
        _CONTEXT_ENABLED = False
    if args.no_params:
        _PARAMS_ENABLED = False
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
        _enrich_structured_metrics(models)
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
