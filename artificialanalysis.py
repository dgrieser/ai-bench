#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argparse
import html
import json
import os
import re
import sys
import time
from datetime import datetime, date

import argcomplete
import requests
from tabulate import tabulate
import yaml

from _context import format_context_tokens, snap_context_tokens
from _matching import normalize_slug
from _params import format_params

API_BASE_URL = "https://artificialanalysis.ai/api/v2"
# The legacy /api/v2/data/llms/models route answers 410 Gone from 2026-11-04
# (https://artificialanalysis.ai/data-api/migrate-v2-data). These are its
# documented replacements. Pro carries the licensing, parameter, context-window
# and Hugging Face fields the table would otherwise scrape off the model pages,
# so it is tried first and the free route is the fallback for a key without a
# subscription -- same shape, fewer fields.
MODELS_URL = f"{API_BASE_URL}/language/models"
MODELS_FREE_URL = f"{API_BASE_URL}/language/models/free"
TIERS = ("auto", "pro", "free")
_ENV_TIER = os.getenv("ARTIFICIAL_ANALYSIS_API_TIER", "").strip().lower()
DEFAULT_TIER = _ENV_TIER if _ENV_TIER in TIERS else "auto"
# The list endpoints page at 200 records. The cap is a runaway guard against a
# response that keeps claiming another page, not a limit we expect to reach:
# AA lists ~600 models.
MAX_PAGES = 25
FORMATS = {"json", "yaml", "md", "text"}
MODEL_PAGE_URL = "https://artificialanalysis.ai/models/{}"
# One update-all run reads the list three times in three processes:
# update_artificialanalysis_mapping.py and update.py each ask for the slugs,
# then update.py asks for the records. Paged, that is 3 x 4 requests against a
# free key's 100 a day, which the three-hourly cron blows through by tea time.
# The answer is the same list every time, so the whole response is cached on
# disk between processes and the run costs one fetch. The window is well under
# the cron interval, so no cron ever serves another's data.
RESPONSE_CACHE_PATH = os.path.expanduser("~/.cache/artificialanalysis/response.json")
try:
    RESPONSE_CACHE_TTL = int(os.getenv("ARTIFICIAL_ANALYSIS_CACHE_TTL", "3600"))
except ValueError:
    RESPONSE_CACHE_TTL = 3600
# A 403 from the Pro route costs a request of the same budget, so "auto"
# remembers the refusal rather than paying to rediscover it every fetch.
PRO_RETRY_INTERVAL = 86400
# A page that dies at the transport layer takes the whole walk with it, and the
# pages already read are spent either way. One retry, since a request that got
# no response was not billed; an HTTP status is never retried, because it was.
PAGE_RETRIES = 1
PAGE_RETRY_DELAY = 2.0
# --publish-models refuses to write fewer than this. AA lists ~644; a paged
# walk fails whole rather than short, so a truncated list cannot reach the
# writer -- this is the guard against the day AA answers 200 OK with almost
# nothing in it, which would otherwise commit an empty list over a good one.
MIN_PUBLISHED_MODELS = 100
_PAGE_METRICS_CACHE = {}
_CONTEXT_ENABLED = True
_PARAMS_ENABLED = True
_MMMU_PRO_ENABLED = True
_VERBOSE = False
CACHE_PATH = os.path.expanduser("~/.cache/artificialanalysis/models.json")
_CACHE_WARMED = False


def _is_open_source(model: dict):
    licensing = model.get("licensing")
    if isinstance(licensing, dict) and licensing.get("is_open_weights") is not None:
        return bool(licensing["is_open_weights"])
    for key in ("open_source", "is_open_source", "open"):
        if key in model:
            return bool(model.get(key))
    # The free tier drops the licensing block; fall back to the model page,
    # which links a weights repo ("Model weights" row) only for open models.
    # Page metrics are cached, so enrichment reuses this fetch.
    slug = model.get("slug", "")
    if not slug:
        return None
    return bool(_fetch_page_metrics(slug).get("hugging_face_url", ""))


def _parse_release_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


# The documented V2 contract spells a number of fields differently from the
# legacy /api/v2/data route. Everything downstream is keyed on the old names --
# the table below, update.py's SCORE_MAPPINGS, and the model pages this script
# still scrapes for the benchmarks no endpoint carries -- so the response is
# translated once on the way in rather than every reader learning both
# spellings.
_API_EVAL_ALIASES = {
    "artificial_analysis_agentic_index": "agentic_index",
    "artificial_analysis_openness_index": "openness_index",
    "aa_lcr": "lcr",
    "tau2_telecom": "tau2",
    "gpqa_diamond": "gpqa",
    "aa_omniscience_index": "omniscience",
    "aa_omniscience_accuracy": "omniscience_accuracy",
    "gdpval_aa_elo": "gdpval",
    "gdpval_aa_normalized": "gdpval_normalized",
}

# Pro's performance block carries the speed spread the model pages report as
# "outputSpeedVariance" / "timeToFirstChunkVariance". Same numbers, so they land
# under the same keys and the page fetch is left to fill the gaps.
_API_PERFORMANCE_ALIASES = {
    "percentile_05_output_tokens_per_second": "output_speed_p05",
    "quartile_25_output_tokens_per_second": "output_speed_q25",
    "median_output_tokens_per_second": "output_speed_median",
    "quartile_75_output_tokens_per_second": "output_speed_q75",
    "percentile_95_output_tokens_per_second": "output_speed_p95",
    "percentile_05_time_to_first_token_seconds": "ttft_p05",
    "quartile_25_time_to_first_token_seconds": "ttft_q25",
    "median_time_to_first_token_seconds": "ttft_median",
    "quartile_75_time_to_first_token_seconds": "ttft_q75",
    "percentile_95_time_to_first_token_seconds": "ttft_p95",
}


def _normalize_model(m: dict) -> dict:
    """Rewrite a V2 record into the field names the rest of this script uses."""
    evals = m.get("evaluations")
    if isinstance(evals, dict):
        for api_name, internal in _API_EVAL_ALIASES.items():
            if api_name not in evals:
                continue
            value = evals.pop(api_name)
            if evals.get(internal) is None:
                evals[internal] = value

    performance = m.get("performance")
    if isinstance(performance, dict):
        for api_name, internal in _API_PERFORMANCE_ALIASES.items():
            value = performance.get(api_name)
            if value is not None and m.get(internal) is None:
                m[internal] = value

    return m


def _normalize_models(models):
    for m in models:
        if isinstance(m, dict):
            _normalize_model(m)
    return models


def _tier_url(tier: str) -> str:
    return MODELS_FREE_URL if tier == "free" else MODELS_URL


def _read_response_cache() -> dict:
    try:
        with open(RESPONSE_CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return cached if isinstance(cached, dict) else {}


def _write_response_cache(cached: dict) -> None:
    try:
        os.makedirs(os.path.dirname(RESPONSE_CACHE_PATH), exist_ok=True)
        with open(RESPONSE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cached, f)
    except OSError:
        pass


def _cached_response(tier: str):
    """A payload from a recent fetch, where one is still inside the window.

    A pinned tier only accepts its own; "auto" takes whichever tier the last
    fetch reached, which is the one it would reach again.
    """
    if RESPONSE_CACHE_TTL <= 0:
        return None
    cached = _read_response_cache()
    payload = cached.get("payload")
    if not isinstance(payload, dict):
        return None
    if tier != "auto" and cached.get("tier") != tier:
        return None
    age = time.time() - cached.get("fetched_at", 0)
    if age < 0 or age > RESPONSE_CACHE_TTL:
        return None
    if _VERBOSE:
        print(
            f"< {len(payload.get('data', []))} models from the response cache "
            f"({int(age)}s old, tier {cached.get('tier')})",
            file=sys.stderr,
        )
    return payload


def _store_response(tier: str, payload: dict) -> None:
    cached = _read_response_cache()
    cached.update({"tier": tier, "fetched_at": time.time(), "payload": payload})
    _write_response_cache(cached)


def _pro_recently_denied() -> bool:
    denied_at = _read_response_cache().get("pro_denied_at")
    if not isinstance(denied_at, (int, float)):
        return False
    return 0 <= time.time() - denied_at <= PRO_RETRY_INTERVAL


def _remember_pro_denied() -> None:
    cached = _read_response_cache()
    cached["pro_denied_at"] = time.time()
    _write_response_cache(cached)


def _log_rate_limit(resp) -> None:
    if not _VERBOSE:
        return
    limit = resp.headers.get("X-RateLimit-Limit")
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if limit or remaining:
        print(
            f"< rate limit {remaining or '?'}/{limit or '?'} left this window",
            file=sys.stderr,
        )


def _error_message(resp) -> str:
    """The API's own error text where it sent one, else the bare status."""
    message = f"status {resp.status_code}"
    try:
        payload = resp.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("error"), str) and payload["error"]:
        message = f"{message}: {payload['error']}"
    retry_after = resp.headers.get("Retry-After")
    if resp.status_code == 429 and retry_after:
        message = f"{message} (retry after {retry_after}s)"
    return message


def _fetch_all_pages(url: str, api_key: str, timeout: int):
    """Walk one endpoint's pages. Returns (payload, error, status)."""
    merged = None
    models = []
    page = 1
    while page <= MAX_PAGES:
        resp = None
        for attempt in range(PAGE_RETRIES + 1):
            try:
                if _VERBOSE:
                    print(f"> GET {url}?page={page}", file=sys.stderr)
                resp = requests.get(
                    url,
                    headers={"x-api-key": api_key},
                    params={"page": page},
                    timeout=timeout,
                )
                break
            except requests.RequestException as exc:
                if attempt == PAGE_RETRIES:
                    return None, f"request failed: {exc}", None
                if _VERBOSE:
                    print(f"< {exc}; retrying page {page}", file=sys.stderr)
                time.sleep(PAGE_RETRY_DELAY)
        if _VERBOSE:
            print(f"< {resp.status_code} {url}?page={page}", file=sys.stderr)
        _log_rate_limit(resp)
        if resp.status_code != 200:
            return None, _error_message(resp), resp.status_code
        try:
            payload = resp.json()
        except ValueError:
            return None, "response was not JSON", resp.status_code
        if not isinstance(payload, dict):
            return None, "response was not an object", resp.status_code

        data = payload.get("data")
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            models.extend(item for item in data if isinstance(item, dict))
        if merged is None:
            # The envelope (tier, intelligence_index_version) is the same on
            # every page; the pagination block describes one page and would
            # only mislead about the merged list, so it is dropped.
            merged = {k: v for k, v in payload.items() if k not in {"data", "pagination"}}

        pagination = payload.get("pagination")
        if not isinstance(pagination, dict) or not pagination.get("has_more"):
            break
        page += 1

    merged = merged if merged is not None else {}
    merged["data"] = _normalize_models(models)
    return merged, "", 200


def _fetch_models(api_key: str, tier: str = "auto", timeout: int = 30, use_cache: bool = True):
    """Every model AA lists, as (payload, error).

    The V2 list endpoints paginate, so this walks them and hands back one
    payload whose "data" holds the lot, normalized -- served from the response
    cache where a recent fetch left one. On "auto" a 403 from the Pro route --
    a key without a Pro subscription -- falls back to the free route, and the
    refusal is remembered so the next fetch does not pay for it again. Any
    other failure is reported as it stands rather than silently downgrading the
    fields the caller gets.
    """
    if use_cache:
        payload = _cached_response(tier)
        if payload is not None:
            return payload, ""

    if tier == "auto":
        order = ["free"] if _pro_recently_denied() else ["pro", "free"]
    else:
        order = [tier]

    error = ""
    for index, attempt in enumerate(order):
        url = _tier_url(attempt)
        payload, failure, status = _fetch_all_pages(url, api_key, timeout)
        if payload is not None:
            _store_response(attempt, payload)
            return payload, ""
        error = f"{url}: {failure}"
        if status == 403 and attempt == "pro":
            _remember_pro_denied()
        if status != 403 or index == len(order) - 1:
            break
        if _VERBOSE:
            print(f"< no Pro access; retrying on {MODELS_FREE_URL}", file=sys.stderr)
    return None, error


def _published_models(models):
    """The published list: what a page needs to offer a slug, and nothing else.

    Sorted, and deliberately carrying no timestamp. This file is committed on
    every refresh, and a wall clock in it would rewrite it on runs where AA
    published nothing new -- the same churn pending_prompts.py keeps out of
    the queue. The fields below change about as often as the list itself does,
    so an unchanged list is an unchanged file and an empty diff.
    """
    published = []
    for m in models:
        slug = m.get("slug")
        if not isinstance(slug, str) or not slug:
            continue
        creator = m.get("model_creator")
        published.append(
            {
                "slug": slug,
                "name": m.get("name") or "",
                "creator": (creator or {}).get("name", "") if isinstance(creator, dict) else "",
                "release_date": m.get("release_date") or "",
            }
        )
    # Case-insensitive, so "QwQ-32B-Preview" sits with its neighbours rather
    # than above every lowercase slug -- then the rest of the record, so the
    # order is total even where two entries share a slug. Sorting on the slug
    # alone would leave those two in whatever order the API happened to send,
    # which is the one input to this file that is not stable run to run.
    published.sort(
        key=lambda entry: (
            entry["slug"].lower(),
            entry["slug"],
            entry["name"],
            entry["creator"],
            entry["release_date"],
        )
    )

    # And then one entry per slug. Offset paging can hand back the same model
    # twice all by itself -- four page reads seconds apart, and a model
    # inserted at AA's end between two of them shifts everything after it --
    # so a duplicate here is a fetch artefact rather than news. The page keys
    # its lookup by slug and would otherwise offer the same suggestion twice.
    deduped = []
    for entry in published:
        if deduped and deduped[-1]["slug"] == entry["slug"]:
            continue
        deduped.append(entry)
    return {"count": len(deduped), "models": deduped}


def _write_published_models(models, path: str) -> int:
    payload = _published_models(models)
    if payload["count"] < MIN_PUBLISHED_MODELS:
        # Leave whatever is already committed in place. A page with a stale
        # list still offers the right slugs for every model AA had yesterday;
        # a page with an empty one offers nothing at all.
        print(
            f"refusing to publish {payload['count']} models (under {MIN_PUBLISHED_MODELS});"
            " leaving the existing file alone",
            file=sys.stderr,
        )
        return 1

    out = os.path.abspath(path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"published {payload['count']} models to {path}", file=sys.stderr)
    return 0


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


def _creator_slug(m: dict) -> str:
    """The creator's slug, derived from its name where the API sends none.

    The legacy route carried "model_creator.slug"; the V2 records carry an id,
    a name and (on Pro) a country. The names slugify to what AA itself uses in
    its urls ("OpenAI" -> "openai"), which is what --creator has always taken
    and what the completion cache is filled with.
    """
    creator = m.get("model_creator")
    if not isinstance(creator, dict):
        return ""
    slug = creator.get("slug")
    if isinstance(slug, str) and slug:
        return slug
    name = creator.get("name")
    if isinstance(name, str) and name:
        return normalize_slug(name)
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
    # These pages are megabytes each and they carry most of a free-tier run's
    # columns, so a dropped connection is worth one retry: the alternative is a
    # model that silently reports none of them.
    resp = None
    for attempt in range(PAGE_RETRIES + 1):
        try:
            if _VERBOSE:
                print(f"> GET {url}", file=sys.stderr)
            resp = requests.get(url, headers={"RSC": "1"}, timeout=15)
            break
        except requests.RequestException as exc:
            if attempt == PAGE_RETRIES:
                if _VERBOSE:
                    print(f"< {exc}; giving up on {url}", file=sys.stderr)
                _PAGE_METRICS_CACHE[slug] = result
                return result
            if _VERBOSE:
                print(f"< {exc}; retrying {url}", file=sys.stderr)
            time.sleep(PAGE_RETRY_DELAY)

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

    _PAGE_METRICS_CACHE[slug] = result
    return result


def _api_context_window(m: dict):
    tokens = m.get("context_window_tokens")
    if not isinstance(tokens, (int, float)) or isinstance(tokens, bool) or tokens <= 0:
        return ""
    # The API reports the raw count, same as the pages do, so it goes through
    # the same snap back to the advertised size (131072 -> 128k).
    return format_context_tokens(snap_context_tokens(int(tokens)))


def _api_params(m: dict):
    parameters = m.get("parameters")
    if not isinstance(parameters, dict):
        return ""
    return format_params(parameters.get("total"), parameters.get("active"))


def _extract_context_window(m: dict):
    value = _api_context_window(m)
    if value:
        return value
    if not _CONTEXT_ENABLED:
        return ""
    return _fetch_page_metrics(m.get("slug", "")).get("context_window", "")


def _extract_params(m: dict):
    value = _api_params(m)
    if value:
        return value
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
    url = m.get("huggingface_url")
    if isinstance(url, str) and url:
        return _canonical_hf_url(url)
    return _fetch_page_metrics(m.get("slug", "")).get("hugging_face_url", "")


def _extract_mmmu_pro(m: dict):
    val = _extract_eval_any(m, ["mmmu_pro"])
    if val is not None:
        return val
    if not _MMMU_PRO_ENABLED:
        return None
    return _fetch_page_metrics(m.get("slug", "")).get("mmmu_pro")


# Benchmarks the model pages carry. Pro now answers most of them itself, so
# these fill the gaps rather than override: a page value is used where the API
# sent none, which is every one of them on the free tier and the handful below
# that no endpoint carries at all.
_PAGE_EVALS = [
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


def _extract_metric(m: dict, key: str):
    """A metric off the API record where it carries one, else off the page.

    Normalization lands the API's own values under these keys -- the speed
    spread at the top level, the benchmarks under "evaluations" -- so both
    shapes are checked before paying for a page fetch.
    """
    val = m.get(key)
    if val is not None:
        return val
    val = _extract_eval_any(m, [key])
    if val is not None:
        return val
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
        for key in _PAGE_EVALS:
            if evals.get(key) is None:
                val = page_metrics.get(key)
                if val is not None:
                    evals[key] = val

        # The one field the API states the other way round: it reports the
        # share of answers that are *not* hallucinated, while the pages, the
        # column and update.py are keyed on the rate itself. Taken as the
        # complement, and only after the page pass above, so a rate AA reported
        # directly is never displaced by one inferred from its opposite.
        if evals.get("omniscience_hallucination_rate") is None:
            rate = evals.get("aa_omniscience_non_hallucination_rate")
            if isinstance(rate, (int, float)) and not isinstance(rate, bool):
                evals["omniscience_hallucination_rate"] = 1 - rate

        m["evaluations"] = evals

        for key in _PAGE_META_KEYS:
            if m.get(key) is None:
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
            cslug = _creator_slug(m)
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
    payload, _error = _fetch_models(api_key, tier=DEFAULT_TIER, timeout=15)
    if payload is not None:
        _save_cache(payload.get("data", []))
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
        ("Agentic Index", lambda m: _extract_metric(m, "agentic_index")),
        ("AA-Omniscience", lambda m: _extract_metric(m, "omniscience")),
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
        ("GDPval", lambda m: _extract_metric(m, "gdpval_normalized")),
        ("IT-Bench SRE", lambda m: _extract_metric(m, "it_bench_sre")),
        ("Briefcase", lambda m: _extract_metric(m, "briefcase")),
        ("Crit-Pt", lambda m: _extract_metric(m, "critpt")),
        ("Apex Agents", lambda m: _extract_metric(m, "apex_agents")),
        ("Openness", lambda m: _extract_metric(m, "openness_index")),
        ("Out Speed p05", lambda m: _extract_metric(m, "output_speed_p05")),
        ("Out Speed p95", lambda m: _extract_metric(m, "output_speed_p95")),
        ("TTFT p05", lambda m: _extract_metric(m, "ttft_p05")),
        ("TTFT p95", lambda m: _extract_metric(m, "ttft_p95")),
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
    parser.add_argument(
        "--tier",
        choices=TIERS,
        default=DEFAULT_TIER,
        help="which endpoint to read (auto falls back to free without Pro access)",
    )
    parser.add_argument(
        "--publish-models",
        metavar="FILE",
        help="write the slug list the admin page reads to FILE, and stop",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="re-read the API even if a recent response is cached",
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
    if not args.list_models and not args.publish_models and not has_filters:
        parser.print_usage(sys.stderr)
        return 2

    api_key = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY")
    if not api_key:
        print("ARTIFICIAL_ANALYSIS_API_KEY is not set", file=sys.stderr)
        return 1

    payload, error = _fetch_models(api_key, tier=args.tier, use_cache=not args.no_cache)
    if payload is None:
        print(f"request failed: {error}", file=sys.stderr)
        return 1

    models = payload.get("data", [])
    _save_cache(models)

    if args.publish_models:
        return _write_published_models(models, args.publish_models)

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
        wanted = {normalize_slug(c) for c in args.creator}
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
        payload["data"] = models
        print(json.dumps(payload, indent=2))
        return 0
    if args.output == "yaml":
        _enrich_structured_metrics(models)
        payload["data"] = models
        print(yaml.safe_dump(payload, sort_keys=False))
        return 0

    _print_table(models, args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
