#!/usr/bin/env python3
"""Update benchmark JSON scores using artificialanalysis.py and swe-rebench."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import artificialanalysis
import derive_coding_index
import fetch_datacurve
import fetch_deepswe
import fetch_evals_report
import fetch_frontiercode
import fetch_frontierswe
import fetch_huggingface
import fetch_llmstats
import fetch_osworld
import fetch_swe_atlas
import fetch_swe_marathon
import fetch_swe_rebench
import fetch_toolathlon
from _context import format_context_tokens, snap_context_tokens
from _params import fetch_hf_params, normalize_params
from _scores import round_score, stamp_score_source, stamp_score_updated
from fill_source_urls import canonical

from _swe_rebench_mapping import load_rebench_to_slug_mapping
from _osworld_mapping import load_osworld_to_slug_mapping
from _huggingface_mapping import load_hf_to_key_mapping
from _deepswe_mapping import load_deepswe_to_slug_mapping
from _toolathlon_mapping import load_toolathlon_to_slug_mapping
from _frontierswe_mapping import load_frontierswe_to_slug_mapping
from _frontiercode_mapping import load_frontiercode_to_slug_mapping
from _swe_atlas_mapping import load_swe_atlas_to_slug_mapping
from _evals_report_mapping import load_evals_report_to_slug_mapping
from _swe_marathon_mapping import load_swe_marathon_to_slug_mapping
from _spheron_mapping import load_spheron_to_slug_mapping
from _llmstats_mapping import (
    load_llmstats_to_slug_mapping,
    load_llmstats_benchmark_to_key_mapping,
)
from _artificialanalysis_mapping import load_llm_to_aa_slugs

AA_SCRIPT = Path(__file__).resolve().with_name("artificialanalysis.py")
SWE_REBENCH_SCRIPT = Path(__file__).resolve().with_name("fetch_swe_rebench.py")
OSWORLD_SCRIPT = Path(__file__).resolve().with_name("fetch_osworld.py")
HF_SCRIPT = Path(__file__).resolve().with_name("fetch_huggingface.py")
DEEPSWE_SCRIPT = Path(__file__).resolve().with_name("fetch_deepswe.py")
DATACURVE_SCRIPT = Path(__file__).resolve().with_name("fetch_datacurve.py")
TOOLATHLON_SCRIPT = Path(__file__).resolve().with_name("fetch_toolathlon.py")
FRONTIERSWE_SCRIPT = Path(__file__).resolve().with_name("fetch_frontierswe.py")
FRONTIERCODE_SCRIPT = Path(__file__).resolve().with_name("fetch_frontiercode.py")
SWE_ATLAS_SCRIPT = Path(__file__).resolve().with_name("fetch_swe_atlas.py")
EVALS_REPORT_SCRIPT = Path(__file__).resolve().with_name("fetch_evals_report.py")
SWE_MARATHON_SCRIPT = Path(__file__).resolve().with_name("fetch_swe_marathon.py")
SPHERON_SCRIPT = Path(__file__).resolve().with_name("fetch_spheron.py")
LLMSTATS_SCRIPT = Path(__file__).resolve().with_name("fetch_llmstats.py")
DEFAULT_LLM_JSON = Path(__file__).resolve().with_name("llm.json")
JSON_DUMP_KWARGS = {"indent": 2, "ensure_ascii": False}


class HelpOnErrorArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_help(sys.stderr)
        self.exit(2, f"\nError: {message}\n")


def to_percent(value: Any) -> float | None:
    """Scale an Artificial Analysis 0-1 fraction to a percentage.

    Scaling only: apply_score() rounds every score, from every source, onto the
    benchmark's grid, so nothing rounds on the way in.
    """
    if value is None:
        return None
    return float(value) * 100.0


def to_index(value: Any) -> float | None:
    """Pass an Artificial Analysis index/Elo through unscaled.

    AA-Omniscience (-100..100) and GDPval-AA Elo (human baseline = 1000) are not
    fractions, so to_percent() would multiply them by 100.
    """
    if value is None:
        return None
    return float(value)


def fmt_change_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_aa_value(value: Any) -> Any:
    # Treat zero values from Artificial Analysis as unset/null.
    if value == 0:
        return None
    return value


def normalize_context(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        raw = str(int(value))
    elif isinstance(value, str):
        raw = value.strip().lower()
    else:
        return None

    if not raw:
        return None

    match = re.fullmatch(r"([0-9]+)\s*([kmb]?)", raw)
    if not match:
        return None

    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "b":
        tokens = amount * 1_000_000_000
    elif unit == "m":
        tokens = amount * 1_000_000
    elif unit == "k":
        tokens = amount * 1_000
    else:
        # Bare values are ambiguous; treat large values as raw tokens,
        # otherwise as shorthand for kilotokens (e.g. 262 -> 262k).
        tokens = amount if amount >= 10_000 else amount * 1_000

    # Snap close binary-window aliases from AA pages (e.g. 262k -> 256k).
    if unit in {"", "k"}:
        tokens = snap_context_tokens(tokens)

    return format_context_tokens(tokens)


def print_changes_table(changes: list[tuple[str, str, Any, Any]]) -> None:
    headers = ("MODEL", "BENCHMARK", "VALUE", "UPDATED")
    rows: list[tuple[str, str, str, str]] = [
        (model, benchmark, fmt_change_value(prev_value), fmt_change_value(new_value))
        for model, benchmark, prev_value, new_value in changes
    ]

    data = [headers, *rows]
    widths = [max(len(row[i]) for row in data) for i in range(4)]
    fmt = f"{{:<{widths[0]}}}  {{:<{widths[1]}}}  {{:<{widths[2]}}}  {{:<{widths[3]}}}"

    print(fmt.format(*headers))
    if not rows:
        print(fmt.format("-", "-", "-", "-"))
        return
    for row in rows:
        print(fmt.format(*row))


SCORE_MAPPINGS: dict[str, tuple[tuple[str, ...], Callable[[Any], Any]]] = {
    "terminal_bench_hard": (("terminalbench_hard",), to_percent),
    "terminal_bench_2_1": (("terminalbench_v2_1",), to_percent),
    # AA reports the τ³ Banking domain under the version-less key "tau_banking".
    "tau3_bench_banking": (("tau_banking",), to_percent),
    "tau2_bench_telecom": (("tau2",), to_percent),
    "gdpval_aa": (("gdpval",), to_index),
    "aa_omniscience": (("omniscience",), to_index),
    "aa_omniscience_hallucination": (("omniscience_hallucination_rate",), to_percent),
    "aa_lcr": (("lcr",), to_percent),
    "critpt": (("critpt",), to_percent),
    "aime_2025": (("aime_25",), to_percent),
    "mmmu_pro": (("mmmu_pro",), to_percent),
    "gpqa_diamond": (("gpqa",), to_percent),
    "livecodebench": (("livecodebench",), to_percent),
    "scicode": (("scicode",), to_percent),
    "hle": (("hle",), to_percent),
    "aa_intelligence_index": (("artificial_analysis_intelligence_index",), lambda v: v),
    "aa_coding_index": (("artificial_analysis_coding_index",), lambda v: v),
}

# Per-score source pages, stamped into models[].scores_source alongside every
# score write. Read from the fetch_*.py constants (the fill_source_urls.py
# rule) and stored canonicalized like every URL in llm.json. AA and Hugging
# Face pages are per-model and resolved where the score is written; SWE Atlas
# and evals.report resolve per benchmark key below.
SWE_REBENCH_SOURCE_URL = canonical(fetch_swe_rebench.URL)
OSWORLD_SOURCE_URL = canonical(fetch_osworld.OSWORLD_SITE_URL)
LLMSTATS_SOURCE_URL = canonical(fetch_llmstats.LEADERBOARD_URL)
TOOLATHLON_SOURCE_URL = canonical(fetch_toolathlon.URL)
DEEPSWE_SOURCE_URL = canonical(fetch_deepswe.URL)
# The leaderboard page, not the JSON artifact it hydrates from.
DATACURVE_SOURCE_URL = canonical(fetch_datacurve.SITE_URL)
FRONTIERSWE_SOURCE_URL = canonical(fetch_frontierswe.URL)
# The leaderboard page, not the JSON it loads: the page is what a reader opens.
FRONTIERCODE_SOURCE_URL = canonical(fetch_frontiercode.LEADERBOARD_URL)
SWE_MARATHON_SOURCE_URL = canonical(fetch_swe_marathon.URL)
SWE_ATLAS_KEY_URLS = {
    key: canonical(fetch_swe_atlas.BASE_URL.format(track=track))
    for track, key in fetch_swe_atlas.TRACKS.items()
}
EVALS_REPORT_KEY_URLS = {
    key: canonical(fetch_evals_report.BASE_URL.format(slug=slug))
    for slug, key in fetch_evals_report.BENCHMARKS.items()
}


def aa_model_page_url(aa_slug: str) -> str:
    return canonical(artificialanalysis.MODEL_PAGE_URL.format(aa_slug))


def parse_args() -> argparse.Namespace:
    parser = HelpOnErrorArgumentParser(
        description="Update benchmark JSON scores by querying artificialanalysis.py and swe-rebench."
    )
    parser.add_argument(
        "json_file",
        nargs="?",
        default=str(DEFAULT_LLM_JSON),
        help='Path to JSON file to read/update (default: "./llm.json" next to this script)',
    )
    parser.add_argument(
        "--write",
        "-w",
        action="store_true",
        help="Write changes back to the input JSON file (default is dry-run).",
    )
    parser.add_argument(
        "--fill-source-urls",
        action="store_true",
        help=(
            "Backfill missing scores_source URLs instead of updating scores: "
            "for each source, in the usual order, attribute its page to every "
            "stored score its freshly fetched value equals, only where no URL "
            "is stored yet. Scores and dates are not touched. Respects the "
            "--skip-* flags and -w."
        ),
    )
    parser.add_argument(
        "--skip-aa",
        action="store_true",
        help="Skip fetching scores from artificialanalysis.py.",
    )
    parser.add_argument(
        "--skip-swe-rebench",
        action="store_true",
        help="Skip fetching scores from swe-rebench.",
    )
    parser.add_argument(
        "--skip-osworld",
        action="store_true",
        help="Skip fetching scores from osworld.",
    )
    parser.add_argument(
        "--skip-huggingface",
        action="store_true",
        help="Skip fetching scores and fallback parameter counts from huggingface.",
    )
    parser.add_argument(
        "--skip-toolathlon",
        action="store_true",
        help="Skip fetching scores from toolathlon.",
    )
    parser.add_argument(
        "--skip-deepswe",
        action="store_true",
        help="Skip fetching scores from deepswe.",
    )
    parser.add_argument(
        "--skip-datacurve",
        action="store_true",
        help="Skip fetching DeepSWE scores from deepswe.datacurve.ai.",
    )
    parser.add_argument(
        "--skip-frontierswe",
        action="store_true",
        help="Skip fetching scores from frontierswe.",
    )
    parser.add_argument(
        "--skip-frontiercode",
        action="store_true",
        help="Skip fetching scores from cognition.com/frontiercode.",
    )
    parser.add_argument(
        "--skip-swe-atlas",
        action="store_true",
        help="Skip fetching scores from SWE Atlas.",
    )
    parser.add_argument(
        "--skip-evals-report",
        action="store_true",
        help="Skip fetching scores from evals.report.",
    )
    parser.add_argument(
        "--skip-swe-marathon",
        action="store_true",
        help="Skip fetching scores from swe-marathon.org.",
    )
    parser.add_argument(
        "--skip-spheron",
        action="store_true",
        help="Skip fetching VRAM estimates from Spheron.",
    )
    parser.add_argument(
        "--skip-llmstats",
        action="store_true",
        help="Skip fetching scores from llm-stats.com.",
    )
    return parser.parse_args()


def unique_names(models: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for model in models:
        name = model.get("name")
        if isinstance(name, str) and name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def build_list_models_cmd(aa_script: Path) -> list[str]:
    return [sys.executable, str(aa_script), "--list-models"]


def build_fetch_data_cmd(aa_script: Path, slugs: list[str]) -> list[str]:
    cmd = [sys.executable, str(aa_script), "-o", "json"]
    for slug in slugs:
        cmd.extend(["-m", slug])
    return cmd


def build_fetch_swe_rebench_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--all-models", "--format", "json"]


def fetch_available_slugs(aa_script: Path) -> set[str]:
    cmd = build_list_models_cmd(aa_script)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"artificialanalysis.py --list-models failed ({proc.returncode}): {proc.stderr.strip()}")
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def fetch_aa_data(aa_script: Path, slugs: list[str]) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_data_cmd(aa_script, slugs)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"artificialanalysis.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    data = payload.get("data", [])
    by_slug: dict[str, dict[str, Any]] = {}
    for row in data:
        slug = row.get("slug")
        if isinstance(slug, str) and slug:
            by_slug[slug] = row
    return by_slug


def resolve_aa_slugs(
    slugs: list[str], available_slugs: set[str], mapping_path: Path
) -> dict[str, list[str]]:
    """AA slugs to read per llm.json model, highest priority first.

    Usually one slug. A mapping entry may name several: Artificial Analysis
    sometimes tracks the same model under more than one slug, each carrying a
    different slice of the benchmarks, and every slug listed is read. A model's
    own slug leads unless the entry places it somewhere else.
    """
    llm_to_aa = load_llm_to_aa_slugs(mapping_path)
    resolved: dict[str, list[str]] = {}
    for slug in slugs:
        candidates = [
            mapped for mapped in llm_to_aa.get(slug, []) if mapped in available_slugs
        ]
        if slug in available_slugs and slug not in candidates:
            candidates.insert(0, slug)
        if candidates:
            resolved[slug] = candidates
    return resolved


def aa_value_missing(value: Any) -> bool:
    # AA reports an untested benchmark as null, and on some rows as 0 -- the
    # same reading normalize_aa_value() applies when the score is written.
    if isinstance(value, bool):
        return False
    return value is None or value == "" or value == 0


def merge_aa_models(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold the AA records of one model into one, earlier records winning.

    The leading record decides every value it measured; the rest only fill the
    gaps it leaves, per benchmark rather than per record. "_eval_origins"
    remembers which record's AA slug supplied each evaluation, so the source
    URL can point at the model page the number actually sits on.
    """
    merged = dict(records[0])
    evaluations = dict(merged.get("evaluations") or {})
    origins = {key: merged.get("slug") for key in evaluations}
    for record in records[1:]:
        for key, value in record.items():
            if key == "evaluations":
                continue
            if aa_value_missing(merged.get(key)) and not aa_value_missing(value):
                merged[key] = value
        for key, value in (record.get("evaluations") or {}).items():
            if aa_value_missing(evaluations.get(key)) and not aa_value_missing(value):
                evaluations[key] = value
                origins[key] = record.get("slug")
    if evaluations:
        merged["evaluations"] = evaluations
    merged["_eval_origins"] = origins
    return merged


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def keep_best_row(
    by_slug: dict[str, dict[str, Any]],
    slug: str,
    row: dict[str, Any],
    score_key: str,
) -> None:
    """Keep the higher-scoring row when several source rows fold onto one slug.

    The mapping files fold a model's variants -- a base row and its "[high]"
    sibling, or one label spelled two ways -- onto a single llm.json slug. Which
    row is read must not depend on the leaderboard's own ordering, so the best
    reported run wins, the rule the swe-atlas, evals.report and SWE-Marathon
    ingests already apply.
    """
    current = by_slug.get(slug)
    if current is None:
        by_slug[slug] = row
        return
    new_score = row.get(score_key)
    if not is_number(new_score):
        return
    old_score = current.get(score_key)
    if not is_number(old_score) or new_score > old_score:
        by_slug[slug] = row


def apply_score(
    doc: dict[str, Any],
    model: dict[str, Any],
    slug: str,
    key: str,
    new_value: Any,
    url: str,
    changes: list[tuple[str, str, Any, Any]],
    *,
    fill_only: bool = False,
    fill_urls_only: bool = False,
) -> int:
    """Write one benchmark score with its date and source-page bookkeeping.

    Returns the number of updates made (0 or 1). Shared by every source so the
    write rules live in one place:

      * round onto the benchmark's grid first, so a leaderboard that reports
        two decimals and one that reports one cannot disagree about a score the
        site prints identically either way;
      * never overwrite an existing non-null value with null;
      * fill_only: only fill nulls, never overwrite (the low-trust rule the
        Hugging Face and llm-stats aggregates follow);
      * fill_urls_only (--fill-source-urls): scores and dates stay untouched;
        the URL is stamped only where none is stored yet and this source's
        fetched value equals the stored score, so the first source in the
        usual update order claims a score it could have produced. Rounding
        comes first here too, or a stored score could never match the raw
        number the source that wrote it hands back today.
    """
    scores = model.setdefault("scores", {})
    if not isinstance(scores, dict):
        return 0
    new_value = round_score(doc, key, new_value)
    old_value = scores.get(key)

    if fill_urls_only:
        if new_value is None or old_value != new_value:
            return 0
        existing = model.get("scores_source")
        if isinstance(existing, dict) and existing.get(key) is not None:
            return 0
        stamp_score_source(model, key, url)
        changes.append((slug, key, old_value, url))
        return 1

    if fill_only and old_value is not None:
        return 0
    # Never overwrite an existing non-null value with null.
    if old_value is not None and new_value is None:
        return 0
    if old_value == new_value:
        return 0
    scores[key] = new_value
    stamp_score_updated(model, key)
    stamp_score_source(model, key, url)
    changes.append((slug, key, old_value, new_value))
    return 1


def ensure_scores_source(model: dict[str, Any], benchmark_keys: list[str]) -> None:
    """Materialize the full-key scores_source map, placed after scores_updated.

    Mirrors the scores/scores_updated convention: every benchmark key present,
    null until a source is attributed. Existing values are preserved.
    """
    existing = model.get("scores_source")
    filled = {
        key: (existing.get(key) if isinstance(existing, dict) else None)
        for key in benchmark_keys
    }
    if "scores_source" in model:
        model["scores_source"] = filled
        return
    items = list(model.items())
    model.clear()
    for key, value in items:
        model[key] = value
        if key == "scores_updated":
            model["scores_source"] = filled
    if "scores_source" not in model:
        model["scores_source"] = filled


def fetch_swe_rebench_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_swe_rebench_cmd(script)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_swe_rebench.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected swe_rebench JSON format: expected a list")

    rebench_to_slug = load_rebench_to_slug_mapping(mapping_path)
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        rebench_name = row.get("model")
        if not isinstance(rebench_name, str) or not rebench_name:
            continue
        slug = rebench_to_slug.get(rebench_name)
        if not slug:
            continue
        keep_best_row(by_slug, slug, row, "resolved_rate")
    return by_slug


def update_scores(
    doc: dict[str, Any],
    by_slug: dict[str, dict[str, Any]],
    fill_urls_only: bool = False,
) -> tuple[int, int, set[str], list[tuple[str, str, Any, Any]]]:
    models = doc.get("models", [])
    matched = 0
    updated = 0
    seen_eval_keys: set[str] = set()
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        if not isinstance(slug, str) or not slug:
            continue
        aa_model = by_slug.get(slug)
        if aa_model is None:
            continue

        matched += 1
        evaluations = aa_model.get("evaluations", {})
        if not isinstance(evaluations, dict):
            continue
        seen_eval_keys.update(k for k in evaluations.keys() if isinstance(k, str))

        if not fill_urls_only:
            old_context = model.get("context")
            new_context = normalize_context(aa_model.get("context"))
            if not (old_context is not None and new_context is None) and old_context != new_context:
                model["context"] = new_context
                updated += 1
                changes.append((slug, "context", old_context, new_context))

            # Params are filled, never refreshed: AA reports measured counts, so a
            # refresh would overwrite curated advertised sizes (E2B -> 5.1B-A2.3B).
            old_params = model.get("params")
            if not old_params:
                new_params = normalize_params(aa_model.get("params"))
                if new_params:
                    model["params"] = new_params
                    updated += 1
                    changes.append((slug, "params", old_params, new_params))

        origins = aa_model.get("_eval_origins") or {}
        for llm_key, (aa_keys, transform) in SCORE_MAPPINGS.items():
            aa_value = None
            aa_key_used = None
            for aa_key in aa_keys:
                if aa_key in evaluations and evaluations.get(aa_key) is not None:
                    aa_value = evaluations.get(aa_key)
                    aa_key_used = aa_key
                    break
            new_value = transform(normalize_aa_value(aa_value))
            origin_slug = origins.get(aa_key_used) or aa_model.get("slug")
            url = aa_model_page_url(origin_slug)
            updated += apply_score(
                doc, model, slug, llm_key, new_value, url, changes,
                fill_urls_only=fill_urls_only,
            )

    return matched, updated, seen_eval_keys, changes


def update_swe_rebench_scores(
    doc: dict[str, Any],
    by_slug: dict[str, dict[str, Any]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    models = doc.get("models", [])
    matched = 0
    updated = 0
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        if not isinstance(slug, str) or not slug:
            continue
        rebench_model = by_slug.get(slug)
        if rebench_model is None:
            continue

        matched += 1
        updated += apply_score(
            doc, model, slug, "swe_rebench", rebench_model.get("resolved_rate"),
            SWE_REBENCH_SOURCE_URL, changes, fill_urls_only=fill_urls_only,
        )

    return matched, updated, changes


def build_fetch_osworld_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_osworld_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_osworld_cmd(script)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_osworld.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected osworld JSON format: expected a list")

    osworld_to_slug = load_osworld_to_slug_mapping(mapping_path)
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        osworld_name = row.get("model")
        if not isinstance(osworld_name, str) or not osworld_name:
            continue
        slug = osworld_to_slug.get(osworld_name)
        if not slug:
            continue
        keep_best_row(by_slug, slug, row, "success_rate")
    return by_slug


def update_osworld_scores(
    doc: dict[str, Any],
    by_slug: dict[str, dict[str, Any]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    models = doc.get("models", [])
    matched = 0
    updated = 0
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        if not isinstance(slug, str) or not slug:
            continue
        osworld_model = by_slug.get(slug)
        if osworld_model is None:
            continue

        matched += 1
        updated += apply_score(
            doc, model, slug, "osworld_verified", osworld_model.get("success_rate"),
            OSWORLD_SOURCE_URL, changes, fill_urls_only=fill_urls_only,
        )

    return matched, updated, changes


def build_fetch_huggingface_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--all-models", "--format", "json"]


def fetch_huggingface_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_huggingface_cmd(script)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_huggingface.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected huggingface JSON format: expected a list")

    hf_to_key = load_hf_to_key_mapping(mapping_path)
    # slug -> {benchmark_key -> (best score, model-card URL it came from)}
    by_slug: dict[str, dict[str, tuple[Any, str]]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        slug = row.get("model")
        repo = row.get("repo")
        scores = row.get("scores")
        if not isinstance(slug, str) or not slug or not isinstance(scores, dict):
            continue
        if not isinstance(repo, str) or not repo:
            continue
        url = canonical(f"{fetch_huggingface.HF_BASE}/{repo}")
        mapped: dict[str, tuple[Any, str]] = {}
        for label, value in scores.items():
            key = hf_to_key.get(label)
            if not key or value is None:
                continue
            # Several leaderboard labels can alias one llm.json benchmark; the
            # best reported run wins rather than whichever label came first.
            old = mapped.get(key)
            if old is None or not is_number(old[0]) or (is_number(value) and value > old[0]):
                mapped[key] = (value, url)
        if not mapped:
            continue
        merged = by_slug.setdefault(slug, {})
        for key, pair in mapped.items():
            old = merged.get(key)
            if old is None or not is_number(old[0]) or (is_number(pair[0]) and pair[0] > old[0]):
                merged[key] = pair
    return by_slug


def update_huggingface_scores(
    doc: dict[str, Any],
    by_slug: dict[str, dict[str, tuple[Any, str]]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    models = doc.get("models", [])
    matched = 0
    updated = 0
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        if not isinstance(slug, str) or not slug:
            continue
        hf_scores = by_slug.get(slug)
        if not hf_scores:
            continue

        matched += 1
        for benchmark_key, (new_value, url) in hf_scores.items():
            # HF self-reports are lowest-trust: only fill nulls, never overwrite.
            updated += apply_score(
                doc, model, slug, benchmark_key, new_value, url, changes,
                fill_only=True, fill_urls_only=fill_urls_only,
            )

    return matched, updated, changes


def fill_missing_params_from_huggingface(
    doc: dict[str, Any]
) -> tuple[int, list[tuple[str, str, Any, Any]]]:
    """Last resort for params: models AA has no page for (or no count on it).

    Hugging Face carries no active-parameter count, so MoE models land here as a
    total only ("117B") and the "-A..." half stays a manual edit.
    """
    models = doc.get("models", [])
    filled = 0
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        old_params = model.get("params")
        if not isinstance(slug, str) or not slug or old_params:
            continue

        new_params = fetch_hf_params(model.get("url"))
        if not new_params:
            continue

        model["params"] = new_params
        filled += 1
        changes.append((slug, "params", old_params, new_params))

    return filled, changes


def build_fetch_toolathlon_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_toolathlon_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_toolathlon_cmd(script)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_toolathlon.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected toolathlon JSON format: expected a list")

    toolathlon_to_slug = load_toolathlon_to_slug_mapping(mapping_path)
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        toolathlon_name = row.get("model")
        if not isinstance(toolathlon_name, str) or not toolathlon_name:
            continue
        slug = toolathlon_to_slug.get(toolathlon_name)
        if not slug:
            continue
        keep_best_row(by_slug, slug, row, "score")
    return by_slug


def update_toolathlon_scores(
    doc: dict[str, Any],
    by_slug: dict[str, dict[str, Any]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    models = doc.get("models", [])
    matched = 0
    updated = 0
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        if not isinstance(slug, str) or not slug:
            continue
        toolathlon_model = by_slug.get(slug)
        if toolathlon_model is None:
            continue

        matched += 1
        updated += apply_score(
            doc, model, slug, "toolathlon", toolathlon_model.get("score"),
            TOOLATHLON_SOURCE_URL, changes, fill_urls_only=fill_urls_only,
        )

    return matched, updated, changes


def build_fetch_deepswe_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_deepswe_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_deepswe_cmd(script)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_deepswe.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected deepswe JSON format: expected a list")

    deepswe_to_slug = load_deepswe_to_slug_mapping(mapping_path)
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        deepswe_name = row.get("model")
        if not isinstance(deepswe_name, str) or not deepswe_name:
            continue
        slug = deepswe_to_slug.get(deepswe_name)
        if not slug:
            continue
        keep_best_row(by_slug, slug, row, "score")
    return by_slug


def update_deepswe_scores(
    doc: dict[str, Any],
    by_slug: dict[str, dict[str, Any]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    models = doc.get("models", [])
    matched = 0
    updated = 0
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        if not isinstance(slug, str) or not slug:
            continue
        deepswe_model = by_slug.get(slug)
        if deepswe_model is None:
            continue

        matched += 1
        updated += apply_score(
            doc, model, slug, "deepswe", deepswe_model.get("score"),
            DEEPSWE_SOURCE_URL, changes, fill_urls_only=fill_urls_only,
        )

    return matched, updated, changes


def build_fetch_frontierswe_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_frontierswe_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_frontierswe_cmd(script)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_frontierswe.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected frontierswe JSON format: expected a list")

    frontierswe_to_slug = load_frontierswe_to_slug_mapping(mapping_path)
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        frontierswe_name = row.get("model")
        if not isinstance(frontierswe_name, str) or not frontierswe_name:
            continue
        slug = frontierswe_to_slug.get(frontierswe_name)
        if not slug:
            continue
        keep_best_row(by_slug, slug, row, "score")
    return by_slug


def update_frontierswe_scores(
    doc: dict[str, Any],
    by_slug: dict[str, dict[str, Any]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    models = doc.get("models", [])
    matched = 0
    updated = 0
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        if not isinstance(slug, str) or not slug:
            continue
        frontierswe_model = by_slug.get(slug)
        if frontierswe_model is None:
            continue

        matched += 1
        updated += apply_score(
            doc, model, slug, "frontierswe", frontierswe_model.get("score"),
            FRONTIERSWE_SOURCE_URL, changes, fill_urls_only=fill_urls_only,
        )

    return matched, updated, changes


def keep_newest_frontiercode_row(
    by_slug: dict[str, dict[str, Any]], slug: str, row: dict[str, Any]
) -> None:
    """Newer benchmark revision wins; within one revision, the better score wins.

    fetch_frontiercode.py already resolves a model listed in several revisions,
    but two *different* leaderboard names can fold onto one llm.json slug across
    revisions (a renamed release, a variant label). The revision has to decide
    there too -- keep_best_row's best-run rule would otherwise let a retired
    revision's higher number outrank the current re-run.
    """
    current = by_slug.get(slug)
    if current is None:
        by_slug[slug] = row
        return

    new_rank = fetch_frontiercode.revision_rank(row.get("revision") or "")
    old_rank = fetch_frontiercode.revision_rank(current.get("revision") or "")
    if new_rank == old_rank:
        keep_best_row(by_slug, slug, row, "score")
        return
    if new_rank > old_rank and is_number(row.get("score")):
        by_slug[slug] = row


def build_fetch_datacurve_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json", "--all-configs"]


def fetch_datacurve_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    """DeepSWE scores from the benchmark's own leaderboard.

    Shares the benchlm mapping file: both sources label a run
    "<model>[<effort>]", so a configuration reviewed once is mapped for both.
    Every configuration is fetched (--all-configs) and the best one that maps to
    a slug wins, the same rule keep_best_row applies to harness variants.
    """
    cmd = build_fetch_datacurve_cmd(script)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_datacurve.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected datacurve JSON format: expected a list")

    deepswe_to_slug = load_deepswe_to_slug_mapping(mapping_path)
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        datacurve_name = row.get("model")
        if not isinstance(datacurve_name, str) or not datacurve_name:
            continue
        slug = deepswe_to_slug.get(datacurve_name)
        if not slug:
            continue
        keep_best_row(by_slug, slug, row, "score")
    return by_slug


def update_datacurve_scores(
    doc: dict[str, Any],
    by_slug: dict[str, dict[str, Any]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    models = doc.get("models", [])
    matched = 0
    updated = 0
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        if not isinstance(slug, str) or not slug:
            continue
        datacurve_model = by_slug.get(slug)
        if datacurve_model is None:
            continue

        matched += 1
        updated += apply_score(
            doc, model, slug, "deepswe", datacurve_model.get("score"),
            DATACURVE_SOURCE_URL, changes, fill_urls_only=fill_urls_only,
        )

    return matched, updated, changes


def build_fetch_frontiercode_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_frontiercode_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_frontiercode_cmd(script)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_frontiercode.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected frontiercode JSON format: expected a list")

    frontiercode_to_slug = load_frontiercode_to_slug_mapping(mapping_path)
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        frontiercode_name = row.get("model")
        if not isinstance(frontiercode_name, str) or not frontiercode_name:
            continue
        slug = frontiercode_to_slug.get(frontiercode_name)
        if not slug:
            continue
        keep_newest_frontiercode_row(by_slug, slug, row)
    return by_slug


def update_frontiercode_scores(
    doc: dict[str, Any],
    by_slug: dict[str, dict[str, Any]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    models = doc.get("models", [])
    matched = 0
    updated = 0
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        if not isinstance(slug, str) or not slug:
            continue
        frontiercode_model = by_slug.get(slug)
        if frontiercode_model is None:
            continue

        matched += 1
        updated += apply_score(
            doc, model, slug, "frontiercode", frontiercode_model.get("score"),
            FRONTIERCODE_SOURCE_URL, changes, fill_urls_only=fill_urls_only,
        )

    return matched, updated, changes


def build_fetch_swe_atlas_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--track", "all", "--format", "json"]


def fetch_swe_atlas_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_swe_atlas_cmd(script)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_swe_atlas.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected swe_atlas JSON format: expected a list")

    swe_atlas_to_slug = load_swe_atlas_to_slug_mapping(mapping_path)
    # slug -> {benchmark_key -> best score across harness variants}
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        name = row.get("model")
        key = row.get("key")
        score = row.get("score")
        if not isinstance(name, str) or not isinstance(key, str):
            continue
        if not isinstance(score, (int, float)):
            continue
        slug = swe_atlas_to_slug.get(name)
        if not slug:
            continue
        scores = by_slug.setdefault(slug, {})
        # A model may appear under several harnesses per track; keep the best.
        if key not in scores or score > scores[key]:
            scores[key] = score
    return by_slug


def update_swe_atlas_scores(
    doc: dict[str, Any],
    by_slug: dict[str, dict[str, Any]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    models = doc.get("models", [])
    matched = 0
    updated = 0
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        if not isinstance(slug, str) or not slug:
            continue
        swe_atlas_scores = by_slug.get(slug)
        if not swe_atlas_scores:
            continue

        matched += 1
        for benchmark_key, new_value in swe_atlas_scores.items():
            updated += apply_score(
                doc, model, slug, benchmark_key, new_value,
                SWE_ATLAS_KEY_URLS[benchmark_key], changes,
                fill_urls_only=fill_urls_only,
            )

    return matched, updated, changes


def build_fetch_evals_report_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--benchmark", "all", "--format", "json"]


def fetch_evals_report_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_evals_report_cmd(script)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_evals_report.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected evals_report JSON format: expected a list")

    evals_report_to_slug = load_evals_report_to_slug_mapping(mapping_path)
    # slug -> {benchmark_key -> best score across reported runs}
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        name = row.get("model")
        key = row.get("key")
        score = row.get("score")
        if not isinstance(name, str) or not isinstance(key, str):
            continue
        if not isinstance(score, (int, float)):
            continue
        slug = evals_report_to_slug.get(name)
        if not slug:
            continue
        scores = by_slug.setdefault(slug, {})
        # A model may have several reported runs per benchmark; keep the best.
        if key not in scores or score > scores[key]:
            scores[key] = score
    return by_slug


def update_evals_report_scores(
    doc: dict[str, Any],
    by_slug: dict[str, dict[str, Any]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    models = doc.get("models", [])
    matched = 0
    updated = 0
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        if not isinstance(slug, str) or not slug:
            continue
        evals_report_scores = by_slug.get(slug)
        if not evals_report_scores:
            continue

        matched += 1
        for benchmark_key, new_value in evals_report_scores.items():
            updated += apply_score(
                doc, model, slug, benchmark_key, new_value,
                EVALS_REPORT_KEY_URLS[benchmark_key], changes,
                fill_urls_only=fill_urls_only,
            )

    return matched, updated, changes


def build_fetch_swe_marathon_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_swe_marathon_data(script: Path, mapping_path: Path) -> dict[str, Any]:
    """Return slug -> best SWE-Marathon pass@1 across the model's scaffolds.

    The leaderboard lists one row per (model, scaffold) pair and the score is
    scaffold-dependent, so the model's best harness result is what lands in
    llm.json -- same "best reported run wins" rule the evals.report ingest uses.
    """
    cmd = build_fetch_swe_marathon_cmd(script)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_swe_marathon.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected swe_marathon JSON format: expected a list")

    swe_marathon_to_slug = load_swe_marathon_to_slug_mapping(mapping_path)
    by_slug: dict[str, Any] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        name = row.get("model")
        score = row.get("score")
        if not isinstance(name, str) or isinstance(score, bool):
            continue
        if not isinstance(score, (int, float)):
            continue
        slug = swe_marathon_to_slug.get(name)
        if not slug:
            continue
        if slug not in by_slug or score > by_slug[slug]:
            by_slug[slug] = score
    return by_slug


def update_swe_marathon_scores(
    doc: dict[str, Any],
    by_slug: dict[str, Any],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    models = doc.get("models", [])
    matched = 0
    updated = 0
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        if not isinstance(slug, str) or not slug:
            continue
        if slug not in by_slug:
            continue

        matched += 1
        updated += apply_score(
            doc, model, slug, "swe_marathon", by_slug[slug],
            SWE_MARATHON_SOURCE_URL, changes, fill_urls_only=fill_urls_only,
        )

    return matched, updated, changes


def build_fetch_spheron_cmd(script: Path, paths: list[str]) -> list[str]:
    cmd = [sys.executable, str(script), "--format", "json"]
    for path in paths:
        cmd.extend(["--model", path])
    return cmd


def fetch_spheron_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    spheron_to_slug = load_spheron_to_slug_mapping(mapping_path)
    paths = sorted(spheron_to_slug)
    if not paths:
        return {}

    cmd = build_fetch_spheron_cmd(script, paths)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_spheron.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected spheron JSON format: expected a list")

    # slug -> {quant -> vram GB, source -> {quant -> per-model page}}
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        name = row.get("model")
        if not isinstance(name, str):
            continue
        slug = spheron_to_slug.get(name)
        if not slug:
            continue
        vram = by_slug.setdefault(
            slug,
            {"fp16": None, "int8": None, "int4": None, "source": {}},
        )
        source = row.get("source")
        for quant, source_key in (
            ("fp16", "vram_fp16"),
            ("int8", "vram_int8"),
            ("int4", "vram_int4"),
        ):
            value = row.get(source_key)
            old = vram.get(quant)
            # Two spheron paths can fold onto one slug (a repo and its dated
            # revision). VRAM is a requirement rather than a score, so the
            # larger estimate wins: it is the one the model actually fits in.
            if not is_number(old) or (is_number(value) and value > old):
                vram[quant] = value
                if isinstance(source, str) and source:
                    vram["source"][quant] = source
    return by_slug


def update_spheron_vram(
    doc: dict[str, Any], by_slug: dict[str, dict[str, Any]]
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    models = doc.get("models", [])
    matched = 0
    updated = 0
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        if not isinstance(slug, str) or not slug:
            continue
        vram_data = by_slug.get(slug)
        if not vram_data:
            continue

        matched += 1
        vram = model.setdefault("vram", {})
        if not isinstance(vram, dict):
            continue
        vram_source = model.get("vram_source")
        if not isinstance(vram_source, dict):
            vram_source = None

        for quant in ("fp16", "int8", "int4"):
            new_value = vram_data.get(quant)
            old_value = vram.get(quant)
            # Never overwrite an existing non-null value with null.
            if old_value is not None and new_value is None:
                continue
            if old_value != new_value:
                vram[quant] = new_value
                updated += 1
                changes.append((slug, f"vram_{quant}", old_value, new_value))
            source = vram_data.get("source", {}).get(quant)
            if new_value is not None and isinstance(source, str) and source:
                if vram_source is None:
                    vram_source = {}
                    model["vram_source"] = vram_source
                vram_source[quant] = source

    return matched, updated, changes


def build_fetch_llmstats_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_llmstats_data(
    script: Path, model_mapping_path: Path, benchmark_mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_llmstats_cmd(script)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_llmstats.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected llmstats JSON format: expected a list")

    llmstats_to_slug = load_llmstats_to_slug_mapping(model_mapping_path)
    label_to_key = load_llmstats_benchmark_to_key_mapping(benchmark_mapping_path)
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        llmstats_name = row.get("model")
        scores = row.get("scores")
        if not isinstance(llmstats_name, str) or not llmstats_name or not isinstance(scores, dict):
            continue
        slug = llmstats_to_slug.get(llmstats_name)
        if not slug:
            continue
        mapped: dict[str, Any] = {}
        for label, value in scores.items():
            key = label_to_key.get(label)
            if not key or value is None:
                continue
            # llm-stats reports 0-1 scores; store as percentages like other sources.
            # Several labels can alias one benchmark; the best run wins.
            percent = to_percent(value)
            old = mapped.get(key)
            if not is_number(old) or (is_number(percent) and percent > old):
                mapped[key] = percent
        if not mapped:
            continue
        # Several llm-stats ids fold onto one slug (a base id and its "-high"
        # sibling, a dated re-release). Merge them per benchmark, best run wins,
        # so the payload's ordering cannot decide the number.
        merged = by_slug.setdefault(slug, {})
        for key, value in mapped.items():
            old = merged.get(key)
            if not is_number(old) or (is_number(value) and value > old):
                merged[key] = value
    return by_slug


def update_llmstats_scores(
    doc: dict[str, Any],
    by_slug: dict[str, dict[str, Any]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    models = doc.get("models", [])
    matched = 0
    updated = 0
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        if not isinstance(slug, str) or not slug:
            continue
        llmstats_scores = by_slug.get(slug)
        if not llmstats_scores:
            continue

        matched += 1
        for benchmark_key, new_value in llmstats_scores.items():
            # General aggregator: keep aa (and every named source) leading.
            # Only fill nulls, never overwrite an existing value.
            updated += apply_score(
                doc, model, slug, benchmark_key, new_value,
                LLMSTATS_SOURCE_URL, changes,
                fill_only=True, fill_urls_only=fill_urls_only,
            )

    return matched, updated, changes


def main() -> int:
    args = parse_args()
    if args.fill_source_urls:
        # Spheron carries VRAM estimates, not benchmark scores; the URL
        # backfill has nothing to attribute there.
        args.skip_spheron = True
    llm_path = Path(args.json_file)
    aa_path = AA_SCRIPT
    swe_rebench_path = SWE_REBENCH_SCRIPT
    osworld_path = OSWORLD_SCRIPT
    huggingface_path = HF_SCRIPT
    deepswe_path = DEEPSWE_SCRIPT
    datacurve_path = DATACURVE_SCRIPT
    toolathlon_path = TOOLATHLON_SCRIPT
    frontierswe_path = FRONTIERSWE_SCRIPT
    frontiercode_path = FRONTIERCODE_SCRIPT
    swe_atlas_path = SWE_ATLAS_SCRIPT
    evals_report_path = EVALS_REPORT_SCRIPT
    swe_marathon_path = SWE_MARATHON_SCRIPT
    spheron_path = SPHERON_SCRIPT
    llmstats_path = LLMSTATS_SCRIPT
    swe_rebench_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-rebench-to-artificialanalysis.json"
    )
    osworld_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-osworld-to-artificialanalysis.json"
    )
    huggingface_mapping_path = Path(__file__).resolve().with_name(
        "huggingface-benchmark-name-mapping.json"
    )
    deepswe_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-deepswe-to-artificialanalysis.json"
    )
    toolathlon_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-toolathlon-to-artificialanalysis.json"
    )
    frontierswe_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-frontierswe-to-artificialanalysis.json"
    )
    frontiercode_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-frontiercode-to-artificialanalysis.json"
    )
    swe_atlas_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-swe-atlas-to-artificialanalysis.json"
    )
    evals_report_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-evals-report-to-artificialanalysis.json"
    )
    swe_marathon_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-swe-marathon-to-artificialanalysis.json"
    )
    spheron_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-spheron-to-artificialanalysis.json"
    )
    llmstats_model_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-llmstats-to-artificialanalysis.json"
    )
    aa_model_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-llm-to-artificialanalysis.json"
    )
    llmstats_benchmark_mapping_path = Path(__file__).resolve().with_name(
        "llmstats-benchmark-name-mapping.json"
    )

    doc = json.loads(llm_path.read_text(encoding="utf-8"))
    models = doc.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError("Invalid JSON: models must be a list")

    if args.fill_source_urls:
        # Materialize the map for every model so the file ends up with the
        # same full-key null-placeholder convention scores/scores_updated use.
        benchmark_keys = list((doc.get("benchmarks") or {}).keys())
        for model in models:
            if isinstance(model, dict):
                ensure_scores_source(model, benchmark_keys)

    slugs = unique_names(models)
    spheron_paths = sorted(load_spheron_to_slug_mapping(spheron_mapping_path))
    print("commands:")
    if not args.skip_aa:
        print(f"  - {shlex.join(build_list_models_cmd(aa_path))}")
    if not args.skip_swe_rebench:
        print(f"  - {shlex.join(build_fetch_swe_rebench_cmd(swe_rebench_path))}")
    if not args.skip_osworld:
        print(f"  - {shlex.join(build_fetch_osworld_cmd(osworld_path))}")
    if not args.skip_llmstats:
        print(f"  - {shlex.join(build_fetch_llmstats_cmd(llmstats_path))}")
    if not args.skip_huggingface:
        print(f"  - {shlex.join(build_fetch_huggingface_cmd(huggingface_path))}")
    if not args.skip_toolathlon:
        print(f"  - {shlex.join(build_fetch_toolathlon_cmd(toolathlon_path))}")
    if not args.skip_deepswe:
        print(f"  - {shlex.join(build_fetch_deepswe_cmd(deepswe_path))}")
    if not args.skip_datacurve:
        print(f"  - {shlex.join(build_fetch_datacurve_cmd(datacurve_path))}")
    if not args.skip_frontierswe:
        print(f"  - {shlex.join(build_fetch_frontierswe_cmd(frontierswe_path))}")
    if not args.skip_frontiercode:
        print(f"  - {shlex.join(build_fetch_frontiercode_cmd(frontiercode_path))}")
    if not args.skip_swe_atlas:
        print(f"  - {shlex.join(build_fetch_swe_atlas_cmd(swe_atlas_path))}")
    if not args.skip_evals_report:
        print(f"  - {shlex.join(build_fetch_evals_report_cmd(evals_report_path))}")
    if not args.skip_swe_marathon:
        print(f"  - {shlex.join(build_fetch_swe_marathon_cmd(swe_marathon_path))}")
    if not args.skip_spheron and spheron_paths:
        print(f"  - {shlex.join(build_fetch_spheron_cmd(spheron_path, spheron_paths))}")

    changes: list[tuple[str, str, Any, Any]] = []
    available_slugs: set[str] = set()
    existing_slugs: list[str] = []
    aa_slug_by_model: dict[str, list[str]] = {}
    by_slug: dict[str, dict[str, Any]] = {}
    matched = 0
    aa_updated = 0
    seen_eval_keys: set[str] = set()

    if not args.skip_aa:
        available_slugs = fetch_available_slugs(aa_path)
        aa_slug_by_model = resolve_aa_slugs(slugs, available_slugs, aa_model_mapping_path)
        existing_slugs = list(
            dict.fromkeys(
                aa_slug
                for aa_slugs in aa_slug_by_model.values()
                for aa_slug in aa_slugs
            )
        )
        print(f"  - {shlex.join(build_fetch_data_cmd(aa_path, existing_slugs))}")
    print()

    if not args.skip_aa:
        by_aa_slug = fetch_aa_data(aa_path, existing_slugs)
        by_slug = {}
        for slug, aa_slugs in aa_slug_by_model.items():
            records = [by_aa_slug[aa_slug] for aa_slug in aa_slugs if aa_slug in by_aa_slug]
            if records:
                by_slug[slug] = merge_aa_models(records)
        matched, aa_updated, seen_eval_keys, aa_changes = update_scores(
            doc, by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(aa_changes)

    swe_rebench_by_slug: dict[str, dict[str, Any]] = {}
    swe_matched = 0
    swe_updated = 0
    if not args.skip_swe_rebench:
        swe_rebench_by_slug = fetch_swe_rebench_data(swe_rebench_path, swe_rebench_mapping_path)
        swe_matched, swe_updated, swe_changes = update_swe_rebench_scores(
            doc, swe_rebench_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(swe_changes)

    osworld_by_slug: dict[str, dict[str, Any]] = {}
    osworld_matched = 0
    osworld_updated = 0
    if not args.skip_osworld:
        osworld_by_slug = fetch_osworld_data(osworld_path, osworld_mapping_path)
        osworld_matched, osworld_updated, osworld_changes = update_osworld_scores(
            doc, osworld_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(osworld_changes)

    llmstats_by_slug: dict[str, dict[str, Any]] = {}
    llmstats_matched = 0
    llmstats_updated = 0
    if not args.skip_llmstats:
        llmstats_by_slug = fetch_llmstats_data(
            llmstats_path, llmstats_model_mapping_path, llmstats_benchmark_mapping_path
        )
        llmstats_matched, llmstats_updated, llmstats_changes = update_llmstats_scores(
            doc, llmstats_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(llmstats_changes)

    huggingface_by_slug: dict[str, dict[str, Any]] = {}
    hf_matched = 0
    hf_updated = 0
    hf_params_filled = 0
    if not args.skip_huggingface:
        huggingface_by_slug = fetch_huggingface_data(huggingface_path, huggingface_mapping_path)
        hf_matched, hf_updated, hf_changes = update_huggingface_scores(
            doc, huggingface_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(hf_changes)
        # Runs after the AA pass above so AA's total+active pair wins. Params
        # are not scores, so the URL backfill leaves them alone.
        if not args.fill_source_urls:
            hf_params_filled, hf_params_changes = fill_missing_params_from_huggingface(doc)
            changes.extend(hf_params_changes)

    toolathlon_by_slug: dict[str, dict[str, Any]] = {}
    toolathlon_matched = 0
    toolathlon_updated = 0
    if not args.skip_toolathlon:
        toolathlon_by_slug = fetch_toolathlon_data(toolathlon_path, toolathlon_mapping_path)
        toolathlon_matched, toolathlon_updated, toolathlon_changes = update_toolathlon_scores(
            doc, toolathlon_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(toolathlon_changes)

    deepswe_by_slug: dict[str, dict[str, Any]] = {}
    deepswe_matched = 0
    deepswe_updated = 0
    if not args.skip_deepswe:
        deepswe_by_slug = fetch_deepswe_data(deepswe_path, deepswe_mapping_path)
        deepswe_matched, deepswe_updated, deepswe_changes = update_deepswe_scores(
            doc, deepswe_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(deepswe_changes)

    # Runs after benchlm.ai so DeepSWE's own leaderboard wins on disagreement.
    datacurve_by_slug: dict[str, dict[str, Any]] = {}
    datacurve_matched = 0
    datacurve_updated = 0
    if not args.skip_datacurve:
        datacurve_by_slug = fetch_datacurve_data(datacurve_path, deepswe_mapping_path)
        datacurve_matched, datacurve_updated, datacurve_changes = update_datacurve_scores(
            doc, datacurve_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(datacurve_changes)

    frontierswe_by_slug: dict[str, dict[str, Any]] = {}
    frontierswe_matched = 0
    frontierswe_updated = 0
    if not args.skip_frontierswe:
        frontierswe_by_slug = fetch_frontierswe_data(frontierswe_path, frontierswe_mapping_path)
        frontierswe_matched, frontierswe_updated, frontierswe_changes = update_frontierswe_scores(
            doc, frontierswe_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(frontierswe_changes)

    swe_atlas_by_slug: dict[str, dict[str, Any]] = {}
    swe_atlas_matched = 0
    swe_atlas_updated = 0
    if not args.skip_swe_atlas:
        swe_atlas_by_slug = fetch_swe_atlas_data(swe_atlas_path, swe_atlas_mapping_path)
        swe_atlas_matched, swe_atlas_updated, swe_atlas_changes = update_swe_atlas_scores(
            doc, swe_atlas_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(swe_atlas_changes)

    evals_report_by_slug: dict[str, dict[str, Any]] = {}
    evals_report_matched = 0
    evals_report_updated = 0
    if not args.skip_evals_report:
        evals_report_by_slug = fetch_evals_report_data(evals_report_path, evals_report_mapping_path)
        evals_report_matched, evals_report_updated, evals_report_changes = update_evals_report_scores(
            doc, evals_report_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(evals_report_changes)

    # Runs after evals.report so the benchmark's own site wins on disagreement.
    frontiercode_by_slug: dict[str, dict[str, Any]] = {}
    frontiercode_matched = 0
    frontiercode_updated = 0
    if not args.skip_frontiercode:
        frontiercode_by_slug = fetch_frontiercode_data(frontiercode_path, frontiercode_mapping_path)
        frontiercode_matched, frontiercode_updated, frontiercode_changes = update_frontiercode_scores(
            doc, frontiercode_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(frontiercode_changes)

    # Same reason as frontiercode above.
    swe_marathon_by_slug: dict[str, Any] = {}
    swe_marathon_matched = 0
    swe_marathon_updated = 0
    if not args.skip_swe_marathon:
        swe_marathon_by_slug = fetch_swe_marathon_data(swe_marathon_path, swe_marathon_mapping_path)
        swe_marathon_matched, swe_marathon_updated, swe_marathon_changes = update_swe_marathon_scores(
            doc, swe_marathon_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(swe_marathon_changes)

    spheron_by_slug: dict[str, dict[str, Any]] = {}
    spheron_matched = 0
    spheron_updated = 0
    if not args.skip_spheron:
        spheron_by_slug = fetch_spheron_data(spheron_path, spheron_mapping_path)
        spheron_matched, spheron_updated, spheron_changes = update_spheron_vram(doc, spheron_by_slug)
        changes.extend(spheron_changes)

    missing = [slug for slug in slugs if slug not in aa_slug_by_model] if not args.skip_aa else []
    if args.write:
        # The derived Coding index is a function of the scores just fetched, so
        # it goes stale the moment any of them moves. Refreshed in memory here
        # so a direct `update.py -w` leaves llm.json consistent on its own,
        # without depending on update-all's later derive step. The URL backfill
        # moves no score, and skipping the refresh keeps its diff pure.
        if not args.fill_source_urls:
            derive_coding_index.refresh_and_report(doc)
        llm_path.write_text(json.dumps(doc, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8")

    print(f"models in {llm_path}: {len(slugs)}")
    if not args.skip_aa:
        print(f"models available on artificialanalysis.py: {len(existing_slugs)}")
        print(f"models returned by artificialanalysis.py: {len(by_slug)}")
    if not args.skip_swe_rebench:
        print(f"models returned by swe_rebench: {len(swe_rebench_by_slug)}")
    if not args.skip_osworld:
        print(f"models returned by osworld: {len(osworld_by_slug)}")
    if not args.skip_llmstats:
        print(f"models returned by llmstats: {len(llmstats_by_slug)}")
    if not args.skip_huggingface:
        print(f"models returned by huggingface: {len(huggingface_by_slug)}")
    if not args.skip_toolathlon:
        print(f"models returned by toolathlon: {len(toolathlon_by_slug)}")
    if not args.skip_deepswe:
        print(f"models returned by deepswe: {len(deepswe_by_slug)}")
    if not args.skip_datacurve:
        print(f"models returned by datacurve: {len(datacurve_by_slug)}")
    if not args.skip_frontierswe:
        print(f"models returned by frontierswe: {len(frontierswe_by_slug)}")
    if not args.skip_frontiercode:
        print(f"models returned by frontiercode: {len(frontiercode_by_slug)}")
    if not args.skip_swe_atlas:
        print(f"models returned by swe_atlas: {len(swe_atlas_by_slug)}")
    if not args.skip_evals_report:
        print(f"models returned by evals_report: {len(evals_report_by_slug)}")
    if not args.skip_swe_marathon:
        print(f"models returned by swe_marathon: {len(swe_marathon_by_slug)}")
    if not args.skip_spheron:
        print(f"models returned by spheron: {len(spheron_by_slug)}")
    if missing:
        print("missing models:")
        for slug in missing:
            print(f"  - {slug}")
    if not args.skip_aa:
        mapped_aa_keys = {aa_key for aa_keys, _transform in SCORE_MAPPINGS.values() for aa_key in aa_keys}
        ignored_aa_keys = sorted(seen_eval_keys - mapped_aa_keys)
        print("ignored keys:")
        if ignored_aa_keys:
            for key in ignored_aa_keys:
                print(f"  - {key}")
        else:
            print("  - (none)")
    print()
    if not args.skip_aa:
        print(f"models matched on artificialanalysis.py: {matched}")
    if not args.skip_swe_rebench:
        print(f"models matched on swe_rebench: {swe_matched}")
    if not args.skip_osworld:
        print(f"models matched on osworld: {osworld_matched}")
    if not args.skip_llmstats:
        print(f"models matched on llmstats: {llmstats_matched}")
    if not args.skip_huggingface:
        print(f"models matched on huggingface: {hf_matched}")
    if not args.skip_toolathlon:
        print(f"models matched on toolathlon: {toolathlon_matched}")
    if not args.skip_deepswe:
        print(f"models matched on deepswe: {deepswe_matched}")
    if not args.skip_datacurve:
        print(f"models matched on datacurve: {datacurve_matched}")
    if not args.skip_frontierswe:
        print(f"models matched on frontierswe: {frontierswe_matched}")
    if not args.skip_frontiercode:
        print(f"models matched on frontiercode: {frontiercode_matched}")
    if not args.skip_swe_atlas:
        print(f"models matched on swe_atlas: {swe_atlas_matched}")
    if not args.skip_evals_report:
        print(f"models matched on evals_report: {evals_report_matched}")
    if not args.skip_swe_marathon:
        print(f"models matched on swe_marathon: {swe_marathon_matched}")
    if not args.skip_spheron:
        print(f"models matched on spheron: {spheron_matched}")
    action = "source URLs filled" if args.fill_source_urls else "score values updated"
    if not args.skip_aa:
        print(f"{action} from artificialanalysis.py: {aa_updated}")
    if not args.skip_swe_rebench:
        print(f"{action} from swe_rebench: {swe_updated}")
    if not args.skip_osworld:
        print(f"{action} from osworld: {osworld_updated}")
    if not args.skip_llmstats:
        print(f"{action} from llmstats: {llmstats_updated}")
    if not args.skip_huggingface:
        print(f"{action} from huggingface: {hf_updated}")
        if not args.fill_source_urls:
            print(f"params values filled from huggingface: {hf_params_filled}")
    if not args.skip_toolathlon:
        print(f"{action} from toolathlon: {toolathlon_updated}")
    if not args.skip_deepswe:
        print(f"{action} from deepswe: {deepswe_updated}")
    if not args.skip_datacurve:
        print(f"{action} from datacurve: {datacurve_updated}")
    if not args.skip_frontierswe:
        print(f"{action} from frontierswe: {frontierswe_updated}")
    if not args.skip_frontiercode:
        print(f"{action} from frontiercode: {frontiercode_updated}")
    if not args.skip_swe_atlas:
        print(f"{action} from swe_atlas: {swe_atlas_updated}")
    if not args.skip_evals_report:
        print(f"{action} from evals_report: {evals_report_updated}")
    if not args.skip_swe_marathon:
        print(f"{action} from swe_marathon: {swe_marathon_updated}")
    if not args.skip_spheron:
        print(f"vram values updated from spheron: {spheron_updated}")
    print()
    print_changes_table(changes)
    print()
    if not args.write:
        print("dry-run only, pass --write to persist changes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
