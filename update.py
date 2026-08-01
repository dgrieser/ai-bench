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

from _scores import stamp_score_updated

from _swe_rebench_mapping import load_rebench_to_slug_mapping
from _osworld_mapping import load_osworld_to_slug_mapping
from _huggingface_mapping import load_hf_to_key_mapping
from _deepswe_mapping import load_deepswe_to_slug_mapping
from _toolathlon_mapping import load_toolathlon_to_slug_mapping
from _frontierswe_mapping import load_frontierswe_to_slug_mapping
from _swe_atlas_mapping import load_swe_atlas_to_slug_mapping
from _evals_report_mapping import load_evals_report_to_slug_mapping
from _swe_marathon_mapping import load_swe_marathon_to_slug_mapping
from _spheron_mapping import load_spheron_to_slug_mapping
from _llmstats_mapping import (
    load_llmstats_to_slug_mapping,
    load_llmstats_benchmark_to_key_mapping,
)
from _artificialanalysis_mapping import load_llm_to_aa_mapping

AA_SCRIPT = Path(__file__).resolve().with_name("artificialanalysis.py")
SWE_REBENCH_SCRIPT = Path(__file__).resolve().with_name("fetch_swe_rebench.py")
OSWORLD_SCRIPT = Path(__file__).resolve().with_name("fetch_osworld.py")
HF_SCRIPT = Path(__file__).resolve().with_name("fetch_huggingface.py")
DEEPSWE_SCRIPT = Path(__file__).resolve().with_name("fetch_deepswe.py")
TOOLATHLON_SCRIPT = Path(__file__).resolve().with_name("fetch_toolathlon.py")
FRONTIERSWE_SCRIPT = Path(__file__).resolve().with_name("fetch_frontierswe.py")
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


def to_percent(value: Any) -> int | float | None:
    if value is None:
        return None
    pct = float(value) * 100.0
    rounded = round(pct, 1)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def to_index(value: Any) -> int | float | None:
    """Pass an Artificial Analysis index/Elo through unscaled, rounded to 0.1.

    AA-Omniscience (-100..100) and GDPval-AA Elo (human baseline = 1000) are not
    fractions, so to_percent() would multiply them by 100.
    """
    if value is None:
        return None
    rounded = round(float(value), 1)
    return int(rounded) if rounded.is_integer() else rounded


def fmt_change_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_aa_value(value: Any) -> Any:
    # Treat zero values from Artificial Analysis as unset/null.
    if value == 0:
        return None
    return value


def _format_context_tokens(tokens: int) -> str:
    if tokens % 1_000_000_000 == 0:
        return f"{tokens // 1_000_000_000}b"
    if tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}m"
    if tokens % 1_000 == 0:
        return f"{tokens // 1_000}k"
    return str(tokens)


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
    should_snap = unit in {"", "k"}
    if should_snap:
        canonical = [
            1_000,
            2_000,
            4_000,
            8_000,
            16_000,
            32_000,
            64_000,
            128_000,
            256_000,
            512_000,
            1_024_000,
            2_048_000,
        ]
        for target in canonical:
            if abs(tokens - target) / target <= 0.03:
                tokens = target
                break

    return _format_context_tokens(tokens)


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
        help="Skip fetching scores from huggingface.",
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
        "--skip-frontierswe",
        action="store_true",
        help="Skip fetching scores from frontierswe.",
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
) -> dict[str, str]:
    llm_to_aa = load_llm_to_aa_mapping(mapping_path)
    resolved: dict[str, str] = {}
    for slug in slugs:
        if slug in available_slugs:
            resolved[slug] = slug
            continue
        mapped_slug = llm_to_aa.get(slug)
        if mapped_slug in available_slugs:
            resolved[slug] = mapped_slug
    return resolved


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
        by_slug.setdefault(slug, row)
    return by_slug


def update_scores(
    doc: dict[str, Any], by_slug: dict[str, dict[str, Any]]
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

        old_context = model.get("context")
        new_context = normalize_context(aa_model.get("context"))
        if not (old_context is not None and new_context is None) and old_context != new_context:
            model["context"] = new_context
            updated += 1
            changes.append((slug, "context", old_context, new_context))

        scores = model.setdefault("scores", {})
        if not isinstance(scores, dict):
            continue

        for llm_key, (aa_keys, transform) in SCORE_MAPPINGS.items():
            aa_value = None
            for aa_key in aa_keys:
                if aa_key in evaluations and evaluations.get(aa_key) is not None:
                    aa_value = evaluations.get(aa_key)
                    break
            new_value = transform(normalize_aa_value(aa_value))
            old_value = scores.get(llm_key)

            # Never overwrite an existing non-null value with null.
            if old_value is not None and new_value is None:
                continue

            if old_value != new_value:
                scores[llm_key] = new_value
                stamp_score_updated(model, llm_key)
                updated += 1
                changes.append((slug, llm_key, old_value, new_value))

    return matched, updated, seen_eval_keys, changes


def update_swe_rebench_scores(
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
        rebench_model = by_slug.get(slug)
        if rebench_model is None:
            continue

        matched += 1
        scores = model.setdefault("scores", {})
        if not isinstance(scores, dict):
            continue

        old_value = scores.get("swe_rebench")
        new_value = rebench_model.get("resolved_rate")

        # Never overwrite an existing non-null value with null.
        if old_value is not None and new_value is None:
            continue

        if old_value != new_value:
            scores["swe_rebench"] = new_value
            stamp_score_updated(model, "swe_rebench")
            updated += 1
            changes.append((slug, "swe_rebench", old_value, new_value))

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
        by_slug.setdefault(slug, row)
    return by_slug


def update_osworld_scores(
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
        osworld_model = by_slug.get(slug)
        if osworld_model is None:
            continue

        matched += 1
        scores = model.setdefault("scores", {})
        if not isinstance(scores, dict):
            continue

        old_value = scores.get("osworld_verified")
        new_value = osworld_model.get("success_rate")

        # Never overwrite an existing non-null value with null.
        if old_value is not None and new_value is None:
            continue

        if old_value != new_value:
            scores["osworld_verified"] = new_value
            stamp_score_updated(model, "osworld_verified")
            updated += 1
            changes.append((slug, "osworld_verified", old_value, new_value))

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
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        slug = row.get("model")
        scores = row.get("scores")
        if not isinstance(slug, str) or not slug or not isinstance(scores, dict):
            continue
        mapped: dict[str, Any] = {}
        for label, value in scores.items():
            key = hf_to_key.get(label)
            if not key or value is None:
                continue
            mapped.setdefault(key, value)
        if mapped:
            by_slug[slug] = mapped
    return by_slug


def update_huggingface_scores(
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
        hf_scores = by_slug.get(slug)
        if not hf_scores:
            continue

        matched += 1
        scores = model.setdefault("scores", {})
        if not isinstance(scores, dict):
            continue

        for benchmark_key, new_value in hf_scores.items():
            if new_value is None:
                continue
            # HF self-reports are lowest-trust: only fill nulls, never overwrite.
            old_value = scores.get(benchmark_key)
            if old_value is not None:
                continue
            scores[benchmark_key] = new_value
            stamp_score_updated(model, benchmark_key)
            updated += 1
            changes.append((slug, benchmark_key, old_value, new_value))

    return matched, updated, changes


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
        by_slug.setdefault(slug, row)
    return by_slug


def update_toolathlon_scores(
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
        toolathlon_model = by_slug.get(slug)
        if toolathlon_model is None:
            continue

        matched += 1
        scores = model.setdefault("scores", {})
        if not isinstance(scores, dict):
            continue

        old_value = scores.get("toolathlon")
        new_value = toolathlon_model.get("score")

        # Never overwrite an existing non-null value with null.
        if old_value is not None and new_value is None:
            continue

        if old_value != new_value:
            scores["toolathlon"] = new_value
            stamp_score_updated(model, "toolathlon")
            updated += 1
            changes.append((slug, "toolathlon", old_value, new_value))

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
        by_slug.setdefault(slug, row)
    return by_slug


def update_deepswe_scores(
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
        deepswe_model = by_slug.get(slug)
        if deepswe_model is None:
            continue

        matched += 1
        scores = model.setdefault("scores", {})
        if not isinstance(scores, dict):
            continue

        old_value = scores.get("deepswe")
        new_value = deepswe_model.get("score")

        # Never overwrite an existing non-null value with null.
        if old_value is not None and new_value is None:
            continue

        if old_value != new_value:
            scores["deepswe"] = new_value
            stamp_score_updated(model, "deepswe")
            updated += 1
            changes.append((slug, "deepswe", old_value, new_value))

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
        by_slug.setdefault(slug, row)
    return by_slug


def update_frontierswe_scores(
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
        frontierswe_model = by_slug.get(slug)
        if frontierswe_model is None:
            continue

        matched += 1
        scores = model.setdefault("scores", {})
        if not isinstance(scores, dict):
            continue

        old_value = scores.get("frontierswe")
        new_value = frontierswe_model.get("score")

        # Never overwrite an existing non-null value with null.
        if old_value is not None and new_value is None:
            continue

        if old_value != new_value:
            scores["frontierswe"] = new_value
            stamp_score_updated(model, "frontierswe")
            updated += 1
            changes.append((slug, "frontierswe", old_value, new_value))

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
        swe_atlas_scores = by_slug.get(slug)
        if not swe_atlas_scores:
            continue

        matched += 1
        scores = model.setdefault("scores", {})
        if not isinstance(scores, dict):
            continue

        for benchmark_key, new_value in swe_atlas_scores.items():
            old_value = scores.get(benchmark_key)
            # Never overwrite an existing non-null value with null.
            if old_value is not None and new_value is None:
                continue
            if old_value != new_value:
                scores[benchmark_key] = new_value
                stamp_score_updated(model, benchmark_key)
                updated += 1
                changes.append((slug, benchmark_key, old_value, new_value))

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
        evals_report_scores = by_slug.get(slug)
        if not evals_report_scores:
            continue

        matched += 1
        scores = model.setdefault("scores", {})
        if not isinstance(scores, dict):
            continue

        for benchmark_key, new_value in evals_report_scores.items():
            old_value = scores.get(benchmark_key)
            # Never overwrite an existing non-null value with null.
            if old_value is not None and new_value is None:
                continue
            if old_value != new_value:
                scores[benchmark_key] = new_value
                stamp_score_updated(model, benchmark_key)
                updated += 1
                changes.append((slug, benchmark_key, old_value, new_value))

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
    doc: dict[str, Any], by_slug: dict[str, Any]
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
        scores = model.setdefault("scores", {})
        if not isinstance(scores, dict):
            continue

        new_value = by_slug[slug]
        old_value = scores.get("swe_marathon")
        if old_value == new_value:
            continue
        scores["swe_marathon"] = new_value
        stamp_score_updated(model, "swe_marathon")
        updated += 1
        changes.append((slug, "swe_marathon", old_value, new_value))

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

    # slug -> {quant -> vram GB}
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
        by_slug[slug] = {
            "fp16": row.get("vram_fp16"),
            "int8": row.get("vram_int8"),
            "int4": row.get("vram_int4"),
        }
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
            mapped.setdefault(key, to_percent(value))
        if mapped:
            by_slug.setdefault(slug, {}).update(mapped)
    return by_slug


def update_llmstats_scores(
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
        llmstats_scores = by_slug.get(slug)
        if not llmstats_scores:
            continue

        matched += 1
        scores = model.setdefault("scores", {})
        if not isinstance(scores, dict):
            continue

        for benchmark_key, new_value in llmstats_scores.items():
            if new_value is None:
                continue
            # General aggregator: keep aa (and every named source) leading.
            # Only fill nulls, never overwrite an existing value.
            old_value = scores.get(benchmark_key)
            if old_value is not None:
                continue
            scores[benchmark_key] = new_value
            stamp_score_updated(model, benchmark_key)
            updated += 1
            changes.append((slug, benchmark_key, old_value, new_value))

    return matched, updated, changes


def main() -> int:
    args = parse_args()
    llm_path = Path(args.json_file)
    aa_path = AA_SCRIPT
    swe_rebench_path = SWE_REBENCH_SCRIPT
    osworld_path = OSWORLD_SCRIPT
    huggingface_path = HF_SCRIPT
    deepswe_path = DEEPSWE_SCRIPT
    toolathlon_path = TOOLATHLON_SCRIPT
    frontierswe_path = FRONTIERSWE_SCRIPT
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
    if not args.skip_frontierswe:
        print(f"  - {shlex.join(build_fetch_frontierswe_cmd(frontierswe_path))}")
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
    aa_slug_by_model: dict[str, str] = {}
    by_slug: dict[str, dict[str, Any]] = {}
    matched = 0
    aa_updated = 0
    seen_eval_keys: set[str] = set()

    if not args.skip_aa:
        available_slugs = fetch_available_slugs(aa_path)
        aa_slug_by_model = resolve_aa_slugs(slugs, available_slugs, aa_model_mapping_path)
        existing_slugs = list(dict.fromkeys(aa_slug_by_model.values()))
        print(f"  - {shlex.join(build_fetch_data_cmd(aa_path, existing_slugs))}")
    print()

    if not args.skip_aa:
        by_aa_slug = fetch_aa_data(aa_path, existing_slugs)
        by_slug = {
            slug: by_aa_slug[aa_slug]
            for slug, aa_slug in aa_slug_by_model.items()
            if aa_slug in by_aa_slug
        }
        matched, aa_updated, seen_eval_keys, aa_changes = update_scores(doc, by_slug)
        changes.extend(aa_changes)

    swe_rebench_by_slug: dict[str, dict[str, Any]] = {}
    swe_matched = 0
    swe_updated = 0
    if not args.skip_swe_rebench:
        swe_rebench_by_slug = fetch_swe_rebench_data(swe_rebench_path, swe_rebench_mapping_path)
        swe_matched, swe_updated, swe_changes = update_swe_rebench_scores(doc, swe_rebench_by_slug)
        changes.extend(swe_changes)

    osworld_by_slug: dict[str, dict[str, Any]] = {}
    osworld_matched = 0
    osworld_updated = 0
    if not args.skip_osworld:
        osworld_by_slug = fetch_osworld_data(osworld_path, osworld_mapping_path)
        osworld_matched, osworld_updated, osworld_changes = update_osworld_scores(doc, osworld_by_slug)
        changes.extend(osworld_changes)

    llmstats_by_slug: dict[str, dict[str, Any]] = {}
    llmstats_matched = 0
    llmstats_updated = 0
    if not args.skip_llmstats:
        llmstats_by_slug = fetch_llmstats_data(
            llmstats_path, llmstats_model_mapping_path, llmstats_benchmark_mapping_path
        )
        llmstats_matched, llmstats_updated, llmstats_changes = update_llmstats_scores(doc, llmstats_by_slug)
        changes.extend(llmstats_changes)

    huggingface_by_slug: dict[str, dict[str, Any]] = {}
    hf_matched = 0
    hf_updated = 0
    if not args.skip_huggingface:
        huggingface_by_slug = fetch_huggingface_data(huggingface_path, huggingface_mapping_path)
        hf_matched, hf_updated, hf_changes = update_huggingface_scores(doc, huggingface_by_slug)
        changes.extend(hf_changes)

    toolathlon_by_slug: dict[str, dict[str, Any]] = {}
    toolathlon_matched = 0
    toolathlon_updated = 0
    if not args.skip_toolathlon:
        toolathlon_by_slug = fetch_toolathlon_data(toolathlon_path, toolathlon_mapping_path)
        toolathlon_matched, toolathlon_updated, toolathlon_changes = update_toolathlon_scores(
            doc, toolathlon_by_slug
        )
        changes.extend(toolathlon_changes)

    deepswe_by_slug: dict[str, dict[str, Any]] = {}
    deepswe_matched = 0
    deepswe_updated = 0
    if not args.skip_deepswe:
        deepswe_by_slug = fetch_deepswe_data(deepswe_path, deepswe_mapping_path)
        deepswe_matched, deepswe_updated, deepswe_changes = update_deepswe_scores(doc, deepswe_by_slug)
        changes.extend(deepswe_changes)

    frontierswe_by_slug: dict[str, dict[str, Any]] = {}
    frontierswe_matched = 0
    frontierswe_updated = 0
    if not args.skip_frontierswe:
        frontierswe_by_slug = fetch_frontierswe_data(frontierswe_path, frontierswe_mapping_path)
        frontierswe_matched, frontierswe_updated, frontierswe_changes = update_frontierswe_scores(doc, frontierswe_by_slug)
        changes.extend(frontierswe_changes)

    swe_atlas_by_slug: dict[str, dict[str, Any]] = {}
    swe_atlas_matched = 0
    swe_atlas_updated = 0
    if not args.skip_swe_atlas:
        swe_atlas_by_slug = fetch_swe_atlas_data(swe_atlas_path, swe_atlas_mapping_path)
        swe_atlas_matched, swe_atlas_updated, swe_atlas_changes = update_swe_atlas_scores(doc, swe_atlas_by_slug)
        changes.extend(swe_atlas_changes)

    evals_report_by_slug: dict[str, dict[str, Any]] = {}
    evals_report_matched = 0
    evals_report_updated = 0
    if not args.skip_evals_report:
        evals_report_by_slug = fetch_evals_report_data(evals_report_path, evals_report_mapping_path)
        evals_report_matched, evals_report_updated, evals_report_changes = update_evals_report_scores(doc, evals_report_by_slug)
        changes.extend(evals_report_changes)

    # Runs after evals.report so the benchmark's own site wins on disagreement.
    swe_marathon_by_slug: dict[str, Any] = {}
    swe_marathon_matched = 0
    swe_marathon_updated = 0
    if not args.skip_swe_marathon:
        swe_marathon_by_slug = fetch_swe_marathon_data(swe_marathon_path, swe_marathon_mapping_path)
        swe_marathon_matched, swe_marathon_updated, swe_marathon_changes = update_swe_marathon_scores(doc, swe_marathon_by_slug)
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
    if not args.skip_frontierswe:
        print(f"models returned by frontierswe: {len(frontierswe_by_slug)}")
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
    if not args.skip_frontierswe:
        print(f"models matched on frontierswe: {frontierswe_matched}")
    if not args.skip_swe_atlas:
        print(f"models matched on swe_atlas: {swe_atlas_matched}")
    if not args.skip_evals_report:
        print(f"models matched on evals_report: {evals_report_matched}")
    if not args.skip_swe_marathon:
        print(f"models matched on swe_marathon: {swe_marathon_matched}")
    if not args.skip_spheron:
        print(f"models matched on spheron: {spheron_matched}")
    if not args.skip_aa:
        print(f"score values updated from artificialanalysis.py: {aa_updated}")
    if not args.skip_swe_rebench:
        print(f"score values updated from swe_rebench: {swe_updated}")
    if not args.skip_osworld:
        print(f"score values updated from osworld: {osworld_updated}")
    if not args.skip_llmstats:
        print(f"score values updated from llmstats: {llmstats_updated}")
    if not args.skip_huggingface:
        print(f"score values updated from huggingface: {hf_updated}")
    if not args.skip_toolathlon:
        print(f"score values updated from toolathlon: {toolathlon_updated}")
    if not args.skip_deepswe:
        print(f"score values updated from deepswe: {deepswe_updated}")
    if not args.skip_frontierswe:
        print(f"score values updated from frontierswe: {frontierswe_updated}")
    if not args.skip_swe_atlas:
        print(f"score values updated from swe_atlas: {swe_atlas_updated}")
    if not args.skip_evals_report:
        print(f"score values updated from evals_report: {evals_report_updated}")
    if not args.skip_swe_marathon:
        print(f"score values updated from swe_marathon: {swe_marathon_updated}")
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
