#!/usr/bin/env python3
"""Update benchmark JSON scores using artificialanalysis.py and the benchmark leaderboard scrapers."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import _fetch_warnings
import _history
import artificialanalysis
import derive_indexes
import fetch_aa_coding_agents
import fetch_agents_last_exam
import fetch_bfcl
import fetch_datacurve
import fetch_deepswe
import fetch_evals_report
import fetch_frontierswe
import fetch_huggingface
import fetch_llmstats
import fetch_mcp_atlas
import fetch_openrouter
import fetch_osworld
import fetch_programbench
import fetch_real_swe
import fetch_swe_atlas
import fetch_swe_marathon
import fetch_tbench
import fetch_toolathlon
import fetch_vals
import fetch_zerobench
from _context import format_context_tokens, snap_context_tokens
from _params import fetch_hf_params, normalize_params
from _precedence import (
    AA_CODING_AGENTS_SOURCE_URL,
    AGENTS_LAST_EXAM_SOURCE_URL,
    BFCL_SOURCE_URL,
    DATACURVE_SOURCE_URL,
    DEEPSWE_SOURCE_URL,
    EVALS_REPORT_KEY_URLS,
    FRONTIERCODE_SOURCE_URL,
    FRONTIERSWE_KEY_URLS,
    FRONTIERSWE_SOURCE_URL,
    LLMSTATS_SOURCE_URL,
    MCP_ATLAS_SOURCE_URL,
    OSWORLD_KEY_URLS,
    OSWORLD_SOURCE_URL,
    PROGRAMBENCH_SOURCE_URL,
    REAL_SWE_SOURCE_URL,
    RANK_AA,
    SWE_ATLAS_KEY_URLS,
    SWE_MARATHON_SOURCE_URL,
    TBENCH_KEY_URLS,
    TOOLATHLON_SOURCE_URL,
    VALS_KEY_URLS,
    ZEROBENCH_SOURCE_URL,
    is_fetcher_source,
    may_overwrite,
    source_rank,
    yields_to_fill_only,
)
from _reference import apply_reference_flags, missing_reference_models
from _scores import (
    check_score_range, round_score, score_source, stamp_score_source, stamp_score_updated,
)
# run_fetch below accounts for the fetcher subprocesses, timed() for everything
# else this script does, so the two together add up to the wall time update-all
# attributes to update.py. They did not use to: the fetchers came to about 109
# of a 150-second run, and the missing 41 seconds -- the in-memory index
# refresh, mostly -- were invisible, which is the reverse of what a reader
# needs, since a slow fetcher is someone else's outage and a slow phase here is
# ours.
from _timing import timed
from fill_source_urls import canonical

from _aa_coding_agents_mapping import load_aa_coding_agents_to_slug_mapping
from _bfcl_mapping import load_bfcl_to_slug_mapping
from _mcp_atlas_mapping import load_mcp_atlas_to_slug_mapping
from _zerobench_mapping import load_zerobench_to_slug_mapping
from _openrouter_mapping import load_openrouter_to_slug_mapping
from _osworld_mapping import load_osworld_to_slug_mapping
from _huggingface_mapping import load_hf_to_key_mapping
from _deepswe_mapping import load_deepswe_to_slug_mapping
from _toolathlon_mapping import load_toolathlon_to_slug_mapping
from _programbench_mapping import load_programbench_to_slug_mapping
from _real_swe_mapping import load_real_swe_to_slug_mapping
from _frontierswe_mapping import load_frontierswe_to_slug_mapping
from _frontiercode_mapping import load_frontiercode_to_slug_mapping
from _swe_atlas_mapping import load_swe_atlas_to_slug_mapping
from _evals_report_mapping import load_evals_report_to_slug_mapping
from _vals_mapping import load_vals_to_slug_mapping
from _revisions import known_revision_key, subset_base
from _swe_marathon_mapping import load_swe_marathon_to_slug_mapping
from _tbench_mapping import load_tbench_to_slug_mapping
from _agents_last_exam_mapping import load_agents_last_exam_to_slug_mapping
from _spheron_mapping import load_spheron_to_slug_mapping
from _llmstats_mapping import (
    load_llmstats_to_slug_mapping,
    load_llmstats_benchmark_to_key_mapping,
)
from _artificialanalysis_mapping import load_llm_to_aa_slugs

AA_SCRIPT = Path(__file__).resolve().with_name("artificialanalysis.py")
AA_CODING_AGENTS_SCRIPT = Path(__file__).resolve().with_name("fetch_aa_coding_agents.py")
OSWORLD_SCRIPT = Path(__file__).resolve().with_name("fetch_osworld.py")
HF_SCRIPT = Path(__file__).resolve().with_name("fetch_huggingface.py")
DEEPSWE_SCRIPT = Path(__file__).resolve().with_name("fetch_deepswe.py")
DATACURVE_SCRIPT = Path(__file__).resolve().with_name("fetch_datacurve.py")
TOOLATHLON_SCRIPT = Path(__file__).resolve().with_name("fetch_toolathlon.py")
MCP_ATLAS_SCRIPT = Path(__file__).resolve().with_name("fetch_mcp_atlas.py")
ZEROBENCH_SCRIPT = Path(__file__).resolve().with_name("fetch_zerobench.py")
PROGRAMBENCH_SCRIPT = Path(__file__).resolve().with_name("fetch_programbench.py")
REAL_SWE_SCRIPT = Path(__file__).resolve().with_name("fetch_real_swe.py")
BFCL_SCRIPT = Path(__file__).resolve().with_name("fetch_bfcl.py")
FRONTIERSWE_SCRIPT = Path(__file__).resolve().with_name("fetch_frontierswe.py")
TBENCH_SCRIPT = Path(__file__).resolve().with_name("fetch_tbench.py")
AGENTS_LAST_EXAM_SCRIPT = Path(__file__).resolve().with_name("fetch_agents_last_exam.py")
FRONTIERCODE_SCRIPT = Path(__file__).resolve().with_name("fetch_frontiercode.py")
SWE_ATLAS_SCRIPT = Path(__file__).resolve().with_name("fetch_swe_atlas.py")
EVALS_REPORT_SCRIPT = Path(__file__).resolve().with_name("fetch_evals_report.py")
VALS_SCRIPT = Path(__file__).resolve().with_name("fetch_vals.py")
SWE_MARATHON_SCRIPT = Path(__file__).resolve().with_name("fetch_swe_marathon.py")
SPHERON_SCRIPT = Path(__file__).resolve().with_name("fetch_spheron.py")
LLMSTATS_SCRIPT = Path(__file__).resolve().with_name("fetch_llmstats.py")
OPENROUTER_SCRIPT = Path(__file__).resolve().with_name("fetch_openrouter.py")
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

    AA-Omniscience (-100..100) and the GDPval-AA and AA-Briefcase Elos are not
    fractions, so to_percent() would multiply them by 100.
    """
    if value is None:
        return None
    return float(value)


def fmt_change_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


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
    # AA runs both live Terminal-Bench revisions itself, under its own harness,
    # so each lands in the column its own board also publishes -- the 4.0 one
    # beside fetch_tbench.py's, which AA outranks (_precedence.RANK_AA), and the
    # two disagree by a few points because they are different harnesses over the
    # same task set. The second spelling is the one the v2 API would use if it
    # ever carries the field; the pages spell it the first way.
    "terminal_bench_4_0": (("terminalbench_4_0", "terminal_bench_4_0"), to_percent),
    "terminal_bench_2_1": (("terminalbench_v2_1",), to_percent),
    # AA reports the τ³ Banking domain under the version-less key "tau_banking".
    "tau3_bench_banking": (("tau_banking",), to_percent),
    "tau2_bench_telecom": (("tau2",), to_percent),
    # AA's independent ITBench implementation; the page reports the SRE track.
    "itbench_aa": (("it_bench_sre",), to_percent),
    "gdpval_aa": (("gdpval",), to_index),
    # AA-Briefcase reports a composite Elo (GPT-5.5 (medium) = 1000), not a fraction.
    "aa_briefcase": (("briefcase",), to_index),
    "aa_omniscience": (("omniscience",), to_index),
    "aa_omniscience_hallucination": (("omniscience_hallucination_rate",), to_percent),
    # The other half of the same run: correct share before the Omniscience
    # Index nets confident errors off against it. Both halves feed the Trust
    # index, where the accuracy is what stops a model that abstains on
    # everything from topping the hallucination rate.
    "aa_omniscience_accuracy": (("omniscience_accuracy",), to_percent),
    "aa_lcr": (("lcr",), to_percent),
    # Ai2's IFBench, run independently by AA; the model pages carry it too.
    "ifbench": (("ifbench",), to_percent),
    "critpt": (("critpt",), to_percent),
    "aime_2025": (("aime_25",), to_percent),
    # AA runs MMLU-Pro itself; the field is API-only, and the HF cards and
    # evals.report carry the same benchmark for models AA has not run.
    "mmlu_pro": (("mmlu_pro",), to_percent),
    "mmmu_pro": (("mmmu_pro",), to_percent),
    "gpqa_diamond": (("gpqa",), to_percent),
    "livecodebench": (("livecodebench",), to_percent),
    "scicode": (("scicode",), to_percent),
    "hle": (("hle",), to_percent),
    "aa_intelligence_index": (("artificial_analysis_intelligence_index",), lambda v: v),
}

# Per-score source pages live in _precedence.py, next to the rank each source
# carries: the page a score is attributed to and the authority that attribution
# confers are the same fact, and stating them apart invites them to drift. Both
# are read from the fetch_*.py constants (the fill_source_urls.py rule).


def aa_model_page_url(aa_slug: str) -> str:
    return canonical(artificialanalysis.MODEL_PAGE_URL.format(aa_slug))


def parse_args() -> argparse.Namespace:
    parser = HelpOnErrorArgumentParser(
        description="Update benchmark JSON scores by querying artificialanalysis.py and the benchmark leaderboard scrapers."
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
        "--skip-aa-coding-agents",
        action="store_true",
        help="Skip fetching scores from the AA Coding Agent Index.",
    )
    parser.add_argument(
        "--skip-osworld",
        action="store_true",
        help="Skip fetching OSWorld-Verified and OSWorld 2.0 scores.",
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
        "--skip-mcp-atlas",
        action="store_true",
        help="Skip fetching scores from the Scale Labs MCP-Atlas leaderboard.",
    )
    parser.add_argument(
        "--skip-zerobench",
        action="store_true",
        help="Skip fetching scores from the ZeroBench leaderboard.",
    )
    parser.add_argument(
        "--skip-bfcl",
        action="store_true",
        help="Skip fetching scores from the Berkeley Function-Calling Leaderboard.",
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
        "--skip-programbench",
        action="store_true",
        help="Skip fetching scores from the ProgramBench leaderboard.",
    )
    parser.add_argument(
        "--skip-real-swe",
        action="store_true",
        help="Skip fetching scores from the Real-SWE leaderboard.",
    )
    parser.add_argument(
        "--skip-tbench",
        action="store_true",
        help="Skip fetching Terminal-Bench 4.0, 2.1 and 2.0 scores from tbench.ai.",
    )
    parser.add_argument(
        "--skip-agents-last-exam",
        action="store_true",
        help="Skip fetching Agents' Last Exam scores from agents-last-exam.org.",
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
        "--skip-vals",
        action="store_true",
        help="Skip fetching scores from vals.ai.",
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
    parser.add_argument(
        "--skip-openrouter",
        action="store_true",
        help="Skip fetching GPQA Diamond scores from OpenRouter's model pages.",
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


# Fetcher subprocesses started ahead of need, keyed by their exact command.
# The fetchers read unrelated hosts and none depends on another's output, so
# prefetch() starts them all at once and each fetch_*_data() below collects its
# result as it reaches it; run one after another they were most of update.py's
# wall time, spent waiting on the network. Output goes to temporary files
# rather than pipes, so a chatty fetcher never stalls on a full pipe while
# nothing is reading it yet.
_PREFETCHED: dict[tuple[str, ...], tuple[subprocess.Popen, Any, Any, float]] = {}

# Seconds one fetcher may run, from launch, before it is killed and its source
# counted as failed. Without it a fetcher hung on a dead host holds update.py
# until the workflow's own timeout kills the job, and then nothing is written.
FETCH_TIMEOUT_VAR = "AI_BENCH_FETCH_TIMEOUT"
DEFAULT_FETCH_TIMEOUT = 900


def fetch_timeout() -> float:
    try:
        return float(os.getenv(FETCH_TIMEOUT_VAR, str(DEFAULT_FETCH_TIMEOUT)))
    except ValueError:
        return float(DEFAULT_FETCH_TIMEOUT)


def _timed_out(cmd: list[str], limit: float) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        cmd, -9, "", f"TimeoutError: killed after {limit:.0f}s ({FETCH_TIMEOUT_VAR})"
    )


def _kill_prefetched() -> None:
    """Stop fetchers nobody will collect -- a run that died early, or ^C."""
    for proc, out, err, _started in _PREFETCHED.values():
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        out.close()
        err.close()
    _PREFETCHED.clear()


def prefetch(cmds: list[list[str]]) -> None:
    """Start every fetcher command now; run_fetch() collects the results."""
    if not _PREFETCHED and cmds:
        atexit.register(_kill_prefetched)
    for cmd in cmds:
        key = tuple(cmd)
        if key in _PREFETCHED:
            continue
        out = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        err = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, text=True)
        _PREFETCHED[key] = (proc, out, err, time.monotonic())


def _collect(
    entry: tuple[subprocess.Popen, Any, Any, float]
) -> subprocess.CompletedProcess[str]:
    proc, out, err, started = entry
    try:
        limit = fetch_timeout()
        try:
            returncode = proc.wait(timeout=max(0.0, limit - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return _timed_out(list(proc.args), limit)
        out.seek(0)
        err.seek(0)
        return subprocess.CompletedProcess(proc.args, returncode, out.read(), err.read())
    finally:
        out.close()
        err.close()


def run_fetch(cmd: list[str], label: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run one fetcher, reporting on stderr what it cost.

    The commands are all listed up front, before any of them runs, so the
    listing says nothing about where a run's minutes went -- and a reader
    watching a log sees a stall with no fetcher's name against it. Worse in
    CI, where this script's stdout is a pipe and therefore block-buffered:
    the listing arrives in whatever chunk the buffer flushes, so a gap after
    one of those lines belongs to no particular source, though it reads
    exactly like the fetcher named just above it hanging.

    The timings go to stderr, which Python keeps line-buffered whether or not
    it is a terminal, so they show up as each fetcher finishes rather than at
    exit. That also leaves stdout as the report proper, for a run that pipes
    it somewhere.

    `label` is for the two commands that share a script name: the AA slug list
    and the AA score fetch would otherwise both report as artificialanalysis.py.

    A command prefetch() already started is collected rather than run again.
    Its line reports how long this process actually waited for it -- with the
    fetchers running side by side the waits, not the run times, are what add
    up to update.py's wall time.
    """
    name = label or Path(cmd[1]).name
    entry = _PREFETCHED.pop(tuple(cmd), None)
    if entry is not None:
        waited_from = time.monotonic()
        proc = _collect(entry)
        waited = time.monotonic() - waited_from
        # The fetcher may have finished long before it was asked for, so the
        # launch-to-collect span is an upper bound on its run time.
        print(
            f"  {name}: waited {waited:.1f}s (concurrent; collected "
            f"{time.monotonic() - entry[3]:.1f}s after launch)",
            file=sys.stderr,
        )
        return proc
    started = time.monotonic()
    try:
        limit = fetch_timeout()
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=limit)
        except subprocess.TimeoutExpired:
            return _timed_out(cmd, limit)
    finally:
        # In a finally, so an interrupted or unlaunchable fetcher is timed
        # too: how long a dead one took before giving up -- a timeout, a
        # retry loop -- is exactly what a reader is after.
        print(f"  {name}: {time.monotonic() - started:.1f}s", file=sys.stderr)


def fetch_available_slugs(aa_script: Path) -> set[str]:
    cmd = build_list_models_cmd(aa_script)
    proc = run_fetch(cmd, "artificialanalysis.py --list-models")
    if proc.returncode != 0:
        raise RuntimeError(f"artificialanalysis.py --list-models failed ({proc.returncode}): {proc.stderr.strip()}")
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def fetch_aa_data(aa_script: Path, slugs: list[str]) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_data_cmd(aa_script, slugs)
    proc = run_fetch(cmd, "artificialanalysis.py -o json")
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


# Artificial Analysis runs one model several ways -- "GLM-4.6 (Reasoning)" and
# "GLM-4.6 (Non-reasoning)", "GPT-5.6 Sol (max)" down to "(low)" -- and gives
# each run its own slug. Which run the bare slug is was AA's choice and differs
# from model to model (glm-4-6 is the non-reasoning run, gpt-oss-120b the high
# one), so reading the bare slug put max-effort reasoning runs and
# non-reasoning runs side by side in one column. The policy instead: a row reads
# its model's highest-effort reasoning run, the bare slug on a tie.
AA_PUBLISHED_MODELS = Path(__file__).resolve().parent / "_aa" / "models.json"

# Slug suffixes that name a run of the same checkpoint rather than another
# model: "glm-4-6-reasoning" is GLM-4.6 run another way, "deepseek-v4-flash-0420"
# is a different checkpoint and never a candidate.
AA_EFFORT_RANKS = {
    "non-reasoning": 0,
    "minimal": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "reasoning": 4,
    "thinking": 4,
    "xhigh": 5,
    "max": 6,
}
# A run AA names without an effort, "(Reasoning)" or nothing at all, is its
# default reasoning run.
AA_DEFAULT_EFFORT = 4
_EFFORT_WORD = re.compile(r"\b(minimal|low|medium|xhigh|high|max)\b")


def load_aa_model_names(path: Path = AA_PUBLISHED_MODELS) -> dict[str, str]:
    """AA slug -> display name, from the list the refresh publishes to _aa/."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    names: dict[str, str] = {}
    for entry in payload.get("models", []) if isinstance(payload, dict) else []:
        if isinstance(entry, dict):
            slug, name = entry.get("slug"), entry.get("name")
            if isinstance(slug, str) and isinstance(name, str) and name:
                names[slug] = name
    return names


def aa_name_effort(name: str) -> int:
    """How hard the run an AA display name describes thinks, 0 = not at all."""
    detail = " ".join(re.findall(r"\(([^)]*)\)", name)).lower()
    if re.search(r"non[- ]reasoning", detail):
        return 0
    levels = [AA_EFFORT_RANKS[word] for word in _EFFORT_WORD.findall(detail)]
    return max(levels) if levels else AA_DEFAULT_EFFORT


def aa_variant_siblings(slug: str, available_slugs: set[str]) -> dict[str, str]:
    """AA slugs that are another run of `slug`'s model, mapped to their suffix."""
    return {
        f"{slug}-{suffix}": suffix
        for suffix in AA_EFFORT_RANKS
        if f"{slug}-{suffix}" in available_slugs
    }


def pick_aa_variant(
    slug: str, available_slugs: set[str], claimed: set[str], aa_names: dict[str, str]
) -> str:
    """The AA slug the variant policy reads for the llm.json row `slug`.

    A sibling that is an llm.json row of its own is that row's to read. Effort
    comes from the display name when the published list has one, else from the
    slug's suffix; a bare slug with no name is AA's default run, which yields
    only to a "-reasoning"/"-thinking" sibling, since that sibling existing is
    what says the bare slug is the non-reasoning run.
    """
    siblings = {
        sibling: suffix
        for sibling, suffix in aa_variant_siblings(slug, available_slugs).items()
        if sibling not in claimed
    }
    if not siblings:
        return slug

    if slug in aa_names:
        own = aa_name_effort(aa_names[slug])
    elif any(suffix in {"reasoning", "thinking"} for suffix in siblings.values()):
        own = 0
    else:
        return slug

    best, best_effort = slug, own
    for sibling in sorted(siblings):
        name = aa_names.get(sibling)
        effort = aa_name_effort(name) if name else AA_EFFORT_RANKS[siblings[sibling]]
        if effort > best_effort:
            best, best_effort = sibling, effort
    return best


def resolve_aa_slugs(
    slugs: list[str],
    available_slugs: set[str],
    mapping_path: Path,
    aa_names: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """AA slugs to read per llm.json model, highest priority first.

    A mapping entry is a reviewer's decision and is read as written: usually
    one slug, sometimes several when Artificial Analysis tracks the same model
    under more than one, each carrying a different slice of the benchmarks. A
    model's own slug leads unless the entry places it somewhere else.

    A model without an entry reads its own slug, or -- when AA also runs it
    another way -- whichever run pick_aa_variant() chooses.
    """
    llm_to_aa = load_llm_to_aa_slugs(mapping_path)
    claimed = set(slugs)
    names = aa_names or {}
    resolved: dict[str, list[str]] = {}
    for slug in slugs:
        if slug not in llm_to_aa:
            if slug in available_slugs:
                resolved[slug] = [pick_aa_variant(slug, available_slugs, claimed, names)]
            continue
        candidates = [
            mapped for mapped in llm_to_aa[slug] if mapped in available_slugs
        ]
        if slug in available_slugs and slug not in candidates:
            candidates.insert(0, slug)
        if candidates:
            resolved[slug] = candidates
    return resolved


def aa_value_missing(value: Any) -> bool:
    # AA reports an untested benchmark as null. A 0 is a measurement -- CritPt
    # and ZeroBench floors are real -- and is kept like any other score.
    if isinstance(value, bool):
        return False
    return value is None or value == ""


def aa_field_missing(value: Any) -> bool:
    # Model fields (context, params) are a different matter: a 0 there is AA
    # not knowing, never a model with no context window.
    if isinstance(value, bool):
        return False
    return aa_value_missing(value) or value == 0


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
            if aa_field_missing(merged.get(key)) and not aa_field_missing(value):
                merged[key] = value
        for key, value in (record.get("evaluations") or {}).items():
            if aa_value_missing(evaluations.get(key)) and not aa_value_missing(value):
                evaluations[key] = value
                origins[key] = record.get("slug")
    if evaluations:
        merged["evaluations"] = evaluations
    merged["_eval_origins"] = origins
    merged["_aa_slugs"] = [record.get("slug") for record in records]
    # Complete only if every page was read: a gap from a page that failed to
    # load is not AA saying it has no number.
    merged["page_read"] = all(record.get("page_read", True) is not False for record in records)
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


def revision_model_count(by_key: dict[str, dict[str, dict[str, Any]]]) -> int:
    """Distinct models a revision-split source matched, across all its columns."""
    return len({slug for by_slug in by_key.values() for slug in by_slug})


def revision_breakdown(by_key: dict[str, dict[str, dict[str, Any]]]) -> str:
    """" (key: n, key: n)" for the summary line, or "" for a source that filled nothing."""
    if not by_key:
        return ""
    parts = ", ".join(f"{key}: {len(by_key[key])}" for key in sorted(by_key))
    return f" ({parts})"


# Values apply_score() refused this run -- out of range, or not a number. The
# run carries on without them and main() exits non-zero listing them.
REJECTED_SCORES: list[str] = []


@dataclass
class RunReports:
    """What this run's fetchers reported, for replacing scores a source dropped.

    apply_score() never overwrites with null, so a score its source stops
    listing -- the model removed from the board, or re-run under a new label --
    would otherwise sit behind that source's rank for good (issue #226). It is
    never withdrawn; instead, once its page has been read in this run and did
    not report it, any other fetcher reporting that score in the same run may
    replace it, whatever the two ranks. Only a page a fetcher writes counts
    (_precedence.is_fetcher_source): a custom source is replaceable anyway.

    Everything here is one run's: a page is "read" only if it produced a row in
    this process, so a fetch that failed or was skipped drops nothing, and a
    write refused in one run is never applied by a later one. A fetcher that
    failed partway drops nothing either: every page its failed step read, and
    any page it reports it could read only in part, is in failed_pages.
    """

    # Canonical pages some fetcher read at least one row from this run.
    read_pages: set[str] = field(default_factory=set)
    # Pages whose read failed partway: what they did not report proves nothing.
    failed_pages: set[str] = field(default_factory=set)
    # benchmark -> pages read this run through a feed that carried that
    # benchmark for no model at all (AA's API-only fields when the API sent
    # none of them): silence there says nothing about the page.
    blind: dict[str, set[str]] = field(default_factory=dict)
    # Pages read by the ingest step now running, so a failure can mark them.
    step_pages: set[str] = field(default_factory=set)
    # (model, benchmark) -> canonical pages that reported a number for it.
    reported: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    # (model, benchmark) -> writes refused by rank or fill-only, in run order:
    # (model, new value, page).
    refused: dict[tuple[str, str], list[tuple[dict[str, Any], Any, str]]] = field(
        default_factory=dict
    )

    def saw(self, slug: str, key: str, value: Any, url: str) -> None:
        page = canonical(url)
        self.read_pages.add(page)
        self.step_pages.add(page)
        if value is not None:
            self.reported.setdefault((slug, key), set()).add(page)

    def dropped(self, slug: str, key: str, stored_url: str | None) -> bool:
        """Whether the stored score's own page was read this run and no longer reports it."""
        if not is_fetcher_source(stored_url):
            return False
        page = canonical(stored_url)
        return (
            page in self.read_pages
            and page not in self.failed_pages
            and page not in self.blind.get(key, set())
            and page not in self.reported.get((slug, key), set())
        )

    def failed(self, url: str) -> None:
        self.failed_pages.add(canonical(url))

    def could_not_show(self, key: str, urls) -> None:
        """These pages were read, but nothing this run could have shown key on them."""
        self.blind.setdefault(key, set()).update(canonical(url) for url in urls)


RUN_REPORTS = RunReports()


def replace_dropped_scores(
    doc: dict[str, Any], changes: list[tuple[str, str, Any, Any]]
) -> int:
    """Apply refused writes whose blocking score its own source no longer reports.

    Called once, after every ingest of the run, so the outcome does not depend
    on whether the replacing fetcher ran before or after the one that dropped
    the score. Of several candidates the best-ranked page wins, then the first
    in run order. The score is never nulled: with no candidate it stays.
    """
    replaced = 0
    for (slug, key), offers in RUN_REPORTS.refused.items():
        model = offers[0][0]
        stored_url = score_source(model, key)
        if not RUN_REPORTS.dropped(slug, key, stored_url):
            continue
        _, new_value, url = min(
            enumerate(offers), key=lambda item: (source_rank(item[1][2]), item[0])
        )[1]
        scores = model.get("scores")
        old_value = scores.get(key) if isinstance(scores, dict) else None
        if old_value is None or old_value == new_value:
            continue
        scores[key] = new_value
        stamp_score_updated(model, key)
        stamp_score_source(model, key, url)
        changes.append((slug, key, old_value, new_value))
        replaced += 1
    return replaced


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

      * refuse a value outside the benchmark's declared range (a percentage
        unless llm.json says otherwise), or one that is not a number: a scale
        flip upstream is recorded in REJECTED_SCORES and skipped rather than
        stored, since rounding cannot tell 0.42 from 0.4;
      * round onto the benchmark's grid first, so a leaderboard that reports
        two decimals and one that reports one cannot disagree about a score the
        site prints identically either way;
      * never overwrite an existing non-null value with null;
      * never overwrite a value credited to a better-ranked source: precedence
        is declared in _precedence.py, not implied by the order main() calls
        the ingests, so which numbers land does not depend on which subset of
        the ingests ran (see the module docstring there);
      * fill_only: only fill nulls, never overwrite (the low-trust rule the
        Hugging Face and llm-stats aggregates follow) -- except a value from a
        custom source, a page no fetcher writes (_precedence.is_fetcher_source),
        which every fetcher may replace, a value credited to the very page
        being read, which is that page refreshing its own number, and an
        OpenRouter value when the fill-only source outranks it
        (_precedence.yields_to_fill_only);
      * a fetcher reporting the very number a custom source gave takes over its
        credit, so the value is attributed to a page this repo re-reads;
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
    try:
        check_score_range(doc, key, new_value, url)
        new_value = round_score(doc, key, new_value)
    except (TypeError, ValueError) as exc:
        # One bad value is that value's problem: it is refused and reported,
        # and every other score of the run still lands.
        REJECTED_SCORES.append(f"{slug} {key}: {exc}")
        print(f"error: {slug} {key}: {exc}", file=sys.stderr)
        return 0
    old_value = scores.get(key)
    if not fill_urls_only:
        RUN_REPORTS.saw(slug, key, new_value, url)

    if fill_urls_only:
        if new_value is None or old_value != new_value:
            return 0
        existing = model.get("scores_source")
        if isinstance(existing, dict) and existing.get(key) is not None:
            return 0
        stamp_score_source(model, key, url)
        changes.append((slug, key, old_value, url))
        return 1

    stored_url = score_source(model, key)
    custom = not is_fetcher_source(stored_url)
    own = stored_url is not None and canonical(stored_url) == canonical(url)
    if fill_only and old_value is not None and not (
        custom or own or yields_to_fill_only(url, stored_url)
    ):
        if new_value is not None and new_value != old_value:
            RUN_REPORTS.refused.setdefault((slug, key), []).append((model, new_value, url))
        return 0
    # Never overwrite an existing non-null value with null.
    if old_value is not None and new_value is None:
        return 0
    if old_value == new_value:
        # An equal AA measurement still takes ownership of a lower-ranked
        # score, so later non-AA updates cannot displace AA's authority -- and
        # the Coding Agent Index, ranked above the other AA surfaces, takes it
        # from them too. So does any fetcher confirming a custom source's number.
        if new_value is not None and (
            (source_rank(url) <= RANK_AA and source_rank(stored_url) > source_rank(url))
            or (custom and is_fetcher_source(url))
        ):
            stamp_score_source(model, key, url)
            changes.append((slug, key, old_value, new_value))
            return 1
        return 0
    # A stored value keeps its number until a source of at least its own
    # standing reports a different one.
    if old_value is not None and not may_overwrite(url, stored_url):
        RUN_REPORTS.refused.setdefault((slug, key), []).append((model, new_value, url))
        return 0
    scores[key] = new_value
    stamp_score_updated(model, key)
    stamp_score_source(model, key, url)
    changes.append((slug, key, old_value, new_value))
    return 1


def keep_best_by_revision(
    by_key: dict[str, dict[str, dict[str, Any]]],
    base: str,
    slug: str,
    row: dict[str, Any],
    score_key: str = "score",
) -> bool:
    """File one source row under its revision's column, best run per model.

    DeepSWE, FrontierCode and SWE-Marathon each publish more than one revision
    of themselves, and llm.json keeps a column per revision because the numbers
    are not comparable across one (see _revisions.py). A row therefore has to
    say which revision it measured before it can be written anywhere, and rows
    are ranked against their own revision only -- keep_best_row's best-run rule
    would otherwise let a retired revision's higher number displace the current
    re-run's, which is exactly the blend the split exists to end.

    Returns False for a row naming no revision, or one naming a revision this
    table has no column for; the caller counts those and reports them rather
    than folding them into a neighbouring column.
    """
    key = known_revision_key(base, row.get("revision"))
    if key is None:
        return False
    keep_best_row(by_key.setdefault(key, {}), slug, row, score_key)
    return True


def apply_revision_scores(
    doc: dict[str, Any],
    by_key: dict[str, dict[str, dict[str, Any]]],
    url: str,
    key_urls: dict[str, str] | None = None,
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    """Write every revision column one source filled, with its source page.

    Most split boards publish both revisions on one page -- Cognition toggles
    between them, Datacurve swaps the artifact behind the same leaderboard --
    so they are credited to that page and the column is what keeps them apart.
    FrontierSWE is the exception: its V1 board is preserved at its own URL, and
    a score cites the page that carries it, so `key_urls` names the page per
    column and `url` covers the rest.
    """
    models = doc.get("models", [])
    matched = 0
    updated = 0
    changes: list[tuple[str, str, Any, Any]] = []

    for model in models:
        slug = model.get("name")
        if not isinstance(slug, str) or not slug:
            continue
        hit = False
        for key, by_slug in by_key.items():
            row = by_slug.get(slug)
            if row is None:
                continue
            hit = True
            updated += apply_score(
                doc, model, slug, key, row.get("score"),
                (key_urls or {}).get(key, url), changes,
                fill_urls_only=fill_urls_only,
            )
        matched += 1 if hit else 0

    return matched, updated, changes


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


def aa_variant_record(aa_model: dict[str, Any]) -> dict[str, Any]:
    """Which Artificial Analysis run a row reads, as stored on the model.

    The slug is the one the row's AA scores lead with; the name is AA's own
    label for that run -- "GLM-4.6 (Reasoning)" -- which is what says which
    variant a column compares. Every score still names its own page in
    scores_source, which is where a gap filled from a second slug shows.
    """
    slug = aa_model.get("slug")
    name = aa_model.get("name")
    return {
        "slug": slug if isinstance(slug, str) else None,
        "name": name if isinstance(name, str) and name else None,
    }


def drop_other_variant_score(
    model: dict[str, Any],
    slug: str,
    key: str,
    read_pages: set[str],
    changes: list[tuple[str, str, Any, Any]],
) -> int:
    """Clear an AA score left behind by a run this row no longer reads.

    apply_score() never overwrites with null, which is right for a source that
    simply has no number today -- but a score credited to another AA model page
    (the row used to read a different variant, or a second checkpoint the
    mapping has since dropped) is not this row's measurement at all, and kept
    it would mix two runs in one row. Anything credited elsewhere is left alone.
    """
    scores = model.get("scores")
    if not isinstance(scores, dict) or scores.get(key) is None:
        return 0
    stored = score_source(model, key)
    if not stored or source_rank(stored) != RANK_AA:
        return 0
    stored = canonical(stored)
    if stored in read_pages or not stored.startswith(aa_model_page_url("") + "/"):
        return 0
    old_value = scores[key]
    scores[key] = None
    stamp_score_updated(model, key)
    stamp_score_source(model, key, None)
    changes.append((slug, key, old_value, None))
    return 1


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
    # Which benchmarks this run's AA data carried for any model at all, and
    # every page it was read from: see RunReports.blind.
    delivered: set[str] = set()
    all_read_pages: set[str] = set()

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

        if not fill_urls_only:
            variant = aa_variant_record(aa_model)
            if model.get("aa_variant") != variant:
                changes.append((slug, "aa_variant", model.get("aa_variant"), variant))
                model["aa_variant"] = variant
                updated += 1

        origins = aa_model.get("_eval_origins") or {}
        complete = aa_model.get("page_read", True) is not False
        read_pages = {
            aa_model_page_url(aa_slug)
            for aa_slug in aa_model.get("_aa_slugs") or [aa_model.get("slug")]
            if isinstance(aa_slug, str) and aa_slug
        }
        all_read_pages |= read_pages
        for llm_key, (aa_keys, transform) in SCORE_MAPPINGS.items():
            aa_value = None
            aa_key_used = None
            for aa_key in aa_keys:
                if aa_key in evaluations and evaluations.get(aa_key) is not None:
                    aa_value = evaluations.get(aa_key)
                    aa_key_used = aa_key
                    break
            new_value = transform(aa_value)
            if new_value is not None:
                delivered.add(llm_key)
            if new_value is None and not fill_urls_only:
                if complete:
                    updated += drop_other_variant_score(model, slug, llm_key, read_pages, changes)
                continue
            origin_slug = origins.get(aa_key_used) or aa_model.get("slug")
            url = aa_model_page_url(origin_slug)
            updated += apply_score(
                doc, model, slug, llm_key, new_value, url, changes,
                fill_urls_only=fill_urls_only,
            )

    # An API-only field (MMLU-Pro, which the model pages do not carry) arrives
    # with the API record or not at all. A run whose API answer carried one for
    # no model -- a tier without it, a field renamed upstream -- still reads
    # every page for the rest, and without this the #226 pass would take that
    # silence for AA withdrawing all of them: the 2026-09-24 11:48 refresh
    # handed 21 mmlu_pro and 22 aa_lcr rows to cards, Vals and evals.report,
    # back when AA-LCR and the tau benches were read off the API alone too.
    if not fill_urls_only:
        for llm_key in SCORE_MAPPINGS.keys() - delivered:
            RUN_REPORTS.could_not_show(llm_key, all_read_pages)

    return matched, updated, seen_eval_keys, changes


def build_fetch_aa_coding_agents_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_aa_coding_agents_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_aa_coding_agents_cmd(script)
    proc = run_fetch(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_aa_coding_agents.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected aa_coding_agents JSON format: expected a list")

    aa_coding_agents_to_slug = load_aa_coding_agents_to_slug_mapping(mapping_path)
    # slug -> {benchmark_key -> best score across agent/effort variants}
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
        slug = aa_coding_agents_to_slug.get(name)
        if not slug:
            continue
        scores = by_slug.setdefault(slug, {})
        # A model appears once per agent and effort variant; keep the best.
        if key not in scores or score > scores[key]:
            scores[key] = score
    return by_slug


def update_aa_coding_agents_scores(
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
        aa_coding_agents_scores = by_slug.get(slug)
        if not aa_coding_agents_scores:
            continue

        matched += 1
        for benchmark_key, new_value in aa_coding_agents_scores.items():
            # Ranked above every other source, AA's model pages included
            # (_precedence.RANK_AA_CODING_AGENTS), so the agent run is the one
            # that lands wherever both AA surfaces report a column.
            updated += apply_score(
                doc, model, slug, benchmark_key, new_value,
                AA_CODING_AGENTS_SOURCE_URL, changes,
                fill_urls_only=fill_urls_only,
            )

    return matched, updated, changes


def build_fetch_osworld_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_osworld_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, dict[str, Any]]]:
    """OSWorld-Verified and the OSWorld 2.0 releases, one column each.

    fetch_osworld.py names the column on every row -- the Verified workbook
    feeds osworld_verified, each tracked OSWorld 2.0 release its own column --
    so a model on several boards arrives once per board and is ranked only
    against that board's runs. One mapping file serves both boards: they are
    one lab's leaderboards and share most of their labels.
    """
    cmd = build_fetch_osworld_cmd(script)
    proc = run_fetch(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_osworld.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected osworld JSON format: expected a list")

    osworld_to_slug = load_osworld_to_slug_mapping(mapping_path)
    by_key: dict[str, dict[str, dict[str, Any]]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        key = row.get("benchmark")
        if key not in OSWORLD_KEY_URLS:
            continue
        osworld_name = row.get("model")
        if not isinstance(osworld_name, str) or not osworld_name:
            continue
        slug = osworld_to_slug.get(osworld_name)
        if not slug:
            continue
        # OSWorld 2.0 publishes a row per (reasoning effort, tool setting) run;
        # the board ranks by the best of them, so the collision rule matches it.
        keep_best_row(by_key.setdefault(key, {}), slug, row, "score")
    return by_key


def update_osworld_scores(
    doc: dict[str, Any],
    by_key: dict[str, dict[str, dict[str, Any]]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    return apply_revision_scores(
        doc, by_key, OSWORLD_KEY_URLS["osworld_verified"],
        key_urls=OSWORLD_KEY_URLS, fill_urls_only=fill_urls_only,
    )


def build_fetch_huggingface_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--all-models", "--format", "json"]


# Words that say a card label names one *run* of a benchmark rather than the
# benchmark: which harness drove it, whether tools were allowed, which subset
# or grading it used. The mapping file parks the qualified spellings it knows
# about (see `huggingface-benchmark-name-mapping.json`), and this is the net
# under it -- a label nobody has reviewed yet cannot displace a plain one.
# `pass@k` counts as a qualifier for every k but 1: best-of-k is a different
# measurement, and the gap is not small -- see docs/cybergym-coverage-2026-08.md
# for pass@1 against pass@10 on the same model.
# "No tools" is deliberately absent: every benchmark column here holds the
# no-tool run, so a label that says so names the column rather than a variant
# of it, and ranking it below a bare label would hand HLE the tools number on
# every card that prints both.
_HF_QUALIFIED_LABEL_RE = re.compile(
    r"\bw/\s*tools?\b|\bwith\s+tools?\b|\btool[- ]augmented\b"
    r"|\bharness\b|\bscaffold\b|\bterminus\b|\bopenhands\b"
    r"|\bopencode\b|\bcodex\b|\bmini-?swe\b|\bbest\b|\bstrict\b|\bpublic\b"
    r"|\bwith\s+python\b|\bw/\s*python\b|\bpass@(?!1\b)\d+\b|\bw/\s*cot\b",
    re.IGNORECASE,
)


def hf_label_rank(label: str) -> int:
    """0 for a label that names the benchmark, 1 for one that names a run of it."""
    return 1 if _HF_QUALIFIED_LABEL_RE.search(label) else 0


def fetch_huggingface_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_huggingface_cmd(script)
    proc = run_fetch(cmd)
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
        channels = row.get("channels")
        channels = channels if isinstance(channels, dict) else {}
        url = canonical(f"{fetch_huggingface.HF_BASE}/{repo}")
        if row.get("partial"):
            # A channel of the card failed to load; what it lacks is unknown.
            RUN_REPORTS.failed(url)
        mapped: dict[str, tuple[int, int, Any, str]] = {}
        for label, value in scores.items():
            key = hf_to_key.get(label)
            if not key or value is None:
                continue
            # Several card labels can alias one llm.json benchmark, and they do
            # not all name the same run. Three things decide between them, in
            # this order:
            #
            #   1. An unqualified label -- one that names the benchmark and
            #      nothing else -- is the column's own number and wins outright
            #      over one carrying a harness, a tool mode or a subset,
            #      however the values compare.
            #   2. The structured channel beats the card's own table. That is
            #      the precedence extract_scores() applies label by label, and
            #      it has to survive the mapping: "Idavidrein/gpqa (diamond)"
            #      and "GPQA Diamond" are one column read two ways, and once
            #      they are both `gpqa_diamond` nothing in the label says which
            #      came from where.
            #   3. Only between labels that tie on both does the best reported
            #      run win, which is what a card printing the same benchmark
            #      twice at two budgets asks for.
            rank = hf_label_rank(label)
            channel = 0 if channels.get(label) == fetch_huggingface.CHANNEL_METADATA else 1
            old = mapped.get(key)
            if old is None or (rank, channel) < old[:2] or (
                (rank, channel) == old[:2]
                and (not is_number(old[2]) or (is_number(value) and value > old[2]))
            ):
                mapped[key] = (rank, channel, value, url)
        if not mapped:
            continue
        merged = by_slug.setdefault(slug, {})
        for key, (rank, channel, value, source) in mapped.items():
            old = merged.get(key)
            if old is None or not is_number(old[0]) or (is_number(value) and value > old[0]):
                merged[key] = (value, source)
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
    proc = run_fetch(cmd)
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


def build_fetch_programbench_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_programbench_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_programbench_cmd(script)
    proc = run_fetch(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_programbench.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected programbench JSON format: expected a list")

    programbench_to_slug = load_programbench_to_slug_mapping(mapping_path)
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        programbench_name = row.get("model")
        if not isinstance(programbench_name, str) or not programbench_name:
            continue
        slug = programbench_to_slug.get(programbench_name)
        if not slug:
            continue
        # The board publishes a row per effort ("GPT-5.6 Sol" beside "GPT-5.6
        # Sol (xhigh)"), so one slug can collect several; the best reported run
        # wins, the same rule every other multi-row source follows.
        keep_best_row(by_slug, slug, row, "score")
    return by_slug


def update_programbench_scores(
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
        programbench_model = by_slug.get(slug)
        if programbench_model is None:
            continue

        matched += 1
        updated += apply_score(
            doc, model, slug, "programbench_almost", programbench_model.get("score"),
            PROGRAMBENCH_SOURCE_URL, changes, fill_urls_only=fill_urls_only,
        )

    return matched, updated, changes


def build_fetch_real_swe_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_real_swe_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_real_swe_cmd(script)
    proc = run_fetch(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_real_swe.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected real-swe JSON format: expected a list")

    real_swe_to_slug = load_real_swe_to_slug_mapping(mapping_path)
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        real_swe_name = row.get("model")
        if not isinstance(real_swe_name, str) or not real_swe_name:
            continue
        slug = real_swe_to_slug.get(real_swe_name)
        if not slug:
            continue
        keep_best_row(by_slug, slug, row, "score")
    return by_slug


def update_real_swe_scores(
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
        real_swe_model = by_slug.get(slug)
        if real_swe_model is None:
            continue

        matched += 1
        updated += apply_score(
            doc, model, slug, "real_swe", real_swe_model.get("score"),
            REAL_SWE_SOURCE_URL, changes, fill_urls_only=fill_urls_only,
        )

    return matched, updated, changes


def build_fetch_mcp_atlas_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_mcp_atlas_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_mcp_atlas_cmd(script)
    proc = run_fetch(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_mcp_atlas.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected mcp_atlas JSON format: expected a list")

    mcp_atlas_to_slug = load_mcp_atlas_to_slug_mapping(mapping_path)
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        mcp_atlas_name = row.get("model")
        if not isinstance(mcp_atlas_name, str) or not mcp_atlas_name:
            continue
        slug = mcp_atlas_to_slug.get(mcp_atlas_name)
        if not slug:
            continue
        keep_best_row(by_slug, slug, row, "score")
    return by_slug


def update_mcp_atlas_scores(
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
        mcp_atlas_model = by_slug.get(slug)
        if mcp_atlas_model is None:
            continue

        matched += 1
        updated += apply_score(
            doc, model, slug, "mcp_atlas", mcp_atlas_model.get("score"),
            MCP_ATLAS_SOURCE_URL, changes, fill_urls_only=fill_urls_only,
        )

    return matched, updated, changes


def build_fetch_zerobench_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_zerobench_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_zerobench_cmd(script)
    proc = run_fetch(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_zerobench.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected zerobench JSON format: expected a list")

    zerobench_to_slug = load_zerobench_to_slug_mapping(mapping_path)
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        zerobench_name = row.get("model")
        if not isinstance(zerobench_name, str) or not zerobench_name:
            continue
        slug = zerobench_to_slug.get(zerobench_name)
        if not slug:
            continue
        keep_best_row(by_slug, slug, row, "score")
    return by_slug


def update_zerobench_scores(
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
        zerobench_model = by_slug.get(slug)
        if zerobench_model is None:
            continue

        matched += 1
        updated += apply_score(
            doc, model, slug, "zerobench", zerobench_model.get("score"),
            ZEROBENCH_SOURCE_URL, changes, fill_urls_only=fill_urls_only,
        )

    return matched, updated, changes


def build_fetch_bfcl_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_bfcl_data(script: Path, mapping_path: Path) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_bfcl_cmd(script)
    proc = run_fetch(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_bfcl.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected bfcl JSON format: expected a list")

    bfcl_to_slug = load_bfcl_to_slug_mapping(mapping_path)
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        bfcl_name = row.get("model")
        if not isinstance(bfcl_name, str) or not bfcl_name:
            continue
        slug = bfcl_to_slug.get(bfcl_name)
        if not slug:
            continue
        # One mapped name covers a model's "(FC)" and "(Prompt)" rows; the
        # better of the two is the model's reported result.
        keep_best_row(by_slug, slug, row, "score")
    return by_slug


def update_bfcl_scores(
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
        bfcl_model = by_slug.get(slug)
        if bfcl_model is None:
            continue

        matched += 1
        updated += apply_score(
            doc, model, slug, "bfcl_v4", bfcl_model.get("score"),
            BFCL_SOURCE_URL, changes, fill_urls_only=fill_urls_only,
        )

    return matched, updated, changes


def build_fetch_deepswe_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_deepswe_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, dict[str, Any]]]:
    """benchlm.ai's DeepSWE mirror, filed under the revision it mirrors.

    benchlm reads one of Datacurve's artifacts and names which; rows that
    arrive without a revision are dropped rather than guessed into a column.
    """
    cmd = build_fetch_deepswe_cmd(script)
    proc = run_fetch(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_deepswe.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected deepswe JSON format: expected a list")

    deepswe_to_slug = load_deepswe_to_slug_mapping(mapping_path)
    by_key: dict[str, dict[str, dict[str, Any]]] = {}
    unversioned = 0
    for row in payload:
        if not isinstance(row, dict):
            continue
        deepswe_name = row.get("model")
        if not isinstance(deepswe_name, str) or not deepswe_name:
            continue
        slug = deepswe_to_slug.get(deepswe_name)
        if not slug:
            continue
        if not keep_best_by_revision(by_key, "deepswe", slug, row):
            unversioned += 1
    if unversioned:
        print(
            f"  skipped {unversioned} deepswe row(s) naming no known revision",
            file=sys.stderr,
        )
    return by_key


def update_deepswe_scores(
    doc: dict[str, Any],
    by_key: dict[str, dict[str, dict[str, Any]]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    return apply_revision_scores(
        doc, by_key, DEEPSWE_SOURCE_URL, fill_urls_only=fill_urls_only
    )


def build_fetch_frontierswe_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_frontierswe_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, dict[str, Any]]]:
    """FrontierSWE's boards, one column per revision the site publishes.

    The scraper reports V2 and the preserved V1 board without merging them, so
    a model on both arrives twice; each row goes to its own revision's column
    and is ranked only against that revision. The two do not share a metric --
    V2 is a mean@5 task percentage, V1 a pairwise win rate -- so a blend would
    not even be a blend of like numbers.
    """
    cmd = build_fetch_frontierswe_cmd(script)
    proc = run_fetch(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_frontierswe.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected frontierswe JSON format: expected a list")

    frontierswe_to_slug = load_frontierswe_to_slug_mapping(mapping_path)
    by_key: dict[str, dict[str, dict[str, Any]]] = {}
    unversioned = 0
    for row in payload:
        if not isinstance(row, dict):
            continue
        frontierswe_name = row.get("model")
        if not isinstance(frontierswe_name, str) or not frontierswe_name:
            continue
        slug = frontierswe_to_slug.get(frontierswe_name)
        if not slug:
            continue
        if not keep_best_by_revision(by_key, "frontierswe", slug, row):
            unversioned += 1
    if unversioned:
        print(
            f"  skipped {unversioned} frontierswe row(s) naming no known revision",
            file=sys.stderr,
        )
    return by_key


def update_frontierswe_scores(
    doc: dict[str, Any],
    by_key: dict[str, dict[str, dict[str, Any]]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    return apply_revision_scores(
        doc, by_key, FRONTIERSWE_SOURCE_URL,
        key_urls=FRONTIERSWE_KEY_URLS, fill_urls_only=fill_urls_only,
    )


def build_fetch_tbench_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_tbench_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, dict[str, Any]]]:
    """tbench.ai's boards, one column per Terminal-Bench revision.

    Each row names the column it was read for, so a model on several boards
    arrives once per board and is ranked only against that board's runs.
    """
    cmd = build_fetch_tbench_cmd(script)
    proc = run_fetch(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_tbench.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected tbench JSON format: expected a list")

    tbench_to_slug = load_tbench_to_slug_mapping(mapping_path)
    by_key: dict[str, dict[str, dict[str, Any]]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        key = row.get("benchmark")
        if key not in TBENCH_KEY_URLS:
            continue
        tbench_name = row.get("model")
        if not isinstance(tbench_name, str) or not tbench_name:
            continue
        slug = tbench_to_slug.get(tbench_name)
        if not slug:
            continue
        # One row per (agent, model, reasoning effort); the leaderboard's own
        # ranking takes the best run, so the collision rule matches it.
        keep_best_row(by_key.setdefault(key, {}), slug, row, "score")
    return by_key


def update_tbench_scores(
    doc: dict[str, Any],
    by_key: dict[str, dict[str, dict[str, Any]]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    return apply_revision_scores(
        doc, by_key, TBENCH_KEY_URLS["terminal_bench_4_0"],
        key_urls=TBENCH_KEY_URLS, fill_urls_only=fill_urls_only,
    )


def build_fetch_agents_last_exam_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_agents_last_exam_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_agents_last_exam_cmd(script)
    proc = run_fetch(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_agents_last_exam.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected agents-last-exam JSON format: expected a list")

    ale_to_slug = load_agents_last_exam_to_slug_mapping(mapping_path)
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        ale_name = row.get("model")
        if not isinstance(ale_name, str) or not ale_name:
            continue
        slug = ale_to_slug.get(ale_name)
        if not slug:
            continue
        # One row per (harness, model, harness variant); best run wins, which
        # is how the leaderboard ranks a model across its efforts.
        keep_best_row(by_slug, slug, row, "score")
    return by_slug


def update_agents_last_exam_scores(
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
        ale_model = by_slug.get(slug)
        if ale_model is None:
            continue

        matched += 1
        updated += apply_score(
            doc, model, slug, "agents_last_exam", ale_model.get("score"),
            AGENTS_LAST_EXAM_SOURCE_URL, changes, fill_urls_only=fill_urls_only,
        )

    return matched, updated, changes


def build_fetch_datacurve_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json", "--all-configs"]


def fetch_datacurve_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, dict[str, Any]]]:
    """DeepSWE scores from the benchmark's own leaderboard, one column per revision.

    Shares the benchlm mapping file: both sources label a run
    "<model>[<effort>]", so a configuration reviewed once is mapped for both.
    Every configuration of every published revision is fetched
    (--all-configs), and within a revision the best configuration that maps to
    a slug wins -- the same rule keep_best_row applies to harness variants,
    applied per revision so a 1.0 number can never outrank a 1.1 one.
    """
    cmd = build_fetch_datacurve_cmd(script)
    proc = run_fetch(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_datacurve.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected datacurve JSON format: expected a list")

    deepswe_to_slug = load_deepswe_to_slug_mapping(mapping_path)
    by_key: dict[str, dict[str, dict[str, Any]]] = {}
    unversioned = 0
    for row in payload:
        if not isinstance(row, dict):
            continue
        datacurve_name = row.get("model")
        if not isinstance(datacurve_name, str) or not datacurve_name:
            continue
        slug = deepswe_to_slug.get(datacurve_name)
        if not slug:
            continue
        if not keep_best_by_revision(by_key, "deepswe", slug, row):
            unversioned += 1
    if unversioned:
        print(
            f"  skipped {unversioned} datacurve row(s) naming no known revision",
            file=sys.stderr,
        )
    return by_key


def update_datacurve_scores(
    doc: dict[str, Any],
    by_key: dict[str, dict[str, dict[str, Any]]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    return apply_revision_scores(
        doc, by_key, DATACURVE_SOURCE_URL, fill_urls_only=fill_urls_only
    )


def build_fetch_frontiercode_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_frontiercode_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, dict[str, Any]]]:
    """Cognition's FrontierCode boards, one column per revision and task subset.

    The scraper reports every revision and every subset without merging them,
    so a model published throughout arrives four times -- 1.1 Main, 1.1
    Extended, 1.0 Main, 1.0 Extended. Each row goes to the column for its own
    board and is ranked only against that board: Extended reads about thirteen
    points above Main for the same model, so letting the two compete for one
    column would be the same blend the revision split exists to end.

    A row whose subset has no column (1.0's Extended board) or whose revision
    has none is counted and reported rather than folded into a neighbour's.
    """
    cmd = build_fetch_frontiercode_cmd(script)
    proc = run_fetch(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_frontiercode.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected frontiercode JSON format: expected a list")

    frontiercode_to_slug = load_frontiercode_to_slug_mapping(mapping_path)
    by_key: dict[str, dict[str, dict[str, Any]]] = {}
    uncolumned = 0
    for row in payload:
        if not isinstance(row, dict):
            continue
        frontiercode_name = row.get("model")
        if not isinstance(frontiercode_name, str) or not frontiercode_name:
            continue
        slug = frontiercode_to_slug.get(frontiercode_name)
        if not slug:
            continue
        base = subset_base("frontiercode", row.get("subset"))
        if base is None or not keep_best_by_revision(by_key, base, slug, row):
            uncolumned += 1
    if uncolumned:
        print(
            f"  skipped {uncolumned} frontiercode row(s) naming no known revision or subset",
            file=sys.stderr,
        )
    return by_key


def update_frontiercode_scores(
    doc: dict[str, Any],
    by_key: dict[str, dict[str, dict[str, Any]]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    return apply_revision_scores(
        doc, by_key, FRONTIERCODE_SOURCE_URL, fill_urls_only=fill_urls_only
    )


def build_fetch_swe_atlas_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--track", "all", "--format", "json"]


def fetch_swe_atlas_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_swe_atlas_cmd(script)
    proc = run_fetch(cmd)
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
    proc = run_fetch(cmd)
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


def build_fetch_vals_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--benchmark", "all", "--format", "json"]


def fetch_vals_data(script: Path, mapping_path: Path) -> dict[str, dict[str, Any]]:
    cmd = build_fetch_vals_cmd(script)
    proc = run_fetch(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_vals.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected vals JSON format: expected a list")

    vals_to_slug = load_vals_to_slug_mapping(mapping_path)
    # slug -> {benchmark_key -> best score across the rows that reach it}
    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        name = row.get("model")
        key = row.get("key")
        score = row.get("score")
        if not isinstance(name, str) or not isinstance(key, str):
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            continue
        slug = vals_to_slug.get(name)
        if not slug:
            continue
        scores = by_slug.setdefault(slug, {})
        # Two Vals paths can map to one slug (a model served by more than one
        # provider); keep the best, the same rule keep_best_row() applies
        # within a single board.
        if key not in scores or score > scores[key]:
            scores[key] = score
    return by_slug


def update_vals_scores(
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
        vals_scores = by_slug.get(slug)
        if not vals_scores:
            continue

        matched += 1
        for benchmark_key, new_value in vals_scores.items():
            updated += apply_score(
                doc, model, slug, benchmark_key, new_value,
                VALS_KEY_URLS[benchmark_key], changes,
                fill_urls_only=fill_urls_only,
            )

    return matched, updated, changes


def build_fetch_swe_marathon_cmd(script: Path) -> list[str]:
    return [sys.executable, str(script), "--format", "json"]


def fetch_swe_marathon_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, dict[str, Any]]]:
    """SWE-Marathon pass@1, one column per published board.

    The site ships a v1.0 archive beside the current v1.1 board and the two are
    not comparable, so each fills its own column. Within a board the rows are
    one per (model, scaffold, reasoning effort) and the score is
    configuration-dependent, so the model's best result on that board is what
    lands -- the same "best reported run wins" rule the evals.report ingest
    uses, applied per revision.
    """
    cmd = build_fetch_swe_marathon_cmd(script)
    proc = run_fetch(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_swe_marathon.py failed ({proc.returncode}): {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected swe_marathon JSON format: expected a list")

    swe_marathon_to_slug = load_swe_marathon_to_slug_mapping(mapping_path)
    by_key: dict[str, dict[str, dict[str, Any]]] = {}
    unversioned = 0
    for row in payload:
        if not isinstance(row, dict):
            continue
        name = row.get("model")
        if not isinstance(name, str) or not is_number(row.get("score")):
            continue
        slug = swe_marathon_to_slug.get(name)
        if not slug:
            continue
        if not keep_best_by_revision(by_key, "swe_marathon", slug, row):
            unversioned += 1
    if unversioned:
        print(
            f"  skipped {unversioned} swe_marathon row(s) naming no known revision",
            file=sys.stderr,
        )
    return by_key


def update_swe_marathon_scores(
    doc: dict[str, Any],
    by_key: dict[str, dict[str, dict[str, Any]]],
    fill_urls_only: bool = False,
) -> tuple[int, int, list[tuple[str, str, Any, Any]]]:
    return apply_revision_scores(
        doc, by_key, SWE_MARATHON_SOURCE_URL, fill_urls_only=fill_urls_only
    )


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
    proc = run_fetch(cmd)
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
    proc = run_fetch(cmd)
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


def build_fetch_openrouter_cmd(script: Path, models: list[str]) -> list[str]:
    cmd = [sys.executable, str(script), "--format", "json"]
    for model in models:
        cmd.extend(["--model", model])
    return cmd


def fetch_openrouter_data(
    script: Path, mapping_path: Path
) -> dict[str, dict[str, Any]]:
    # Per-model like Spheron: the mapped ids are the pages that get read.
    openrouter_to_slug = load_openrouter_to_slug_mapping(mapping_path)
    models = sorted(openrouter_to_slug)
    if not models:
        return {}

    cmd = build_fetch_openrouter_cmd(script, models)
    proc = run_fetch(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_openrouter.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )

    payload = json.loads(proc.stdout)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected openrouter JSON format: expected a list")

    by_slug: dict[str, dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        slug = openrouter_to_slug.get(row.get("model"))
        if not slug:
            continue
        keep_best_row(by_slug, slug, row, "score")
    return by_slug


def update_openrouter_scores(
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
        openrouter_model = by_slug.get(slug)
        if openrouter_model is None:
            continue
        source = openrouter_model.get("source")
        if not isinstance(source, str) or not source:
            continue

        matched += 1
        # Not fill-only: rank decides (_precedence.RANK_ENDPOINT_RUN), so the
        # model cards and hand entries yield to it and every other fetcher,
        # llm-stats' fill-only ingest included, takes it over.
        updated += apply_score(
            doc, model, slug, fetch_openrouter.BENCHMARK, openrouter_model.get("score"),
            source, changes, fill_urls_only=fill_urls_only,
        )

    return matched, updated, changes


def snapshot_scores(doc: dict[str, Any]) -> dict[str, tuple[dict, dict, dict]]:
    """Each model's scores, dates and sources as the run found them."""
    snapshot: dict[str, tuple[dict, dict, dict]] = {}
    for model in doc.get("models", []):
        name = model.get("name")
        if isinstance(name, str):
            snapshot[name] = tuple(
                dict(model.get(field) or {}) if isinstance(model.get(field), dict) else {}
                for field in ("scores", "scores_updated", "scores_source")
            )
    return snapshot


def undo_round_trips(
    doc: dict[str, Any],
    snapshot: dict[str, tuple[dict, dict, dict]],
    changes: list[tuple[str, str, Any, Any]],
) -> list[tuple[str, str, Any, Any]]:
    """Drop the writes that ended where they started; return the changes left.

    Two sources of equal standing that disagree on one column both write it
    every run -- evals.report's 80.9 for Llama 4 Maverick's MMLU-Pro, then Vals'
    79.4 -- so the score ends the run exactly as it began, but its date is
    restamped to today every time. A (model, benchmark) whose value and source
    are back to what the run found gets its original date back, and its
    intermediate writes leave the change table.
    """
    by_name = {m.get("name"): m for m in doc.get("models", []) if isinstance(m, dict)}
    reverted: set[tuple[str, str]] = set()
    for slug, key in {(slug, key) for slug, key, _old, _new in changes}:
        model = by_name.get(slug)
        before = snapshot.get(slug)
        if model is None or before is None or key not in before[0]:
            continue
        scores_before, updated_before, source_before = before
        if (
            (model.get("scores") or {}).get(key) == scores_before.get(key)
            and (model.get("scores_source") or {}).get(key) == source_before.get(key)
        ):
            if isinstance(model.get("scores_updated"), dict) and key in updated_before:
                model["scores_updated"][key] = updated_before[key]
            reverted.add((slug, key))
    return [change for change in changes if (change[0], change[1]) not in reverted]


def source_update(
    failures: list[str], empty: Any, label: str, fn: Callable[..., Any], *args: Any,
    **kwargs: Any,
) -> Any:
    """One source's update_*() under timed(), or `empty` when it raised.

    The ingest counterpart of source_data(): a source whose rows trip a bug
    in its ingest loses the rest of its own writes, not the whole run's. What
    it wrote before raising stays -- each of those writes went through
    apply_score() and is valid on its own.
    """
    RUN_REPORTS.step_pages = set()
    try:
        return timed(label, fn, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - any ingest bug is one source's
        # It stopped partway, so a score missing from what it read is not one
        # its source dropped (replace_dropped_scores).
        RUN_REPORTS.failed_pages |= RUN_REPORTS.step_pages
        failures.append(f"{label}: {type(exc).__name__}: {exc}")
        print(f"error: {label} failed; its remaining writes are skipped: {exc!r}", file=sys.stderr)
        return empty


def source_data(
    failures: list[str], fetch: Callable[..., dict[str, Any]], *args: Any
) -> dict[str, Any]:
    """One source's fetch_*_data(), or {} when that source failed.

    A fetcher raises -- and exits non-zero -- when its page moved: a table gone,
    a column renamed, a scale that no longer reads as a percentage. That is one
    source to look at, not a reason to throw away every other source's scores
    for the run, so the failure is recorded, the source contributes nothing,
    and main() exits WARNINGS_ONLY_EXIT once the rest are written -- a warning
    on CI, not a failed run (_fetch_warnings.py). Which values land does
    not depend on which sources ran (_precedence.py), so a skipped source only
    leaves its own columns where the last good run put them.
    """
    try:
        return fetch(*args)
    except Exception as exc:  # noqa: BLE001 - any failure is this source's
        name = fetch.__name__.removeprefix("fetch_").removesuffix("_data")
        message = str(exc).strip().splitlines()
        summary = message[-1] if message else type(exc).__name__
        failures.append(f"{name}: {summary}")
        _fetch_warnings.record("update.py", str(exc).strip() or summary, source=name)
        print(f"error: source {name} failed and is skipped this run:\n{exc}", file=sys.stderr)
        return {}


def main() -> int:
    # Against the same clock the per-phase lines use, so the phases can be read
    # against the whole and what is left over is visibly what is left over.
    run_started = time.monotonic()
    # One run's reports only: nothing a previous call saw may drop a score.
    global RUN_REPORTS
    RUN_REPORTS = RunReports()
    args = parse_args()
    if args.fill_source_urls:
        # Spheron carries VRAM estimates, not benchmark scores; the URL
        # backfill has nothing to attribute there.
        args.skip_spheron = True
    llm_path = Path(args.json_file)
    aa_path = AA_SCRIPT
    aa_coding_agents_path = AA_CODING_AGENTS_SCRIPT
    osworld_path = OSWORLD_SCRIPT
    huggingface_path = HF_SCRIPT
    deepswe_path = DEEPSWE_SCRIPT
    datacurve_path = DATACURVE_SCRIPT
    toolathlon_path = TOOLATHLON_SCRIPT
    mcp_atlas_path = MCP_ATLAS_SCRIPT
    zerobench_path = ZEROBENCH_SCRIPT
    bfcl_path = BFCL_SCRIPT
    frontierswe_path = FRONTIERSWE_SCRIPT
    programbench_path = PROGRAMBENCH_SCRIPT
    real_swe_path = REAL_SWE_SCRIPT
    tbench_path = TBENCH_SCRIPT
    agents_last_exam_path = AGENTS_LAST_EXAM_SCRIPT
    frontiercode_path = FRONTIERCODE_SCRIPT
    swe_atlas_path = SWE_ATLAS_SCRIPT
    evals_report_path = EVALS_REPORT_SCRIPT
    vals_path = VALS_SCRIPT
    swe_marathon_path = SWE_MARATHON_SCRIPT
    spheron_path = SPHERON_SCRIPT
    llmstats_path = LLMSTATS_SCRIPT
    openrouter_path = OPENROUTER_SCRIPT
    aa_coding_agents_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-aa-coding-agents-to-artificialanalysis.json"
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
    programbench_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-programbench-to-artificialanalysis.json"
    )
    real_swe_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-real-swe-to-artificialanalysis.json"
    )
    mcp_atlas_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-mcp-atlas-to-artificialanalysis.json"
    )
    zerobench_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-zerobench-to-artificialanalysis.json"
    )
    bfcl_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-bfcl-to-artificialanalysis.json"
    )
    frontierswe_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-frontierswe-to-artificialanalysis.json"
    )
    tbench_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-tbench-to-artificialanalysis.json"
    )
    agents_last_exam_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-agents-last-exam-to-artificialanalysis.json"
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
    vals_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-vals-to-artificialanalysis.json"
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
    openrouter_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-openrouter-to-artificialanalysis.json"
    )
    aa_model_mapping_path = Path(__file__).resolve().with_name(
        "model-name-mapping-llm-to-artificialanalysis.json"
    )
    llmstats_benchmark_mapping_path = Path(__file__).resolve().with_name(
        "llmstats-benchmark-name-mapping.json"
    )

    doc = timed(
        f"read {llm_path.name}",
        lambda: json.loads(llm_path.read_text(encoding="utf-8")),
    )
    models = doc.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError("Invalid JSON: models must be a list")

    # Which rows are closed reference models is decided by
    # reference-models.json, and carried in llm.json for the consumers that
    # read nothing else (llm.html, llm-cli, an export). Brought in step here,
    # before any score is written, so a slug added to the list is a reference
    # row from this run on rather than the next one.
    for name, marked in apply_reference_flags(doc):
        print(f"{name}: {'now' if marked else 'no longer'} a reference model")
    for slug in missing_reference_models(doc):
        print(
            f"Warning: reference model '{slug}' has no entry in {llm_path.name}; "
            f"add it with ./add.py --name {slug}",
            file=sys.stderr,
        )

    if args.fill_source_urls:
        # Materialize the map for every model so the file ends up with the
        # same full-key null-placeholder convention scores/scores_updated use.
        benchmark_keys = list((doc.get("benchmarks") or {}).keys())
        for model in models:
            if isinstance(model, dict):
                ensure_scores_source(model, benchmark_keys)

    slugs = unique_names(models)
    spheron_paths = sorted(load_spheron_to_slug_mapping(spheron_mapping_path))
    openrouter_models = sorted(load_openrouter_to_slug_mapping(openrouter_mapping_path))
    listed: list[list[str]] = []
    if not args.skip_aa:
        listed.append(build_list_models_cmd(aa_path))
    if not args.skip_aa_coding_agents:
        listed.append(build_fetch_aa_coding_agents_cmd(aa_coding_agents_path))
    if not args.skip_osworld:
        listed.append(build_fetch_osworld_cmd(osworld_path))
    if not args.skip_llmstats:
        listed.append(build_fetch_llmstats_cmd(llmstats_path))
    if not args.skip_openrouter and openrouter_models:
        listed.append(build_fetch_openrouter_cmd(openrouter_path, openrouter_models))
    if not args.skip_huggingface:
        listed.append(build_fetch_huggingface_cmd(huggingface_path))
    if not args.skip_toolathlon:
        listed.append(build_fetch_toolathlon_cmd(toolathlon_path))
    if not args.skip_programbench:
        listed.append(build_fetch_programbench_cmd(programbench_path))
    if not args.skip_real_swe:
        listed.append(build_fetch_real_swe_cmd(real_swe_path))
    if not args.skip_deepswe:
        listed.append(build_fetch_deepswe_cmd(deepswe_path))
    if not args.skip_datacurve:
        listed.append(build_fetch_datacurve_cmd(datacurve_path))
    if not args.skip_frontierswe:
        listed.append(build_fetch_frontierswe_cmd(frontierswe_path))
    if not args.skip_tbench:
        listed.append(build_fetch_tbench_cmd(tbench_path))
    if not args.skip_agents_last_exam:
        listed.append(build_fetch_agents_last_exam_cmd(agents_last_exam_path))
    if not args.skip_frontiercode:
        listed.append(build_fetch_frontiercode_cmd(frontiercode_path))
    if not args.skip_swe_atlas:
        listed.append(build_fetch_swe_atlas_cmd(swe_atlas_path))
    if not args.skip_evals_report:
        listed.append(build_fetch_evals_report_cmd(evals_report_path))
    if not args.skip_vals:
        listed.append(build_fetch_vals_cmd(vals_path))
    if not args.skip_swe_marathon:
        listed.append(build_fetch_swe_marathon_cmd(swe_marathon_path))
    if not args.skip_mcp_atlas:
        listed.append(build_fetch_mcp_atlas_cmd(mcp_atlas_path))
    if not args.skip_zerobench:
        listed.append(build_fetch_zerobench_cmd(zerobench_path))
    if not args.skip_bfcl:
        listed.append(build_fetch_bfcl_cmd(bfcl_path))
    if not args.skip_spheron and spheron_paths:
        listed.append(build_fetch_spheron_cmd(spheron_path, spheron_paths))

    print("commands:")
    for cmd in listed:
        print(f"  - {shlex.join(cmd)}")
    prefetch(listed)

    changes: list[tuple[str, str, Any, Any]] = []
    # Sources whose fetch failed this run; the rest still land. See source_data().
    # Kept apart from failed_sources, which is everything else that went wrong:
    # a source that could not be read is a warning, a bug here is a failure.
    fetch_failures: list[str] = []
    failed_sources: list[str] = []
    found = snapshot_scores(doc)
    available_slugs: set[str] = set()
    existing_slugs: list[str] = []
    aa_slug_by_model: dict[str, list[str]] = {}
    by_slug: dict[str, dict[str, Any]] = {}
    matched = 0
    aa_updated = 0
    seen_eval_keys: set[str] = set()

    # Artificial Analysis is the base source, but a run without it still
    # writes every other source's scores; aa_ok says whether it answered.
    aa_ok = not args.skip_aa
    if aa_ok:
        try:
            available_slugs = fetch_available_slugs(aa_path)
        except Exception as exc:  # noqa: BLE001 - AA down is one source down
            aa_ok = False
            fetch_failures.append(f"artificialanalysis: {exc}".splitlines()[0])
            _fetch_warnings.record("update.py", str(exc), source="artificialanalysis")
            print(f"error: artificialanalysis failed and is skipped this run:\n{exc}", file=sys.stderr)
    if aa_ok:
        aa_slug_by_model = resolve_aa_slugs(
            slugs, available_slugs, aa_model_mapping_path, load_aa_model_names()
        )
        existing_slugs = list(
            dict.fromkeys(
                aa_slug
                for aa_slugs in aa_slug_by_model.values()
                for aa_slug in aa_slugs
            )
        )
        print(f"  - {shlex.join(build_fetch_data_cmd(aa_path, existing_slugs))}")
    print()

    if aa_ok:
        by_aa_slug = source_data(fetch_failures, fetch_aa_data, aa_path, existing_slugs)
        by_slug = {}
        for slug, aa_slugs in aa_slug_by_model.items():
            records = [by_aa_slug[aa_slug] for aa_slug in aa_slugs if aa_slug in by_aa_slug]
            if records:
                by_slug[slug] = merge_aa_models(records)
        matched, aa_updated, seen_eval_keys, aa_changes = source_update(
            failed_sources, (0, 0, set(), []),
            "update_scores",
            update_scores,
            doc, by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(aa_changes)

    # Not fill-only: the Coding Agent Index is the strongest rung
    # (_precedence.RANK_AA_CODING_AGENTS), so it replaces what the model pages
    # above wrote for Terminal-Bench 4.0, and Datacurve/benchlm for DeepSWE and
    # Scale for SWE Atlas, wherever it reports the model. Rank, not this call's
    # position, decides that: a run that skips the model pages ends the same.
    aa_coding_agents_by_slug: dict[str, dict[str, Any]] = {}
    aa_coding_agents_matched = 0
    aa_coding_agents_updated = 0
    if not args.skip_aa_coding_agents:
        aa_coding_agents_by_slug = source_data(
            fetch_failures, fetch_aa_coding_agents_data,
            aa_coding_agents_path, aa_coding_agents_mapping_path
        )
        aa_coding_agents_matched, aa_coding_agents_updated, aa_coding_agents_changes = (
            source_update(
                failed_sources, (0, 0, []),
                "update_aa_coding_agents_scores",
                update_aa_coding_agents_scores,
                doc, aa_coding_agents_by_slug, fill_urls_only=args.fill_source_urls
            )
        )
        changes.extend(aa_coding_agents_changes)

    osworld_by_slug: dict[str, dict[str, dict[str, Any]]] = {}
    osworld_matched = 0
    osworld_updated = 0
    if not args.skip_osworld:
        osworld_by_slug = source_data(
            fetch_failures, fetch_osworld_data, osworld_path, osworld_mapping_path
        )
        osworld_matched, osworld_updated, osworld_changes = source_update(
            failed_sources, (0, 0, []),
            "update_osworld_scores",
            update_osworld_scores,
            doc, osworld_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(osworld_changes)

    llmstats_by_slug: dict[str, dict[str, Any]] = {}
    llmstats_matched = 0
    llmstats_updated = 0
    if not args.skip_llmstats:
        llmstats_by_slug = source_data(
            fetch_failures, fetch_llmstats_data,
            llmstats_path, llmstats_model_mapping_path, llmstats_benchmark_mapping_path
        )
        llmstats_matched, llmstats_updated, llmstats_changes = source_update(
            failed_sources, (0, 0, []),
            "update_llmstats_scores",
            update_llmstats_scores,
            doc, llmstats_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(llmstats_changes)

    openrouter_by_slug: dict[str, dict[str, Any]] = {}
    openrouter_matched = 0
    openrouter_updated = 0
    if not args.skip_openrouter:
        openrouter_by_slug = source_data(
            fetch_failures, fetch_openrouter_data, openrouter_path, openrouter_mapping_path
        )
        openrouter_matched, openrouter_updated, openrouter_changes = source_update(
            failed_sources, (0, 0, []),
            "update_openrouter_scores",
            update_openrouter_scores,
            doc, openrouter_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(openrouter_changes)

    huggingface_by_slug: dict[str, dict[str, Any]] = {}
    hf_matched = 0
    hf_updated = 0
    hf_params_filled = 0
    if not args.skip_huggingface:
        huggingface_by_slug = source_data(
            fetch_failures, fetch_huggingface_data, huggingface_path, huggingface_mapping_path
        )
        hf_matched, hf_updated, hf_changes = source_update(
            failed_sources, (0, 0, []),
            "update_huggingface_scores",
            update_huggingface_scores,
            doc, huggingface_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(hf_changes)
        # Runs after the AA pass above so AA's total+active pair wins. Params
        # are not scores, so the URL backfill leaves them alone.
        if not args.fill_source_urls:
            hf_params_filled, hf_params_changes = source_update(
                failed_sources, (0, []),
                "fill_missing_params_from_huggingface",
                fill_missing_params_from_huggingface,
                doc,
            )
            changes.extend(hf_params_changes)

    toolathlon_by_slug: dict[str, dict[str, Any]] = {}
    toolathlon_matched = 0
    toolathlon_updated = 0
    if not args.skip_toolathlon:
        toolathlon_by_slug = source_data(
            fetch_failures, fetch_toolathlon_data, toolathlon_path, toolathlon_mapping_path
        )
        toolathlon_matched, toolathlon_updated, toolathlon_changes = source_update(
            failed_sources, (0, 0, []),
            "update_toolathlon_scores",
            update_toolathlon_scores,
            doc, toolathlon_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(toolathlon_changes)

    programbench_by_slug: dict[str, dict[str, Any]] = {}
    programbench_matched = 0
    programbench_updated = 0
    if not args.skip_programbench:
        programbench_by_slug = source_data(
            fetch_failures, fetch_programbench_data,
            programbench_path, programbench_mapping_path
        )
        programbench_matched, programbench_updated, programbench_changes = (
            source_update(
                failed_sources, (0, 0, []),
                "update_programbench_scores",
                update_programbench_scores,
                doc, programbench_by_slug, fill_urls_only=args.fill_source_urls
            )
        )
        changes.extend(programbench_changes)

    real_swe_by_slug: dict[str, dict[str, Any]] = {}
    real_swe_matched = 0
    real_swe_updated = 0
    if not args.skip_real_swe:
        real_swe_by_slug = source_data(
            fetch_failures, fetch_real_swe_data, real_swe_path, real_swe_mapping_path
        )
        real_swe_matched, real_swe_updated, real_swe_changes = source_update(
            failed_sources, (0, 0, []),
            "update_real_swe_scores",
            update_real_swe_scores,
            doc, real_swe_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(real_swe_changes)

    deepswe_by_slug: dict[str, dict[str, Any]] = {}
    deepswe_matched = 0
    deepswe_updated = 0
    if not args.skip_deepswe:
        deepswe_by_slug = source_data(
            fetch_failures, fetch_deepswe_data, deepswe_path, deepswe_mapping_path
        )
        deepswe_matched, deepswe_updated, deepswe_changes = source_update(
            failed_sources, (0, 0, []),
            "update_deepswe_scores",
            update_deepswe_scores,
            doc, deepswe_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(deepswe_changes)

    # Runs after benchlm.ai so DeepSWE's own leaderboard wins on disagreement.
    datacurve_by_slug: dict[str, dict[str, Any]] = {}
    datacurve_matched = 0
    datacurve_updated = 0
    if not args.skip_datacurve:
        datacurve_by_slug = source_data(
            fetch_failures, fetch_datacurve_data, datacurve_path, deepswe_mapping_path
        )
        datacurve_matched, datacurve_updated, datacurve_changes = source_update(
            failed_sources, (0, 0, []),
            "update_datacurve_scores",
            update_datacurve_scores,
            doc, datacurve_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(datacurve_changes)

    frontierswe_by_slug: dict[str, dict[str, dict[str, Any]]] = {}
    frontierswe_matched = 0
    frontierswe_updated = 0
    if not args.skip_frontierswe:
        frontierswe_by_slug = source_data(
            fetch_failures, fetch_frontierswe_data, frontierswe_path, frontierswe_mapping_path
        )
        frontierswe_matched, frontierswe_updated, frontierswe_changes = source_update(
            failed_sources, (0, 0, []),
            "update_frontierswe_scores",
            update_frontierswe_scores,
            doc, frontierswe_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(frontierswe_changes)

    tbench_by_slug: dict[str, dict[str, dict[str, Any]]] = {}
    tbench_matched = 0
    tbench_updated = 0
    if not args.skip_tbench:
        tbench_by_slug = source_data(
            fetch_failures, fetch_tbench_data, tbench_path, tbench_mapping_path
        )
        tbench_matched, tbench_updated, tbench_changes = source_update(
            failed_sources, (0, 0, []),
            "update_tbench_scores",
            update_tbench_scores,
            doc, tbench_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(tbench_changes)

    agents_last_exam_by_slug: dict[str, dict[str, Any]] = {}
    agents_last_exam_matched = 0
    agents_last_exam_updated = 0
    if not args.skip_agents_last_exam:
        agents_last_exam_by_slug = source_data(
            fetch_failures, fetch_agents_last_exam_data,
            agents_last_exam_path, agents_last_exam_mapping_path
        )
        (
            agents_last_exam_matched,
            agents_last_exam_updated,
            agents_last_exam_changes,
        ) = source_update(
            failed_sources, (0, 0, []),
            "update_agents_last_exam_scores",
            update_agents_last_exam_scores,
            doc, agents_last_exam_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(agents_last_exam_changes)

    swe_atlas_by_slug: dict[str, dict[str, Any]] = {}
    swe_atlas_matched = 0
    swe_atlas_updated = 0
    if not args.skip_swe_atlas:
        swe_atlas_by_slug = source_data(
            fetch_failures, fetch_swe_atlas_data, swe_atlas_path, swe_atlas_mapping_path
        )
        swe_atlas_matched, swe_atlas_updated, swe_atlas_changes = source_update(
            failed_sources, (0, 0, []),
            "update_swe_atlas_scores",
            update_swe_atlas_scores,
            doc, swe_atlas_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(swe_atlas_changes)

    evals_report_by_slug: dict[str, dict[str, Any]] = {}
    evals_report_matched = 0
    evals_report_updated = 0
    if not args.skip_evals_report:
        evals_report_by_slug = source_data(
            fetch_failures, fetch_evals_report_data, evals_report_path, evals_report_mapping_path
        )
        evals_report_matched, evals_report_updated, evals_report_changes = source_update(
            failed_sources, (0, 0, []),
            "update_evals_report_scores",
            update_evals_report_scores,
            doc, evals_report_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(evals_report_changes)

    vals_by_slug: dict[str, dict[str, Any]] = {}
    vals_matched = 0
    vals_updated = 0
    if not args.skip_vals:
        vals_by_slug = source_data(fetch_failures, fetch_vals_data, vals_path, vals_mapping_path)
        vals_matched, vals_updated, vals_changes = source_update(
            failed_sources, (0, 0, []),
            "update_vals_scores",
            update_vals_scores,
            doc, vals_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(vals_changes)

    # Runs after evals.report so the benchmark's own site wins on disagreement.
    frontiercode_by_slug: dict[str, dict[str, Any]] = {}
    frontiercode_matched = 0
    frontiercode_updated = 0
    if not args.skip_frontiercode:
        frontiercode_by_slug = source_data(
            fetch_failures, fetch_frontiercode_data, frontiercode_path, frontiercode_mapping_path
        )
        frontiercode_matched, frontiercode_updated, frontiercode_changes = source_update(
            failed_sources, (0, 0, []),
            "update_frontiercode_scores",
            update_frontiercode_scores,
            doc, frontiercode_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(frontiercode_changes)

    # Same reason as frontiercode above.
    swe_marathon_by_slug: dict[str, Any] = {}
    swe_marathon_matched = 0
    swe_marathon_updated = 0
    if not args.skip_swe_marathon:
        swe_marathon_by_slug = source_data(
            fetch_failures, fetch_swe_marathon_data, swe_marathon_path, swe_marathon_mapping_path
        )
        swe_marathon_matched, swe_marathon_updated, swe_marathon_changes = source_update(
            failed_sources, (0, 0, []),
            "update_swe_marathon_scores",
            update_swe_marathon_scores,
            doc, swe_marathon_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(swe_marathon_changes)

    # Same reason as frontiercode above: Scale's own board wins over the
    # self-reported MCP-Atlas numbers llm-stats, evals.report and model cards
    # republish.
    mcp_atlas_by_slug: dict[str, dict[str, Any]] = {}
    mcp_atlas_matched = 0
    mcp_atlas_updated = 0
    if not args.skip_mcp_atlas:
        mcp_atlas_by_slug = source_data(
            fetch_failures, fetch_mcp_atlas_data, mcp_atlas_path, mcp_atlas_mapping_path
        )
        mcp_atlas_matched, mcp_atlas_updated, mcp_atlas_changes = source_update(
            failed_sources, (0, 0, []),
            "update_mcp_atlas_scores",
            update_mcp_atlas_scores,
            doc, mcp_atlas_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(mcp_atlas_changes)

    # And again for ZeroBench: the maintainers' own board over the model-card
    # and evals.report readings of it, which is the whole point of reading it --
    # the board has Maverick at 0.4 and Scout at 1.6 where evals.report has 0.0.
    zerobench_by_slug: dict[str, dict[str, Any]] = {}
    zerobench_matched = 0
    zerobench_updated = 0
    if not args.skip_zerobench:
        zerobench_by_slug = source_data(
            fetch_failures, fetch_zerobench_data, zerobench_path, zerobench_mapping_path
        )
        zerobench_matched, zerobench_updated, zerobench_changes = source_update(
            failed_sources, (0, 0, []),
            "update_zerobench_scores",
            update_zerobench_scores,
            doc, zerobench_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(zerobench_changes)

    # And again: BFCL's own leaderboard over evals.report's mirror of it.
    bfcl_by_slug: dict[str, dict[str, Any]] = {}
    bfcl_matched = 0
    bfcl_updated = 0
    if not args.skip_bfcl:
        bfcl_by_slug = source_data(fetch_failures, fetch_bfcl_data, bfcl_path, bfcl_mapping_path)
        bfcl_matched, bfcl_updated, bfcl_changes = source_update(
            failed_sources, (0, 0, []),
            "update_bfcl_scores",
            update_bfcl_scores,
            doc, bfcl_by_slug, fill_urls_only=args.fill_source_urls
        )
        changes.extend(bfcl_changes)

    spheron_by_slug: dict[str, dict[str, Any]] = {}
    spheron_matched = 0
    spheron_updated = 0
    if not args.skip_spheron:
        spheron_by_slug = source_data(
            fetch_failures, fetch_spheron_data, spheron_path, spheron_mapping_path
        )
        spheron_matched, spheron_updated, spheron_changes = source_update(
            failed_sources, (0, 0, []),
            "update_spheron_vram", update_spheron_vram, doc, spheron_by_slug
        )
        changes.extend(spheron_changes)

    # Last, once every source has reported: a score whose own page was read
    # this run and no longer lists it gives way to one another fetcher did.
    if not args.fill_source_urls:
        dropped_replaced = replace_dropped_scores(doc, changes)
        if dropped_replaced:
            print(f"scores replaced after their source dropped them: {dropped_replaced}")
    changes = undo_round_trips(doc, found, changes)
    missing = [slug for slug in slugs if slug not in aa_slug_by_model] if aa_ok else []
    if args.write:
        # The derived indexes are a function of the scores just fetched, so
        # they go stale the moment any of them moves. Refreshed in memory here
        # so a direct `update.py -w` leaves llm.json consistent on its own,
        # without depending on update-all's later derive step. The URL backfill
        # moves no score, and skipping the refresh keeps its diff pure.
        #
        # Neither refresh may cost the run its scores: one that raises is
        # reported and the file is written with the scores regardless, leaving
        # `./derive_indexes.py -w` (or the next run) to bring them level.
        if not args.fill_source_urls:
            source_update(
                failed_sources, (), "derive_indexes.refresh_and_report",
                derive_indexes.refresh_and_report, doc,
            )
        # Every score this run wrote is now in place with its date and source
        # page; the history is brought level with them in one pass rather than
        # each ingest remembering to log its own writes.
        source_update(failed_sources, (), "_history.sync", _history.sync, doc)
        timed(
            f"write {llm_path.name}",
            lambda: llm_path.write_text(
                json.dumps(doc, **JSON_DUMP_KWARGS) + "\n", encoding="utf-8"
            ),
        )

    print(f"models in {llm_path}: {len(slugs)}")
    if not args.skip_aa:
        print(f"models available on artificialanalysis.py: {len(existing_slugs)}")
        print(f"models returned by artificialanalysis.py: {len(by_slug)}")
    if not args.skip_aa_coding_agents:
        print(f"models returned by aa_coding_agents: {len(aa_coding_agents_by_slug)}")
    if not args.skip_osworld:
        print(f"models returned by osworld: {revision_model_count(osworld_by_slug)}" + revision_breakdown(osworld_by_slug))
    if not args.skip_llmstats:
        print(f"models returned by llmstats: {len(llmstats_by_slug)}")
    if not args.skip_openrouter:
        print(f"models returned by openrouter: {len(openrouter_by_slug)}")
    if not args.skip_huggingface:
        print(f"models returned by huggingface: {len(huggingface_by_slug)}")
    if not args.skip_toolathlon:
        print(f"models returned by toolathlon: {len(toolathlon_by_slug)}")
    if not args.skip_programbench:
        print(f"models returned by programbench: {len(programbench_by_slug)}")
    if not args.skip_real_swe:
        print(f"models returned by real-swe: {len(real_swe_by_slug)}")
    if not args.skip_deepswe:
        print(f"models returned by deepswe: {revision_model_count(deepswe_by_slug)}" + revision_breakdown(deepswe_by_slug))
    if not args.skip_datacurve:
        print(f"models returned by datacurve: {revision_model_count(datacurve_by_slug)}" + revision_breakdown(datacurve_by_slug))
    if not args.skip_frontierswe:
        print(f"models returned by frontierswe: {revision_model_count(frontierswe_by_slug)}" + revision_breakdown(frontierswe_by_slug))
    if not args.skip_tbench:
        print(f"models returned by tbench: {revision_model_count(tbench_by_slug)}" + revision_breakdown(tbench_by_slug))
    if not args.skip_agents_last_exam:
        print(f"models returned by agents-last-exam: {len(agents_last_exam_by_slug)}")
    if not args.skip_frontiercode:
        print(f"models returned by frontiercode: {revision_model_count(frontiercode_by_slug)}" + revision_breakdown(frontiercode_by_slug))
    if not args.skip_swe_atlas:
        print(f"models returned by swe_atlas: {len(swe_atlas_by_slug)}")
    if not args.skip_evals_report:
        print(f"models returned by evals_report: {len(evals_report_by_slug)}")
    if not args.skip_vals:
        print(f"models returned by vals: {len(vals_by_slug)}")
    if not args.skip_swe_marathon:
        print(f"models returned by swe_marathon: {revision_model_count(swe_marathon_by_slug)}" + revision_breakdown(swe_marathon_by_slug))
    if not args.skip_mcp_atlas:
        print(f"models returned by mcp_atlas: {len(mcp_atlas_by_slug)}")
    if not args.skip_zerobench:
        print(f"models returned by zerobench: {len(zerobench_by_slug)}")
    if not args.skip_bfcl:
        print(f"models returned by bfcl: {len(bfcl_by_slug)}")
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
    if not args.skip_aa_coding_agents:
        print(f"models matched on aa_coding_agents: {aa_coding_agents_matched}")
    if not args.skip_osworld:
        print(f"models matched on osworld: {osworld_matched}")
    if not args.skip_llmstats:
        print(f"models matched on llmstats: {llmstats_matched}")
    if not args.skip_openrouter:
        print(f"models matched on openrouter: {openrouter_matched}")
    if not args.skip_huggingface:
        print(f"models matched on huggingface: {hf_matched}")
    if not args.skip_toolathlon:
        print(f"models matched on toolathlon: {toolathlon_matched}")
    if not args.skip_programbench:
        print(f"models matched on programbench: {programbench_matched}")
    if not args.skip_real_swe:
        print(f"models matched on real-swe: {real_swe_matched}")
    if not args.skip_deepswe:
        print(f"models matched on deepswe: {deepswe_matched}")
    if not args.skip_datacurve:
        print(f"models matched on datacurve: {datacurve_matched}")
    if not args.skip_frontierswe:
        print(f"models matched on frontierswe: {frontierswe_matched}")
    if not args.skip_tbench:
        print(f"models matched on tbench: {tbench_matched}")
    if not args.skip_agents_last_exam:
        print(f"models matched on agents-last-exam: {agents_last_exam_matched}")
    if not args.skip_frontiercode:
        print(f"models matched on frontiercode: {frontiercode_matched}")
    if not args.skip_swe_atlas:
        print(f"models matched on swe_atlas: {swe_atlas_matched}")
    if not args.skip_evals_report:
        print(f"models matched on evals_report: {evals_report_matched}")
    if not args.skip_vals:
        print(f"models matched on vals: {vals_matched}")
    if not args.skip_swe_marathon:
        print(f"models matched on swe_marathon: {swe_marathon_matched}")
    if not args.skip_mcp_atlas:
        print(f"models matched on mcp_atlas: {mcp_atlas_matched}")
    if not args.skip_zerobench:
        print(f"models matched on zerobench: {zerobench_matched}")
    if not args.skip_bfcl:
        print(f"models matched on bfcl: {bfcl_matched}")
    if not args.skip_spheron:
        print(f"models matched on spheron: {spheron_matched}")
    action = "source URLs filled" if args.fill_source_urls else "score values updated"
    if not args.skip_aa:
        print(f"{action} from artificialanalysis.py: {aa_updated}")
    if not args.skip_aa_coding_agents:
        print(f"{action} from aa_coding_agents: {aa_coding_agents_updated}")
    if not args.skip_osworld:
        print(f"{action} from osworld: {osworld_updated}")
    if not args.skip_llmstats:
        print(f"{action} from llmstats: {llmstats_updated}")
    if not args.skip_openrouter:
        print(f"{action} from openrouter: {openrouter_updated}")
    if not args.skip_huggingface:
        print(f"{action} from huggingface: {hf_updated}")
        if not args.fill_source_urls:
            print(f"params values filled from huggingface: {hf_params_filled}")
    if not args.skip_toolathlon:
        print(f"{action} from toolathlon: {toolathlon_updated}")
    if not args.skip_programbench:
        print(f"{action} from programbench: {programbench_updated}")
    if not args.skip_real_swe:
        print(f"{action} from real-swe: {real_swe_updated}")
    if not args.skip_deepswe:
        print(f"{action} from deepswe: {deepswe_updated}")
    if not args.skip_datacurve:
        print(f"{action} from datacurve: {datacurve_updated}")
    if not args.skip_frontierswe:
        print(f"{action} from frontierswe: {frontierswe_updated}")
    if not args.skip_tbench:
        print(f"{action} from tbench: {tbench_updated}")
    if not args.skip_agents_last_exam:
        print(f"{action} from agents-last-exam: {agents_last_exam_updated}")
    if not args.skip_frontiercode:
        print(f"{action} from frontiercode: {frontiercode_updated}")
    if not args.skip_swe_atlas:
        print(f"{action} from swe_atlas: {swe_atlas_updated}")
    if not args.skip_evals_report:
        print(f"{action} from evals_report: {evals_report_updated}")
    if not args.skip_vals:
        print(f"{action} from vals: {vals_updated}")
    if not args.skip_swe_marathon:
        print(f"{action} from swe_marathon: {swe_marathon_updated}")
    if not args.skip_mcp_atlas:
        print(f"{action} from mcp_atlas: {mcp_atlas_updated}")
    if not args.skip_zerobench:
        print(f"{action} from zerobench: {zerobench_updated}")
    if not args.skip_bfcl:
        print(f"{action} from bfcl: {bfcl_updated}")
    if not args.skip_spheron:
        print(f"vram values updated from spheron: {spheron_updated}")
    print()
    print_changes_table(changes)
    print()
    if not args.write:
        print("dry-run only, pass --write to persist changes")
    print(f"  update.py total: {time.monotonic() - run_started:.1f}s", file=sys.stderr)
    # Scores are written above whatever failed; the exit status is how the
    # failures reach update-all and the workflow. A source that could not be
    # read alone is WARNINGS_ONLY_EXIT, which update-all reports as a warning
    # rather than a failed step; anything else is 1, and wins.
    status = 0
    if fetch_failures:
        print(file=sys.stderr)
        print(f"{len(fetch_failures)} source(s) could not be read and were skipped:", file=sys.stderr)
        for failure in fetch_failures:
            print(f"  - {failure}", file=sys.stderr)
        status = _fetch_warnings.WARNINGS_ONLY_EXIT
    if failed_sources:
        print(file=sys.stderr)
        print(f"{len(failed_sources)} step(s) failed and were skipped:", file=sys.stderr)
        for failure in failed_sources:
            print(f"  - {failure}", file=sys.stderr)
        status = 1
    if REJECTED_SCORES:
        print(file=sys.stderr)
        print(f"{len(REJECTED_SCORES)} value(s) refused and not stored:", file=sys.stderr)
        for rejected in REJECTED_SCORES:
            print(f"  - {rejected}", file=sys.stderr)
        status = 1
    return status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
