"""Helper module for managing Hugging Face benchmark name mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import PENDING
from _prompts import freeze_decisions

HF_SCRIPT = Path(__file__).resolve().with_name("fetch_huggingface.py")
HF_MAPPING = Path(__file__).resolve().with_name("huggingface-benchmark-name-mapping.json")

# Sentinel value stored for HF labels reviewed but deliberately not mapped.
# Kept in the mapping file so they are not prompted again, but never used as a
# real llm.json benchmark key.
#
# Only two labels near the Artificial Analysis columns are unmappable because
# they name a *different* benchmark, and those two are permanent:
#
#   * "TAU3-Bench" is the τ³ cross-domain aggregate, not the Banking domain that
#     tau3_bench_banking tracks -- one card reports 67.2 where AA's Banking
#     number for the same model is 8.7.
#   * "Long Context" is not AA-LCR; its values reach 99.3 against AA-LCR's 74.7
#     ceiling, so it is some other long-context test.
#
# The rest ("AA-LCR", "CritPt", "τ³-Banking", "GDPval-AA v2 (Elo)", "Toolathlon",
# "Toolathlon-Verified") are mapped, and can only ever fill a gap:
# update_huggingface_scores() writes into nulls and never overwrites, and every
# one of those columns has a first-party source that runs before or after it and
# does overwrite. Worth knowing when reading a filled-in value, though: these
# self-reports agree with the first-party number in only 4 of 17 cases -- AA-LCR
# is the worst, with 13 of 14 cards off by up to 5 points because the labs ran it
# themselves -- and "Toolathlon" without the suffix is the pre-Verified series,
# which is a different score series (see fetch_toolathlon.py).
#
# "Multimodal software engineering SWE-MM" is mapped on the same terms, and is
# the loosest of them: the one card reporting it (Qwen3.8-27B) footnotes the
# public *dev* split of SWE-bench Multimodal run on the Claude Code harness with
# the modifications from Appendix 8.3 of the Claude Opus 4.7 system card, so it
# is not the official test-split number the leaderboard reports. It is mapped
# rather than parked because it names the same benchmark -- the bar for
# __unmappable__ is a label naming a *different* one, and the harness variants
# ("SWE-Bench Verified (OpenHands)" and friends) are mapped already -- and
# because fetch_evals_report.py runs after this ingest and does overwrite, so an
# official run displaces the self-report as soon as one exists.
#
# The mmlu_pro and swe_bench_multilingual aliases were mapped on the same terms
# (evals.report carries both, and runs after this ingest). Two things to know
# when reading a value that came from here:
#
#   * A card can report the same benchmark twice -- once as structured Hub eval
#     metadata under the dataset id, once as a README table cell -- and the two
#     do not always agree. extract_scores() lets the structured value win, which
#     is the channel a benchmark owner can also write to via a Hub PR. On the 27
#     models reporting MMLU-Pro both ways 25 agree exactly, but deepseek-v4-pro
#     reports 87.5 in metadata against 73.5 in its own table.
#   * Scale is not normalized anywhere in the ingest, so a label is only mapped
#     when every card reporting it uses percentage points. That is why
#     "Multilingual MMLU" and "MMLU 5-shot" stay parked: the Ministral cards
#     report them as 0-1 fractions (0.742, 0.794), which would land in llm.json
#     as sub-1% scores. "MMLU-ProX" and "KMMLU-Pro" stay parked for the ordinary
#     reason -- a multilingual extension and a Korean one are not MMLU-Pro.
UNMAPPABLE = "__unmappable__"


def fetch_huggingface_benchmark_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(HF_SCRIPT), "--all-models", "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_huggingface.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_raw_mapping(path: Path = HF_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_hf_to_key_mapping(path: Path = HF_MAPPING) -> dict[str, str]:
    """Real HF label -> llm.json key mappings (excludes unmappable entries)."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in (UNMAPPABLE, PENDING)}


def load_reviewed_hf_labels(path: Path = HF_MAPPING) -> set[str]:
    """All HF labels already reviewed, excluding ones still marked __pending__."""
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING
    }


def write_hf_to_key_mapping(mapping: dict[str, str], path: Path = HF_MAPPING) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_hf_mapping(hf_label: str, key: str, path: Path = HF_MAPPING) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(hf_label) == key:
        return
    mapping[hf_label] = key
    write_hf_to_key_mapping(mapping, path)


def add_hf_unmappable(hf_label: str, path: Path = HF_MAPPING) -> None:
    """Record an HF label as reviewed-but-unmapped so it is not prompted again."""
    add_hf_mapping(hf_label, UNMAPPABLE, path)
