"""Helper module for managing Hugging Face benchmark name mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _prompts import freeze_decisions

HF_SCRIPT = Path(__file__).resolve().with_name("fetch_huggingface.py")
HF_MAPPING = Path(__file__).resolve().with_name("huggingface-benchmark-name-mapping.json")

# Sentinel value stored for HF labels reviewed but deliberately not mapped.
# Kept in the mapping file so they are not prompted again, but never used as a
# real llm.json benchmark key.
#
# Some of those labels look like they should obviously be mapped now that the
# benchmark has a column -- "AA-LCR", "CritPt", "τ³-Banking", "GDPval-AA v2
# (Elo)", "Toolathlon-Verified". They stay unmappable on purpose:
#
#   * Those columns are defined as Artificial Analysis' own runs, and AA already
#     covers every model whose card quotes them, so a mapping adds no model.
#   * Only 4 of 17 such self-reported values match AA's published number.
#     AA-LCR is the worst: 13 of 14 cards differ, by up to 5 points, because the
#     labs ran it themselves rather than citing AA.
#   * Hugging Face is applied after Artificial Analysis in update.py, so a
#     mapping would let a stale or differently-harnessed self-report overwrite
#     the live AA value rather than fill a gap.
#
# Two more are wrong for their own reasons: "TAU3-Bench" is the cross-domain
# aggregate (one card reports 67.2 where AA's Banking domain gives 8.7), and
# "Long Context" is not AA-LCR at all (values up to 99.3 against AA-LCR's 74.7
# ceiling). "Toolathlon" is the pre-Verified series -- see fetch_toolathlon.py.
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
    return {k: v for k, v in _load_raw_mapping(path).items() if v != UNMAPPABLE}


def load_reviewed_hf_labels(path: Path = HF_MAPPING) -> set[str]:
    """All HF labels already reviewed, including ones recorded as unmappable."""
    return set(_load_raw_mapping(path))


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
