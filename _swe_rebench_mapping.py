#!/usr/bin/env python3
"""Helpers for SWE-Rebench name mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, SENTINELS, UNMAPPABLE

SWE_REBENCH_SCRIPT = Path(__file__).resolve().with_name("fetch_swe_rebench.py")
SWE_REBENCH_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-rebench-to-artificialanalysis.json"
)


def fetch_swe_rebench_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(SWE_REBENCH_SCRIPT), "--all-models", "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_swe_rebench.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_raw_mapping(mapping_path: Path = SWE_REBENCH_MAPPING) -> dict[str, str]:
    if not mapping_path.exists():
        return {}
    raw: Any = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Rebench mapping must be a JSON object.")

    mapping: dict[str, str] = {}
    for rebench_name, aa_slug in raw.items():
        if isinstance(rebench_name, str) and rebench_name and isinstance(aa_slug, str) and aa_slug:
            mapping[rebench_name] = aa_slug
    return mapping


def load_rebench_to_slug_mapping(mapping_path: Path = SWE_REBENCH_MAPPING) -> dict[str, str]:
    """Real SWE-Rebench name -> llm.json model slug mappings."""
    return {
        rebench_name: aa_slug
        for rebench_name, aa_slug in _load_raw_mapping(mapping_path).items()
        if aa_slug not in SENTINELS
    }


def load_reviewed_rebench_names(
    mapping_path: Path = SWE_REBENCH_MAPPING, include_closed: bool = True
) -> set[str]:
    """All SWE-Rebench names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(mapping_path).items()
        if include_closed or value != CLOSED_WEIGHTS
    }


def write_rebench_to_slug_mapping(
    mapping: dict[str, str], mapping_path: Path = SWE_REBENCH_MAPPING
) -> None:
    mapping_path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def add_rebench_mapping(
    rebench_name: str, aa_slug: str, mapping_path: Path = SWE_REBENCH_MAPPING
) -> None:
    mapping = _load_raw_mapping(mapping_path)
    if mapping.get(rebench_name) == aa_slug:
        return
    mapping[rebench_name] = aa_slug
    write_rebench_to_slug_mapping(mapping, mapping_path)


def add_rebench_unmappable(rebench_name: str, mapping_path: Path = SWE_REBENCH_MAPPING) -> None:
    """Record a SWE-Rebench name as reviewed-but-unmapped so it is not prompted again."""
    add_rebench_mapping(rebench_name, UNMAPPABLE, mapping_path)


def add_rebench_closed_weights(rebench_name: str, mapping_path: Path = SWE_REBENCH_MAPPING) -> None:
    """Record a SWE-Rebench name as skipped because the source reports closed weights."""
    add_rebench_mapping(rebench_name, CLOSED_WEIGHTS, mapping_path)
