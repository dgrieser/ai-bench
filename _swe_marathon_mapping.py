"""Helper module for managing SWE-Marathon-to-slug name mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, SENTINELS, UNMAPPABLE

SWE_MARATHON_SCRIPT = Path(__file__).resolve().with_name("fetch_swe_marathon.py")
SWE_MARATHON_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-swe-marathon-to-artificialanalysis.json"
)


def fetch_swe_marathon_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(SWE_MARATHON_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_swe_marathon.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_raw_mapping(path: Path = SWE_MARATHON_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_swe_marathon_to_slug_mapping(path: Path = SWE_MARATHON_MAPPING) -> dict[str, str]:
    """Real SWE-Marathon name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_swe_marathon_names(
    path: Path = SWE_MARATHON_MAPPING, include_closed: bool = True
) -> set[str]:
    """All SWE-Marathon names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if include_closed or value != CLOSED_WEIGHTS
    }


def write_swe_marathon_to_slug_mapping(
    mapping: dict[str, str], path: Path = SWE_MARATHON_MAPPING
) -> None:
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_swe_marathon_mapping(
    swe_marathon_name: str, slug: str, path: Path = SWE_MARATHON_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(swe_marathon_name) == slug:
        return
    mapping[swe_marathon_name] = slug
    write_swe_marathon_to_slug_mapping(mapping, path)


def add_swe_marathon_unmappable(swe_marathon_name: str, path: Path = SWE_MARATHON_MAPPING) -> None:
    """Record a SWE-Marathon name as reviewed-but-unmapped so it is not prompted again."""
    add_swe_marathon_mapping(swe_marathon_name, UNMAPPABLE, path)


def add_swe_marathon_closed_weights(swe_marathon_name: str, path: Path = SWE_MARATHON_MAPPING) -> None:
    """Record a SWE-Marathon name as skipped because the source reports closed weights."""
    add_swe_marathon_mapping(swe_marathon_name, CLOSED_WEIGHTS, path)
