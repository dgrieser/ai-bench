"""Helper module for managing SWE Atlas-to-slug name mappings.

SWE Atlas leaderboard labels are normalized to a base model name by
fetch_swe_atlas.py (harness and reasoning modifiers stripped), so a single
name -> slug mapping covers every track/harness variant. This mirrors
_frontierswe_mapping.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, PENDING, SENTINELS, UNMAPPABLE
from _prompts import freeze_decisions

SWE_ATLAS_SCRIPT = Path(__file__).resolve().with_name("fetch_swe_atlas.py")
SWE_ATLAS_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-swe-atlas-to-artificialanalysis.json"
)


def fetch_swe_atlas_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(SWE_ATLAS_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_swe_atlas.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_raw_mapping(path: Path = SWE_ATLAS_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_swe_atlas_to_slug_mapping(path: Path = SWE_ATLAS_MAPPING) -> dict[str, str]:
    """Real SWE Atlas name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_swe_atlas_names(
    path: Path = SWE_ATLAS_MAPPING, include_closed: bool = True
) -> set[str]:
    """All SWE Atlas names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_swe_atlas_to_slug_mapping(
    mapping: dict[str, str], path: Path = SWE_ATLAS_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_swe_atlas_mapping(
    swe_atlas_name: str, slug: str, path: Path = SWE_ATLAS_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(swe_atlas_name) == slug:
        return
    mapping[swe_atlas_name] = slug
    write_swe_atlas_to_slug_mapping(mapping, path)


def add_swe_atlas_unmappable(swe_atlas_name: str, path: Path = SWE_ATLAS_MAPPING) -> None:
    """Record a SWE Atlas name as reviewed-but-unmapped so it is not prompted again."""
    add_swe_atlas_mapping(swe_atlas_name, UNMAPPABLE, path)


def add_swe_atlas_closed_weights(swe_atlas_name: str, path: Path = SWE_ATLAS_MAPPING) -> None:
    """Record a SWE Atlas name as skipped because the source reports closed weights."""
    add_swe_atlas_mapping(swe_atlas_name, CLOSED_WEIGHTS, path)
