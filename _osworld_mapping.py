"""Helper module for managing OSWorld-to-slug name mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, PENDING, SENTINELS, UNMAPPABLE
from _prompts import freeze_decisions

OSWORLD_SCRIPT = Path(__file__).resolve().with_name("fetch_osworld.py")
OSWORLD_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-osworld-to-artificialanalysis.json"
)


def fetch_osworld_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(OSWORLD_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_osworld.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_raw_mapping(path: Path = OSWORLD_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_osworld_to_slug_mapping(path: Path = OSWORLD_MAPPING) -> dict[str, str]:
    """Real OSWorld name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_osworld_names(
    path: Path = OSWORLD_MAPPING, include_closed: bool = True
) -> set[str]:
    """All OSWorld names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_osworld_to_slug_mapping(
    mapping: dict[str, str], path: Path = OSWORLD_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_osworld_mapping(
    osworld_name: str, slug: str, path: Path = OSWORLD_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(osworld_name) == slug:
        return
    mapping[osworld_name] = slug
    write_osworld_to_slug_mapping(mapping, path)


def add_osworld_unmappable(osworld_name: str, path: Path = OSWORLD_MAPPING) -> None:
    """Record an OSWorld name as reviewed-but-unmapped so it is not prompted again."""
    add_osworld_mapping(osworld_name, UNMAPPABLE, path)


def add_osworld_closed_weights(osworld_name: str, path: Path = OSWORLD_MAPPING) -> None:
    """Record a OSWorld name as skipped because the source reports closed weights."""
    add_osworld_mapping(osworld_name, CLOSED_WEIGHTS, path)
