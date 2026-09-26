"""Helper module for managing cybergym.io-to-slug name mappings.

cybergym.io publishes CyberGym and ExploitGym from one site, and the two boards
label models the same way, so one name -> slug mapping serves both columns --
the arrangement _osworld_mapping.py has for the OSWorld boards. fetch_cybergym.py
strips the notes a label carries in a parenthetical ("GPT-5.6 Sol (reasoning
max)"), which never name the model, before the name is looked up here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, PENDING, SENTINELS, UNMAPPABLE
from _prompts import freeze_decisions

CYBERGYM_SCRIPT = Path(__file__).resolve().with_name("fetch_cybergym.py")
CYBERGYM_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-cybergym-to-artificialanalysis.json"
)


def fetch_cybergym_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(CYBERGYM_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_cybergym.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_raw_mapping(path: Path = CYBERGYM_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_cybergym_to_slug_mapping(path: Path = CYBERGYM_MAPPING) -> dict[str, str]:
    """Real cybergym.io name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_cybergym_names(
    path: Path = CYBERGYM_MAPPING, include_closed: bool = True
) -> set[str]:
    """All cybergym.io names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_cybergym_to_slug_mapping(
    mapping: dict[str, str], path: Path = CYBERGYM_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_cybergym_mapping(
    cybergym_name: str, slug: str, path: Path = CYBERGYM_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(cybergym_name) == slug:
        return
    mapping[cybergym_name] = slug
    write_cybergym_to_slug_mapping(mapping, path)


def add_cybergym_unmappable(cybergym_name: str, path: Path = CYBERGYM_MAPPING) -> None:
    """Record a cybergym.io name as reviewed-but-unmapped so it is not prompted again."""
    add_cybergym_mapping(cybergym_name, UNMAPPABLE, path)


def add_cybergym_closed_weights(cybergym_name: str, path: Path = CYBERGYM_MAPPING) -> None:
    """Record a cybergym.io name as skipped because the source reports closed weights."""
    add_cybergym_mapping(cybergym_name, CLOSED_WEIGHTS, path)
