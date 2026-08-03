"""Helper module for managing Toolathlon-to-slug name mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, PENDING, SENTINELS, UNMAPPABLE
from _prompts import freeze_decisions

TOOLATHLON_SCRIPT = Path(__file__).resolve().with_name("fetch_toolathlon.py")
TOOLATHLON_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-toolathlon-to-artificialanalysis.json"
)


def fetch_toolathlon_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(TOOLATHLON_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_toolathlon.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def fetch_toolathlon_model_openness() -> dict[str, bool | None]:
    """Toolathlon name -> has open weights, as read from the row's model type."""
    proc = subprocess.run(
        [sys.executable, str(TOOLATHLON_SCRIPT), "--format", "json"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_toolathlon.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    entries: Any = json.loads(proc.stdout)
    return {
        entry["model"]: entry.get("open_weights")
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("model"), str)
    }


def _load_raw_mapping(path: Path = TOOLATHLON_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_toolathlon_to_slug_mapping(
    path: Path = TOOLATHLON_MAPPING,
) -> dict[str, str]:
    """Real Toolathlon name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_toolathlon_names(
    path: Path = TOOLATHLON_MAPPING, include_closed: bool = True
) -> set[str]:
    """All Toolathlon names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_toolathlon_to_slug_mapping(
    mapping: dict[str, str], path: Path = TOOLATHLON_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_toolathlon_mapping(
    toolathlon_name: str, slug: str, path: Path = TOOLATHLON_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(toolathlon_name) == slug:
        return
    mapping[toolathlon_name] = slug
    write_toolathlon_to_slug_mapping(mapping, path)


def add_toolathlon_unmappable(
    toolathlon_name: str, path: Path = TOOLATHLON_MAPPING
) -> None:
    """Record a Toolathlon name as reviewed-but-unmapped so it is not prompted again."""
    add_toolathlon_mapping(toolathlon_name, UNMAPPABLE, path)


def add_toolathlon_closed_weights(
    toolathlon_name: str, path: Path = TOOLATHLON_MAPPING
) -> None:
    """Record a Toolathlon name as skipped because the source reports closed weights."""
    add_toolathlon_mapping(toolathlon_name, CLOSED_WEIGHTS, path)
