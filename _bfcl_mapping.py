"""Helper module for managing BFCL-to-slug name mappings.

BFCL leaderboard labels are normalized to a base model name by fetch_bfcl.py
(the "(FC)" / "(Prompt)" calling-mode parenthetical stripped), so a single
name -> slug mapping covers every mode a model is listed in. This mirrors
_toolathlon_mapping.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, PENDING, SENTINELS, UNMAPPABLE
from _prompts import freeze_decisions

BFCL_SCRIPT = Path(__file__).resolve().with_name("fetch_bfcl.py")
BFCL_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-bfcl-to-artificialanalysis.json"
)


def fetch_bfcl_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(BFCL_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_bfcl.py failed ({proc.returncode}): {proc.stderr.strip()}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def fetch_bfcl_model_openness() -> dict[str, bool | None]:
    """BFCL name -> has open weights, as read from the row's License column."""
    proc = subprocess.run(
        [sys.executable, str(BFCL_SCRIPT), "--format", "json"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"fetch_bfcl.py failed ({proc.returncode}): {proc.stderr.strip()}")
    entries: Any = json.loads(proc.stdout)
    return {
        entry["model"]: entry.get("open_weights")
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("model"), str)
    }


def _load_raw_mapping(path: Path = BFCL_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_bfcl_to_slug_mapping(path: Path = BFCL_MAPPING) -> dict[str, str]:
    """Real BFCL name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_bfcl_names(
    path: Path = BFCL_MAPPING, include_closed: bool = True
) -> set[str]:
    """All BFCL names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_bfcl_to_slug_mapping(mapping: dict[str, str], path: Path = BFCL_MAPPING) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_bfcl_mapping(bfcl_name: str, slug: str, path: Path = BFCL_MAPPING) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(bfcl_name) == slug:
        return
    mapping[bfcl_name] = slug
    write_bfcl_to_slug_mapping(mapping, path)


def add_bfcl_unmappable(bfcl_name: str, path: Path = BFCL_MAPPING) -> None:
    """Record a BFCL name as reviewed-but-unmapped so it is not prompted again."""
    add_bfcl_mapping(bfcl_name, UNMAPPABLE, path)


def add_bfcl_closed_weights(bfcl_name: str, path: Path = BFCL_MAPPING) -> None:
    """Record a BFCL name as skipped because the source reports closed weights."""
    add_bfcl_mapping(bfcl_name, CLOSED_WEIGHTS, path)
