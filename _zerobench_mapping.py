"""Helper module for managing ZeroBench-to-slug name mappings.

ZeroBench leaderboard labels are normalized to a base model name by
fetch_zerobench.py (reasoning modifiers stripped, separators unified), so a
single name -> slug mapping covers every effort a model is listed under. This
mirrors _mcp_atlas_mapping.py, the sibling for the other board whose labels
carry their run setting in a parenthetical.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, PENDING, SENTINELS, UNMAPPABLE
from _prompts import freeze_decisions

ZEROBENCH_SCRIPT = Path(__file__).resolve().with_name("fetch_zerobench.py")
ZEROBENCH_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-zerobench-to-artificialanalysis.json"
)


def fetch_zerobench_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(ZEROBENCH_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_zerobench.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_raw_mapping(path: Path = ZEROBENCH_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_zerobench_to_slug_mapping(path: Path = ZEROBENCH_MAPPING) -> dict[str, str]:
    """Real ZeroBench name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_zerobench_names(
    path: Path = ZEROBENCH_MAPPING, include_closed: bool = True
) -> set[str]:
    """All ZeroBench names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_zerobench_to_slug_mapping(
    mapping: dict[str, str], path: Path = ZEROBENCH_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_zerobench_mapping(
    zerobench_name: str, slug: str, path: Path = ZEROBENCH_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(zerobench_name) == slug:
        return
    mapping[zerobench_name] = slug
    write_zerobench_to_slug_mapping(mapping, path)


def add_zerobench_unmappable(zerobench_name: str, path: Path = ZEROBENCH_MAPPING) -> None:
    """Record a ZeroBench name as reviewed-but-unmapped so it is not prompted again."""
    add_zerobench_mapping(zerobench_name, UNMAPPABLE, path)


def add_zerobench_closed_weights(zerobench_name: str, path: Path = ZEROBENCH_MAPPING) -> None:
    """Record a ZeroBench name as skipped because the source reports closed weights."""
    add_zerobench_mapping(zerobench_name, CLOSED_WEIGHTS, path)
