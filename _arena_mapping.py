"""Helper module for managing Arena-to-slug name mappings.

Arena labels are normalized to a base model name by fetch_arena.py (the
reasoning-effort parenthetical stripped, case folded), so a single name -> slug
mapping covers every effort a model is listed under, on every board. This
mirrors _zerobench_mapping.py, the sibling for the other board whose labels
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
import _source_pages

ARENA_SCRIPT = Path(__file__).resolve().with_name("fetch_arena.py")
ARENA_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-arena-to-artificialanalysis.json"
)


def fetch_arena_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(ARENA_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_arena.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def fetch_arena_model_pages() -> dict[str, str | None]:
    """Name -> the page of the first board listing it: several boards, so the
    fetcher's own --format pages."""
    return _source_pages.run(ARENA_SCRIPT)


def _load_raw_mapping(path: Path = ARENA_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_arena_to_slug_mapping(path: Path = ARENA_MAPPING) -> dict[str, str]:
    """Real Arena name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_arena_names(
    path: Path = ARENA_MAPPING, include_closed: bool = True
) -> set[str]:
    """All Arena names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_arena_to_slug_mapping(
    mapping: dict[str, str], path: Path = ARENA_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_arena_mapping(
    arena_name: str, slug: str, path: Path = ARENA_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(arena_name) == slug:
        return
    mapping[arena_name] = slug
    write_arena_to_slug_mapping(mapping, path)


def add_arena_unmappable(arena_name: str, path: Path = ARENA_MAPPING) -> None:
    """Record a Arena name as reviewed-but-unmapped so it is not prompted again."""
    add_arena_mapping(arena_name, UNMAPPABLE, path)


def add_arena_closed_weights(arena_name: str, path: Path = ARENA_MAPPING) -> None:
    """Record a Arena name as skipped because the source reports closed weights."""
    add_arena_mapping(arena_name, CLOSED_WEIGHTS, path)
