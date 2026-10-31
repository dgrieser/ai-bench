"""Helper module for managing ARC Prize-to-slug name mappings.

ARC Prize labels are normalized to a base model name by fetch_arc_prize.py (the
reasoning-effort parenthetical stripped, case folded), so a single name -> slug
mapping covers every effort a model is listed under, on every ARC-AGI version. This
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

ARC_PRIZE_SCRIPT = Path(__file__).resolve().with_name("fetch_arc_prize.py")
ARC_PRIZE_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-arc-prize-to-artificialanalysis.json"
)


def fetch_arc_prize_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(ARC_PRIZE_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_arc_prize.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def fetch_arc_prize_model_pages() -> dict[str, str | None]:
    """Name -> the page of the first board listing it: several boards, so the
    fetcher's own --format pages."""
    return _source_pages.run(ARC_PRIZE_SCRIPT)


def _load_raw_mapping(path: Path = ARC_PRIZE_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_arc_prize_to_slug_mapping(path: Path = ARC_PRIZE_MAPPING) -> dict[str, str]:
    """Real ARC Prize name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_arc_prize_names(
    path: Path = ARC_PRIZE_MAPPING, include_closed: bool = True
) -> set[str]:
    """All ARC Prize names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_arc_prize_to_slug_mapping(
    mapping: dict[str, str], path: Path = ARC_PRIZE_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_arc_prize_mapping(
    arc_prize_name: str, slug: str, path: Path = ARC_PRIZE_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(arc_prize_name) == slug:
        return
    mapping[arc_prize_name] = slug
    write_arc_prize_to_slug_mapping(mapping, path)


def add_arc_prize_unmappable(arc_prize_name: str, path: Path = ARC_PRIZE_MAPPING) -> None:
    """Record a ARC Prize name as reviewed-but-unmapped so it is not prompted again."""
    add_arc_prize_mapping(arc_prize_name, UNMAPPABLE, path)


def add_arc_prize_closed_weights(arc_prize_name: str, path: Path = ARC_PRIZE_MAPPING) -> None:
    """Record a ARC Prize name as skipped because the source reports closed weights."""
    add_arc_prize_mapping(arc_prize_name, CLOSED_WEIGHTS, path)
