"""Helper module for managing Real-SWE-to-slug name mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, PENDING, SENTINELS, UNMAPPABLE
from _prompts import freeze_decisions

REAL_SWE_SCRIPT = Path(__file__).resolve().with_name("fetch_real_swe.py")
REAL_SWE_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-real-swe-to-artificialanalysis.json"
)


def fetch_real_swe_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(REAL_SWE_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_real_swe.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_raw_mapping(path: Path = REAL_SWE_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_real_swe_to_slug_mapping(path: Path = REAL_SWE_MAPPING) -> dict[str, str]:
    """Real Real-SWE board names -> llm.json model slugs, sentinels dropped."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_real_swe_names(
    path: Path = REAL_SWE_MAPPING, include_closed: bool = True
) -> set[str]:
    """All Real-SWE names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_real_swe_to_slug_mapping(
    mapping: dict[str, str], path: Path = REAL_SWE_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_real_swe_mapping(
    real_swe_name: str, slug: str, path: Path = REAL_SWE_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(real_swe_name) == slug:
        return
    mapping[real_swe_name] = slug
    write_real_swe_to_slug_mapping(mapping, path)


def add_real_swe_unmappable(real_swe_name: str, path: Path = REAL_SWE_MAPPING) -> None:
    """Record a Real-SWE name as reviewed-but-unmapped so it is not prompted again."""
    add_real_swe_mapping(real_swe_name, UNMAPPABLE, path)


def add_real_swe_closed_weights(real_swe_name: str, path: Path = REAL_SWE_MAPPING) -> None:
    """Record a Real-SWE name as skipped because its weights are not published."""
    add_real_swe_mapping(real_swe_name, CLOSED_WEIGHTS, path)
