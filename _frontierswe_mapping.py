"""Helper module for managing FrontierSWE-to-slug name mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, SENTINELS, UNMAPPABLE
from _prompts import freeze_decisions

FRONTIERSWE_SCRIPT = Path(__file__).resolve().with_name("fetch_frontierswe.py")
FRONTIERSWE_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-frontierswe-to-artificialanalysis.json"
)


def fetch_frontierswe_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(FRONTIERSWE_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_frontierswe.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_raw_mapping(path: Path = FRONTIERSWE_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_frontierswe_to_slug_mapping(path: Path = FRONTIERSWE_MAPPING) -> dict[str, str]:
    """Real FrontierSWE name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_frontierswe_names(
    path: Path = FRONTIERSWE_MAPPING, include_closed: bool = True
) -> set[str]:
    """All FrontierSWE names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if include_closed or value != CLOSED_WEIGHTS
    }


def write_frontierswe_to_slug_mapping(
    mapping: dict[str, str], path: Path = FRONTIERSWE_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_frontierswe_mapping(
    frontierswe_name: str, slug: str, path: Path = FRONTIERSWE_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(frontierswe_name) == slug:
        return
    mapping[frontierswe_name] = slug
    write_frontierswe_to_slug_mapping(mapping, path)


def add_frontierswe_unmappable(frontierswe_name: str, path: Path = FRONTIERSWE_MAPPING) -> None:
    """Record a FrontierSWE name as reviewed-but-unmapped so it is not prompted again."""
    add_frontierswe_mapping(frontierswe_name, UNMAPPABLE, path)


def add_frontierswe_closed_weights(frontierswe_name: str, path: Path = FRONTIERSWE_MAPPING) -> None:
    """Record a FrontierSWE name as skipped because the source reports closed weights."""
    add_frontierswe_mapping(frontierswe_name, CLOSED_WEIGHTS, path)
