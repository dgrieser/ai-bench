"""Helper module for managing Vals AI-to-slug name mappings.

Keys here are Vals' own ``<provider>/<model>`` model paths, the same shape the
Spheron mapping uses, because that is the only identity Vals publishes in the
leaderboard payload. ``vals_model_label()`` gives the model half, which is what
gets compared against another source's names -- the openness index and the
review candidates both work on it, never on the provider-prefixed path.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, PENDING, SENTINELS, UNMAPPABLE
from _prompts import freeze_decisions
from fetch_vals import model_label as vals_model_label

VALS_SCRIPT = Path(__file__).resolve().with_name("fetch_vals.py")
VALS_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-vals-to-artificialanalysis.json"
)

__all__ = [
    "VALS_SCRIPT",
    "VALS_MAPPING",
    "vals_model_label",
    "fetch_vals_model_names",
    "load_vals_to_slug_mapping",
    "load_reviewed_vals_names",
    "write_vals_to_slug_mapping",
    "add_vals_mapping",
    "add_vals_unmappable",
    "add_vals_closed_weights",
]


def fetch_vals_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(VALS_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_vals.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_raw_mapping(path: Path = VALS_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_vals_to_slug_mapping(path: Path = VALS_MAPPING) -> dict[str, str]:
    """Real Vals AI model path -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_vals_names(
    path: Path = VALS_MAPPING, include_closed: bool = True
) -> set[str]:
    """All Vals AI paths already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_vals_to_slug_mapping(
    mapping: dict[str, str], path: Path = VALS_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_vals_mapping(vals_name: str, slug: str, path: Path = VALS_MAPPING) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(vals_name) == slug:
        return
    mapping[vals_name] = slug
    write_vals_to_slug_mapping(mapping, path)


def add_vals_unmappable(vals_name: str, path: Path = VALS_MAPPING) -> None:
    """Record a Vals AI path as reviewed-but-unmapped so it is not prompted again."""
    add_vals_mapping(vals_name, UNMAPPABLE, path)


def add_vals_closed_weights(vals_name: str, path: Path = VALS_MAPPING) -> None:
    """Record a Vals AI path as skipped because it names a closed-weight model."""
    add_vals_mapping(vals_name, CLOSED_WEIGHTS, path)
