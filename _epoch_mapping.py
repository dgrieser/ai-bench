"""Helper module for managing Epoch AI-to-slug name mappings.

Epoch names a model twice: per configuration ("glm-5.2_max"), which is what
each score row measured, and per model group ("GLM-5.2"), which its
model_metadata.csv files every configuration under. The mapping is keyed by
the group, so one entry covers every reasoning effort and provider copy of a
model, and fetch_epoch.py reports the group as each row's ``model``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, PENDING, SENTINELS, UNMAPPABLE
from _prompts import freeze_decisions

EPOCH_SCRIPT = Path(__file__).resolve().with_name("fetch_epoch.py")
EPOCH_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-epoch-to-artificialanalysis.json"
)


def fetch_epoch_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(EPOCH_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_epoch.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def fetch_epoch_model_openness() -> dict[str, bool | None]:
    """Epoch model group -> has open weights, per model_metadata.csv.

    A group any of whose configurations Epoch lists as open weights counts as
    open; a group it gives no accessibility for is None, left to the index.
    """
    proc = subprocess.run(
        [sys.executable, str(EPOCH_SCRIPT), "--format", "catalogue"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_epoch.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    entries: Any = json.loads(proc.stdout)
    return {
        entry["model"]: entry.get("open_weights")
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("model"), str)
    }


def _load_raw_mapping(path: Path = EPOCH_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_epoch_to_slug_mapping(path: Path = EPOCH_MAPPING) -> dict[str, str]:
    """Real Epoch AI name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_epoch_names(
    path: Path = EPOCH_MAPPING, include_closed: bool = True
) -> set[str]:
    """All Epoch AI names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_epoch_to_slug_mapping(
    mapping: dict[str, str], path: Path = EPOCH_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_epoch_mapping(
    epoch_name: str, slug: str, path: Path = EPOCH_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(epoch_name) == slug:
        return
    mapping[epoch_name] = slug
    write_epoch_to_slug_mapping(mapping, path)


def add_epoch_unmappable(epoch_name: str, path: Path = EPOCH_MAPPING) -> None:
    """Record an Epoch AI name as reviewed-but-unmapped so it is not prompted again."""
    add_epoch_mapping(epoch_name, UNMAPPABLE, path)


def add_epoch_closed_weights(epoch_name: str, path: Path = EPOCH_MAPPING) -> None:
    """Record an Epoch AI name as skipped because the source reports closed weights."""
    add_epoch_mapping(epoch_name, CLOSED_WEIGHTS, path)
