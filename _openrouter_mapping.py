"""Helper module for managing OpenRouter-to-slug name mappings.

OpenRouter names a model by its catalogue id ("anthropic/claude-opus-5.5"),
with variants (":free", ":batch") folded onto the plain id by
fetch_openrouter.py, so a single id -> slug mapping covers every variant. The
mapped ids are also the pages fetch_openrouter.py reads: like Spheron, the
source has one page per model and no board, so the mapping decides what is
fetched as well as where it lands.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, PENDING, SENTINELS, UNMAPPABLE
from _prompts import freeze_decisions

OPENROUTER_SCRIPT = Path(__file__).resolve().with_name("fetch_openrouter.py")
OPENROUTER_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-openrouter-to-artificialanalysis.json"
)


def fetch_openrouter_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(OPENROUTER_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_openrouter.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_raw_mapping(path: Path = OPENROUTER_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_openrouter_to_slug_mapping(path: Path = OPENROUTER_MAPPING) -> dict[str, str]:
    """Real OpenRouter name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_openrouter_names(
    path: Path = OPENROUTER_MAPPING, include_closed: bool = True
) -> set[str]:
    """All OpenRouter names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_openrouter_to_slug_mapping(
    mapping: dict[str, str], path: Path = OPENROUTER_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_openrouter_mapping(
    openrouter_name: str, slug: str, path: Path = OPENROUTER_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(openrouter_name) == slug:
        return
    mapping[openrouter_name] = slug
    write_openrouter_to_slug_mapping(mapping, path)


def add_openrouter_unmappable(openrouter_name: str, path: Path = OPENROUTER_MAPPING) -> None:
    """Record a OpenRouter name as reviewed-but-unmapped so it is not prompted again."""
    add_openrouter_mapping(openrouter_name, UNMAPPABLE, path)


def add_openrouter_closed_weights(openrouter_name: str, path: Path = OPENROUTER_MAPPING) -> None:
    """Record a OpenRouter name as skipped because the source reports closed weights."""
    add_openrouter_mapping(openrouter_name, CLOSED_WEIGHTS, path)
