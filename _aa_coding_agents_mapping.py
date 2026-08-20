"""Helper module for managing Coding Agent Index-to-slug name mappings.

Coding Agent Index row labels are normalized to a base model name by
fetch_aa_coding_agents.py (agent and effort/mode parentheticals stripped), so a
single name -> slug mapping covers every agent/effort variant. This mirrors
_swe_atlas_mapping.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, PENDING, SENTINELS, UNMAPPABLE
from _prompts import freeze_decisions

AA_CODING_AGENTS_SCRIPT = Path(__file__).resolve().with_name("fetch_aa_coding_agents.py")
AA_CODING_AGENTS_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-aa-coding-agents-to-artificialanalysis.json"
)


def fetch_aa_coding_agents_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(AA_CODING_AGENTS_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_aa_coding_agents.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_raw_mapping(path: Path = AA_CODING_AGENTS_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_aa_coding_agents_to_slug_mapping(
    path: Path = AA_CODING_AGENTS_MAPPING,
) -> dict[str, str]:
    """Real Coding Agent Index name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_aa_coding_agents_names(
    path: Path = AA_CODING_AGENTS_MAPPING, include_closed: bool = True
) -> set[str]:
    """All Coding Agent Index names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_aa_coding_agents_to_slug_mapping(
    mapping: dict[str, str], path: Path = AA_CODING_AGENTS_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_aa_coding_agents_mapping(
    aa_coding_agents_name: str, slug: str, path: Path = AA_CODING_AGENTS_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(aa_coding_agents_name) == slug:
        return
    mapping[aa_coding_agents_name] = slug
    write_aa_coding_agents_to_slug_mapping(mapping, path)


def add_aa_coding_agents_unmappable(
    aa_coding_agents_name: str, path: Path = AA_CODING_AGENTS_MAPPING
) -> None:
    """Record a name as reviewed-but-unmapped so it is not prompted again."""
    add_aa_coding_agents_mapping(aa_coding_agents_name, UNMAPPABLE, path)


def add_aa_coding_agents_closed_weights(
    aa_coding_agents_name: str, path: Path = AA_CODING_AGENTS_MAPPING
) -> None:
    """Record a name as skipped because the source reports closed weights."""
    add_aa_coding_agents_mapping(aa_coding_agents_name, CLOSED_WEIGHTS, path)
