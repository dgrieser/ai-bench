"""Helper module for managing Agents' Last Exam-to-slug name mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, PENDING, SENTINELS, UNMAPPABLE
from _prompts import freeze_decisions

AGENTS_LAST_EXAM_SCRIPT = Path(__file__).resolve().with_name("fetch_agents_last_exam.py")
AGENTS_LAST_EXAM_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-agents-last-exam-to-artificialanalysis.json"
)


def fetch_agents_last_exam_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(AGENTS_LAST_EXAM_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_agents_last_exam.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_raw_mapping(path: Path = AGENTS_LAST_EXAM_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_agents_last_exam_to_slug_mapping(
    path: Path = AGENTS_LAST_EXAM_MAPPING,
) -> dict[str, str]:
    """Real Agents' Last Exam name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_agents_last_exam_names(
    path: Path = AGENTS_LAST_EXAM_MAPPING, include_closed: bool = True
) -> set[str]:
    """All Agents' Last Exam names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_agents_last_exam_to_slug_mapping(
    mapping: dict[str, str], path: Path = AGENTS_LAST_EXAM_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_agents_last_exam_mapping(
    agents_last_exam_name: str, slug: str, path: Path = AGENTS_LAST_EXAM_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(agents_last_exam_name) == slug:
        return
    mapping[agents_last_exam_name] = slug
    write_agents_last_exam_to_slug_mapping(mapping, path)


def add_agents_last_exam_unmappable(
    agents_last_exam_name: str, path: Path = AGENTS_LAST_EXAM_MAPPING
) -> None:
    """Record a Agents' Last Exam name as reviewed-but-unmapped so it is not prompted again."""
    add_agents_last_exam_mapping(agents_last_exam_name, UNMAPPABLE, path)


def add_agents_last_exam_closed_weights(
    agents_last_exam_name: str, path: Path = AGENTS_LAST_EXAM_MAPPING
) -> None:
    """Record a Agents' Last Exam name as skipped because the source reports closed weights."""
    add_agents_last_exam_mapping(agents_last_exam_name, CLOSED_WEIGHTS, path)
