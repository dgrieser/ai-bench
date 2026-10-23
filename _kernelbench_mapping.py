"""Helper module for managing KernelBench-to-slug name mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, PENDING, SENTINELS, UNMAPPABLE
from _prompts import freeze_decisions
import _source_pages
import fetch_kernelbench

KERNELBENCH_SCRIPT = Path(__file__).resolve().with_name("fetch_kernelbench.py")
KERNELBENCH_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-kernelbench-to-artificialanalysis.json"
)


def fetch_kernelbench_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(KERNELBENCH_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_kernelbench.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def fetch_kernelbench_model_pages() -> dict[str, str | None]:
    """Name -> the page it was read from: one board, so one page for every name."""
    return _source_pages.same_page(fetch_kernelbench_model_names(), fetch_kernelbench.LEADERBOARD_URL)


def _load_raw_mapping(path: Path = KERNELBENCH_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_kernelbench_to_slug_mapping(path: Path = KERNELBENCH_MAPPING) -> dict[str, str]:
    """Real KernelBench name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_kernelbench_names(
    path: Path = KERNELBENCH_MAPPING, include_closed: bool = True
) -> set[str]:
    """All KernelBench names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_kernelbench_to_slug_mapping(
    mapping: dict[str, str], path: Path = KERNELBENCH_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_kernelbench_mapping(
    kernelbench_name: str, slug: str, path: Path = KERNELBENCH_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(kernelbench_name) == slug:
        return
    mapping[kernelbench_name] = slug
    write_kernelbench_to_slug_mapping(mapping, path)


def add_kernelbench_unmappable(kernelbench_name: str, path: Path = KERNELBENCH_MAPPING) -> None:
    """Record a KernelBench name as reviewed-but-unmapped so it is not prompted again."""
    add_kernelbench_mapping(kernelbench_name, UNMAPPABLE, path)


def add_kernelbench_closed_weights(kernelbench_name: str, path: Path = KERNELBENCH_MAPPING) -> None:
    """Record a KernelBench name as skipped because the source reports closed weights."""
    add_kernelbench_mapping(kernelbench_name, CLOSED_WEIGHTS, path)
