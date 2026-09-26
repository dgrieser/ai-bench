"""Helper module for managing SEC-bench Pro-to-slug name mappings.

The SEC-bench Pro board names a model plainly ("Kimi K2.5", "Opus 4.6") and
keeps its agent and reasoning effort in fields of their own, so a label is
looked up as the board prints it. fetch_sec_bench.py reads one snapshot, and
the mapping is shared by whichever one that is.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, PENDING, SENTINELS, UNMAPPABLE
from _prompts import freeze_decisions

SEC_BENCH_SCRIPT = Path(__file__).resolve().with_name("fetch_sec_bench.py")
SEC_BENCH_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-sec-bench-to-artificialanalysis.json"
)


def fetch_sec_bench_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(SEC_BENCH_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_sec_bench.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_raw_mapping(path: Path = SEC_BENCH_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_sec_bench_to_slug_mapping(path: Path = SEC_BENCH_MAPPING) -> dict[str, str]:
    """Real SEC-bench Pro name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_sec_bench_names(
    path: Path = SEC_BENCH_MAPPING, include_closed: bool = True
) -> set[str]:
    """All SEC-bench Pro names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_sec_bench_to_slug_mapping(
    mapping: dict[str, str], path: Path = SEC_BENCH_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_sec_bench_mapping(
    sec_bench_name: str, slug: str, path: Path = SEC_BENCH_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(sec_bench_name) == slug:
        return
    mapping[sec_bench_name] = slug
    write_sec_bench_to_slug_mapping(mapping, path)


def add_sec_bench_unmappable(sec_bench_name: str, path: Path = SEC_BENCH_MAPPING) -> None:
    """Record a SEC-bench Pro name as reviewed-but-unmapped so it is not prompted again."""
    add_sec_bench_mapping(sec_bench_name, UNMAPPABLE, path)


def add_sec_bench_closed_weights(sec_bench_name: str, path: Path = SEC_BENCH_MAPPING) -> None:
    """Record a SEC-bench Pro name as skipped because the source reports closed weights."""
    add_sec_bench_mapping(sec_bench_name, CLOSED_WEIGHTS, path)
