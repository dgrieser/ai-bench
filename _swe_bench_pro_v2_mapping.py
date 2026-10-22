"""Helper module for managing SWE-Bench Pro V2-to-slug name mappings.

SWE-Bench Pro V2 leaderboard labels are normalized to a base model name by
fetch_swe_bench_pro_v2.py (harness and reasoning modifiers stripped), so a
single name -> slug mapping covers the Full and Hard tables and every
harness/effort variant. This mirrors _swe_bench_pro_v2_mapping.py.
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

SWE_BENCH_PRO_V2_SCRIPT = Path(__file__).resolve().with_name("fetch_swe_bench_pro_v2.py")
SWE_BENCH_PRO_V2_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-swe-bench-pro-v2-to-artificialanalysis.json"
)


def fetch_swe_bench_pro_v2_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(SWE_BENCH_PRO_V2_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_swe_bench_pro_v2.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def fetch_swe_bench_pro_v2_model_pages() -> dict[str, str | None]:
    """Name -> the SWE-Bench Pro V2 board it was read from, for the admin page's link.

    The same names fetch_swe_bench_pro_v2_model_names() lists.
    """
    return _source_pages.run(SWE_BENCH_PRO_V2_SCRIPT)


def _load_raw_mapping(path: Path = SWE_BENCH_PRO_V2_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_swe_bench_pro_v2_to_slug_mapping(path: Path = SWE_BENCH_PRO_V2_MAPPING) -> dict[str, str]:
    """Real SWE-Bench Pro V2 name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_swe_bench_pro_v2_names(
    path: Path = SWE_BENCH_PRO_V2_MAPPING, include_closed: bool = True
) -> set[str]:
    """All SWE-Bench Pro V2 names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_swe_bench_pro_v2_to_slug_mapping(
    mapping: dict[str, str], path: Path = SWE_BENCH_PRO_V2_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_swe_bench_pro_v2_mapping(
    swe_bench_pro_v2_name: str, slug: str, path: Path = SWE_BENCH_PRO_V2_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(swe_bench_pro_v2_name) == slug:
        return
    mapping[swe_bench_pro_v2_name] = slug
    write_swe_bench_pro_v2_to_slug_mapping(mapping, path)


def add_swe_bench_pro_v2_unmappable(swe_bench_pro_v2_name: str, path: Path = SWE_BENCH_PRO_V2_MAPPING) -> None:
    """Record a SWE-Bench Pro V2 name as reviewed-but-unmapped so it is not prompted again."""
    add_swe_bench_pro_v2_mapping(swe_bench_pro_v2_name, UNMAPPABLE, path)


def add_swe_bench_pro_v2_closed_weights(swe_bench_pro_v2_name: str, path: Path = SWE_BENCH_PRO_V2_MAPPING) -> None:
    """Record a SWE-Bench Pro V2 name as skipped because the source reports closed weights."""
    add_swe_bench_pro_v2_mapping(swe_bench_pro_v2_name, CLOSED_WEIGHTS, path)
