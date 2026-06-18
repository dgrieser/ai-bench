"""Helper module for managing DeepSWE-to-slug name mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEEPSWE_SCRIPT = Path(__file__).resolve().with_name("fetch_deepswe.py")
DEEPSWE_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-deepswe-to-artificialanalysis.json"
)

# Sentinel value stored for DeepSWE names reviewed but deliberately not mapped.
# Kept in the mapping file so they are not prompted again, but never used as a
# real llm.json model slug.
UNMAPPABLE = "__unmappable__"


def fetch_deepswe_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(DEEPSWE_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_deepswe.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_raw_mapping(path: Path = DEEPSWE_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_deepswe_to_slug_mapping(path: Path = DEEPSWE_MAPPING) -> dict[str, str]:
    """Real DeepSWE name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v != UNMAPPABLE}


def load_reviewed_deepswe_names(path: Path = DEEPSWE_MAPPING) -> set[str]:
    """All DeepSWE names already reviewed, including unmappable entries."""
    return set(_load_raw_mapping(path))


def write_deepswe_to_slug_mapping(
    mapping: dict[str, str], path: Path = DEEPSWE_MAPPING
) -> None:
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_deepswe_mapping(
    deepswe_name: str, slug: str, path: Path = DEEPSWE_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(deepswe_name) == slug:
        return
    mapping[deepswe_name] = slug
    write_deepswe_to_slug_mapping(mapping, path)


def add_deepswe_unmappable(deepswe_name: str, path: Path = DEEPSWE_MAPPING) -> None:
    """Record a DeepSWE name as reviewed-but-unmapped so it is not prompted again."""
    add_deepswe_mapping(deepswe_name, UNMAPPABLE, path)
