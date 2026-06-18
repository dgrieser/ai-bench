"""Helper module for managing FrontierSWE-to-slug name mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

FRONTIERSWE_SCRIPT = Path(__file__).resolve().with_name("fetch_frontierswe.py")
FRONTIERSWE_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-frontierswe-to-artificialanalysis.json"
)

# Sentinel value stored for FrontierSWE names reviewed but deliberately not mapped.
# Kept in the mapping file so they are not prompted again, but never used as a
# real llm.json model slug.
UNMAPPABLE = "__unmappable__"


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
    return {k: v for k, v in _load_raw_mapping(path).items() if v != UNMAPPABLE}


def load_reviewed_frontierswe_names(path: Path = FRONTIERSWE_MAPPING) -> set[str]:
    """All FrontierSWE names already reviewed, including unmappable entries."""
    return set(_load_raw_mapping(path))


def write_frontierswe_to_slug_mapping(
    mapping: dict[str, str], path: Path = FRONTIERSWE_MAPPING
) -> None:
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
