"""Helper module for managing OSWorld-to-slug name mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

OSWORLD_SCRIPT = Path(__file__).resolve().with_name("fetch_osworld.py")
OSWORLD_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-osworld-to-artificialanalysis.json"
)


def fetch_osworld_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(OSWORLD_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_osworld.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def load_osworld_to_slug_mapping(path: Path = OSWORLD_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def write_osworld_to_slug_mapping(
    mapping: dict[str, str], path: Path = OSWORLD_MAPPING
) -> None:
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_osworld_mapping(
    osworld_name: str, slug: str, path: Path = OSWORLD_MAPPING
) -> None:
    mapping = load_osworld_to_slug_mapping(path)
    if mapping.get(osworld_name) == slug:
        return
    mapping[osworld_name] = slug
    write_osworld_to_slug_mapping(mapping, path)
