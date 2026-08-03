"""Helper module for managing Spheron-to-slug name mappings.

Spheron identifies a model by its HuggingFace-style "org/model" path (e.g.
"Qwen/Qwen3-8B"). Unlike the leaderboard scrapers there is no page listing every
model, so candidate paths are derived from each llm.json model's HuggingFace
`url` (see hf_path_from_url) rather than fetched. This otherwise mirrors
_swe_atlas_mapping.py: a single Spheron path -> slug mapping, with an
__unmappable__ sentinel for closed models that have no Spheron/HF page.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from _openness import PENDING
from _prompts import freeze_decisions

SPHERON_SCRIPT = Path(__file__).resolve().with_name("fetch_spheron.py")
SPHERON_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-spheron-to-artificialanalysis.json"
)

# Sentinel value stored for Spheron paths reviewed but deliberately not mapped.
# Kept in the mapping file so they are not prompted again, but never used as a
# real llm.json model slug.
UNMAPPABLE = "__unmappable__"

# HuggingFace path segments that are not an "org" (so not a model path).
_HF_RESERVED = {"datasets", "spaces", "models", "organizations", "settings", "docs"}
_HF_PATH_RE = re.compile(r"(?:huggingface\.co|hf\.co)/([^/\s?#]+)/([^/\s?#]+)")


def hf_path_from_url(url: Any) -> str | None:
    """Extract an "org/model" Spheron path from a HuggingFace model URL.

    "https://huggingface.co/Qwen/Qwen3-8B"        -> "Qwen/Qwen3-8B"
    "https://huggingface.co/mistralai/Devstral-2-..." -> "mistralai/Devstral-2-..."
    Non-HuggingFace URLs (or dataset/space pages) return None.
    """
    if not isinstance(url, str):
        return None
    match = _HF_PATH_RE.search(url)
    if not match:
        return None
    org, model = match.group(1), match.group(2)
    if org.lower() in _HF_RESERVED:
        return None
    return f"{org}/{model}"


def _load_raw_mapping(path: Path = SPHERON_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_spheron_to_slug_mapping(path: Path = SPHERON_MAPPING) -> dict[str, str]:
    """Real Spheron path -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in (UNMAPPABLE, PENDING)}


def load_reviewed_spheron_names(path: Path = SPHERON_MAPPING) -> set[str]:
    """All Spheron paths already reviewed, excluding ones still marked __pending__."""
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING
    }


def write_spheron_to_slug_mapping(
    mapping: dict[str, str], path: Path = SPHERON_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_spheron_mapping(
    spheron_name: str, slug: str, path: Path = SPHERON_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(spheron_name) == slug:
        return
    mapping[spheron_name] = slug
    write_spheron_to_slug_mapping(mapping, path)


def add_spheron_unmappable(spheron_name: str, path: Path = SPHERON_MAPPING) -> None:
    """Record a Spheron path as reviewed-but-unmapped so it is not prompted again."""
    add_spheron_mapping(spheron_name, UNMAPPABLE, path)
