"""Helper module for llm.json -> Artificial Analysis model slug mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _prompts import freeze_decisions

AA_SCRIPT = Path(__file__).resolve().with_name("artificialanalysis.py")
AA_MODEL_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-llm-to-artificialanalysis.json"
)
AA_MODEL_IGNORES = Path(__file__).resolve().with_name(
    "model-name-mapping-llm-to-artificialanalysis-ignored.json"
)


def fetch_aa_model_names(aa_script: Path = AA_SCRIPT) -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(aa_script), "--list-models"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"artificialanalysis.py --list-models failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _slug_list(value: Any) -> list[str]:
    """The AA slugs in a mapping value, highest priority first.

    A value is either a single slug or a list of them: Artificial Analysis
    sometimes tracks one model under several slugs, each carrying a different
    slice of the benchmarks, and a list reads them all.
    """
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    slugs: list[str] = []
    for item in value:
        if isinstance(item, str) and item and item not in slugs:
            slugs.append(item)
    return slugs


def _load_raw_mapping(path: Path = AA_MODEL_MAPPING) -> dict[str, str | list[str]]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {
        k: v for k, v in raw.items() if isinstance(k, str) and _slug_list(v)
    }


def load_llm_to_aa_slugs(path: Path = AA_MODEL_MAPPING) -> dict[str, list[str]]:
    """Real llm.json model slug -> Artificial Analysis slugs, best first."""
    return {k: _slug_list(v) for k, v in _load_raw_mapping(path).items()}


def load_llm_to_aa_mapping(path: Path = AA_MODEL_MAPPING) -> dict[str, str]:
    """Real llm.json model slug -> leading Artificial Analysis slug."""
    return {k: slugs[0] for k, slugs in load_llm_to_aa_slugs(path).items()}


def mapped_aa_slugs(path: Path = AA_MODEL_MAPPING) -> set[str]:
    """Every AA slug the mapping claims, leading or not."""
    return {slug for slugs in load_llm_to_aa_slugs(path).values() for slug in slugs}


def write_llm_to_aa_mapping(
    mapping: dict[str, str | list[str]], path: Path = AA_MODEL_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_aa_mapping(
    llm_name: str, aa_slug: str, path: Path = AA_MODEL_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    existing = _slug_list(mapping.get(llm_name))
    if aa_slug in existing:
        return
    if len(existing) > 1:
        # A multi-slug entry is a curated priority order; a fresh pick leads it
        # instead of replacing it.
        mapping[llm_name] = [aa_slug, *existing]
    else:
        mapping[llm_name] = aa_slug
    write_llm_to_aa_mapping(mapping, path)


def _load_ignored_raw(path: Path = AA_MODEL_IGNORES) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    out: dict[str, list[str]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, list):
            continue
        slugs = [slug for slug in value if isinstance(slug, str) and slug]
        if slugs:
            out[key] = sorted(set(slugs))
    return out


def load_ignored_aa_suggestions(
    path: Path = AA_MODEL_IGNORES,
) -> dict[str, set[str]]:
    """Rejected AA suggestions per llm.json model slug."""
    return {key: set(slugs) for key, slugs in _load_ignored_raw(path).items()}


def write_ignored_aa_suggestions(
    ignored: dict[str, set[str] | list[str]], path: Path = AA_MODEL_IGNORES
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
    payload = {
        key: sorted(set(slugs))
        for key, slugs in ignored.items()
        if isinstance(key, str) and slugs
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_ignored_aa_suggestions(
    llm_name: str, aa_slugs: list[str], path: Path = AA_MODEL_IGNORES
) -> None:
    slugs = [slug for slug in aa_slugs if slug]
    if not slugs:
        return
    ignored = load_ignored_aa_suggestions(path)
    ignored.setdefault(llm_name, set()).update(slugs)
    write_ignored_aa_suggestions(ignored, path)
