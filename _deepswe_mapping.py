"""Helper module for managing DeepSWE-to-slug name mappings.

Two scrapers report DeepSWE: benchlm.ai (fetch_deepswe.py) and the benchmark's
own leaderboard (fetch_datacurve.py). Both spell a configuration the same way --
"glm-5-2[max]" -- so they share this one mapping file, and the review sees the
union of their names (fetch_all_deepswe_names).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, PENDING, SENTINELS, UNMAPPABLE
from _prompts import freeze_decisions

DEEPSWE_SCRIPT = Path(__file__).resolve().with_name("fetch_deepswe.py")
DATACURVE_SCRIPT = Path(__file__).resolve().with_name("fetch_datacurve.py")
DEEPSWE_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-deepswe-to-artificialanalysis.json"
)


def _fetch_names(script: Path) -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(script), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{script.name} failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def fetch_deepswe_model_names() -> list[str]:
    """Configuration labels benchlm.ai publishes."""
    return _fetch_names(DEEPSWE_SCRIPT)


def fetch_datacurve_model_names() -> list[str]:
    """Configuration labels the benchmark's own leaderboard publishes.

    --all-configs, because the mapping has to cover every label a score can
    arrive under, not only the best-scoring one of each model.
    """
    proc = subprocess.run(
        [sys.executable, str(DATACURVE_SCRIPT), "--format", "names", "--all-configs"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_datacurve.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def fetch_all_deepswe_names() -> list[str]:
    """Union of both DeepSWE sources' labels, benchlm first, deduped.

    One dead scraper must not block the review of the other's names, so a
    failure is reported and skipped; only both failing raises.
    """
    names: list[str] = []
    failures: list[str] = []
    for fetch in (fetch_deepswe_model_names, fetch_datacurve_model_names):
        try:
            names.extend(fetch())
        except RuntimeError as exc:
            failures.append(str(exc))
            print(f"  warning: {exc}", file=sys.stderr)
    if not names and failures:
        raise RuntimeError("; ".join(failures))
    return list(dict.fromkeys(names))


def fetch_deepswe_model_openness() -> dict[str, bool | None]:
    """DeepSWE name -> has open weights, as read from the row's sourceType."""
    proc = subprocess.run(
        [sys.executable, str(DEEPSWE_SCRIPT), "--format", "json"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_deepswe.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    entries: Any = json.loads(proc.stdout)
    return {
        entry["model"]: entry.get("open_weights")
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("model"), str)
    }


def _load_raw_mapping(path: Path = DEEPSWE_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_deepswe_to_slug_mapping(path: Path = DEEPSWE_MAPPING) -> dict[str, str]:
    """Real DeepSWE name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_deepswe_names(
    path: Path = DEEPSWE_MAPPING, include_closed: bool = True
) -> set[str]:
    """All DeepSWE names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if value != PENDING and (include_closed or value != CLOSED_WEIGHTS)
    }


def write_deepswe_to_slug_mapping(
    mapping: dict[str, str], path: Path = DEEPSWE_MAPPING
) -> None:
    # Collect mode queues the question instead of asking it; recording an answer
    # here would stop it ever being asked again.
    if freeze_decisions():
        return
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


def add_deepswe_closed_weights(deepswe_name: str, path: Path = DEEPSWE_MAPPING) -> None:
    """Record a DeepSWE name as skipped because the source reports closed weights."""
    add_deepswe_mapping(deepswe_name, CLOSED_WEIGHTS, path)
