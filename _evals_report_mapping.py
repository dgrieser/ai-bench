"""Helper module for managing evals.report-to-slug name mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

EVALS_REPORT_SCRIPT = Path(__file__).resolve().with_name("fetch_evals_report.py")
EVALS_REPORT_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-evals-report-to-artificialanalysis.json"
)

# Sentinel value stored for evals.report names reviewed but deliberately not
# mapped. Kept in the mapping file so they are not prompted again, but never
# used as a real llm.json model slug.
UNMAPPABLE = "__unmappable__"


def fetch_evals_report_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(EVALS_REPORT_SCRIPT), "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_evals_report.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_raw_mapping(path: Path = EVALS_REPORT_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_evals_report_to_slug_mapping(path: Path = EVALS_REPORT_MAPPING) -> dict[str, str]:
    """Real evals.report name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v != UNMAPPABLE}


def load_reviewed_evals_report_names(path: Path = EVALS_REPORT_MAPPING) -> set[str]:
    """All evals.report names already reviewed, including unmappable entries."""
    return set(_load_raw_mapping(path))


def write_evals_report_to_slug_mapping(
    mapping: dict[str, str], path: Path = EVALS_REPORT_MAPPING
) -> None:
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_evals_report_mapping(
    evals_report_name: str, slug: str, path: Path = EVALS_REPORT_MAPPING
) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(evals_report_name) == slug:
        return
    mapping[evals_report_name] = slug
    write_evals_report_to_slug_mapping(mapping, path)


def add_evals_report_unmappable(evals_report_name: str, path: Path = EVALS_REPORT_MAPPING) -> None:
    """Record an evals.report name as reviewed-but-unmapped so it is not prompted again."""
    add_evals_report_mapping(evals_report_name, UNMAPPABLE, path)
