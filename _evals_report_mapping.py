"""Helper module for managing evals.report-to-slug name mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from _openness import CLOSED_WEIGHTS, SENTINELS, UNMAPPABLE

EVALS_REPORT_SCRIPT = Path(__file__).resolve().with_name("fetch_evals_report.py")
EVALS_REPORT_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-evals-report-to-artificialanalysis.json"
)


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


def fetch_evals_report_model_openness() -> dict[str, bool | None]:
    """evals.report name -> has open weights, as read from the " Open" label.

    A name reported open by any row wins: the same model also appears in
    harness composite rows whose label omits the suffix.
    """
    proc = subprocess.run(
        [
            sys.executable,
            str(EVALS_REPORT_SCRIPT),
            "--format",
            "json",
            "--include-unverified",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_evals_report.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    entries: Any = json.loads(proc.stdout)
    flags: dict[str, set[bool]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("model"), str):
            continue
        seen = flags.setdefault(entry["model"], set())
        if entry.get("open_weights") is not None:
            seen.add(entry["open_weights"])
    return {
        name: True if True in seen else False if seen else None
        for name, seen in flags.items()
    }


def _load_raw_mapping(path: Path = EVALS_REPORT_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def load_evals_report_to_slug_mapping(path: Path = EVALS_REPORT_MAPPING) -> dict[str, str]:
    """Real evals.report name -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v not in SENTINELS}


def load_reviewed_evals_report_names(
    path: Path = EVALS_REPORT_MAPPING, include_closed: bool = True
) -> set[str]:
    """All evals.report names already reviewed, whether mapped or skipped.

    include_closed=False drops the names auto-recorded as closed-weight, so a
    source that mislabelled one can be reviewed again.
    """
    return {
        name
        for name, value in _load_raw_mapping(path).items()
        if include_closed or value != CLOSED_WEIGHTS
    }


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


def add_evals_report_closed_weights(evals_report_name: str, path: Path = EVALS_REPORT_MAPPING) -> None:
    """Record a evals.report name as skipped because the source reports closed weights."""
    add_evals_report_mapping(evals_report_name, CLOSED_WEIGHTS, path)
