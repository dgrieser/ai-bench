"""Helper module for managing llm-stats.com name mappings.

llm-stats is a general (multi-benchmark) source identified by model name, so it
needs two mappings: model_id -> llm.json model slug, and benchmark label ->
llm.json benchmark key. The two halves mirror _deepswe_mapping.py (model names)
and _huggingface_mapping.py (benchmark names).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

LLMSTATS_SCRIPT = Path(__file__).resolve().with_name("fetch_llmstats.py")
LLMSTATS_MODEL_MAPPING = Path(__file__).resolve().with_name(
    "model-name-mapping-llmstats-to-artificialanalysis.json"
)
LLMSTATS_BENCHMARK_MAPPING = Path(__file__).resolve().with_name(
    "llmstats-benchmark-name-mapping.json"
)

# Sentinel value stored for names/labels reviewed but deliberately not mapped.
# Kept in the mapping file so they are not prompted again, but never used as a
# real llm.json slug or benchmark key.
UNMAPPABLE = "__unmappable__"


def _load_raw_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def _write_mapping(mapping: dict[str, str], path: Path) -> None:
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _add_mapping(key: str, value: str, path: Path) -> None:
    mapping = _load_raw_mapping(path)
    if mapping.get(key) == value:
        return
    mapping[key] = value
    _write_mapping(mapping, path)


# --- Model-name side (llm-stats model_id -> llm.json slug) ---------------------


def fetch_llmstats_model_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(LLMSTATS_SCRIPT), "--format", "names", "--names", "models"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_llmstats.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def load_llmstats_to_slug_mapping(
    path: Path = LLMSTATS_MODEL_MAPPING,
) -> dict[str, str]:
    """Real llm-stats model_id -> llm.json model slug mappings."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v != UNMAPPABLE}


def load_reviewed_llmstats_names(path: Path = LLMSTATS_MODEL_MAPPING) -> set[str]:
    """All llm-stats model ids already reviewed, including unmappable entries."""
    return set(_load_raw_mapping(path))


def write_llmstats_to_slug_mapping(
    mapping: dict[str, str], path: Path = LLMSTATS_MODEL_MAPPING
) -> None:
    _write_mapping(mapping, path)


def add_llmstats_mapping(
    llmstats_name: str, slug: str, path: Path = LLMSTATS_MODEL_MAPPING
) -> None:
    _add_mapping(llmstats_name, slug, path)


def add_llmstats_unmappable(
    llmstats_name: str, path: Path = LLMSTATS_MODEL_MAPPING
) -> None:
    """Record an llm-stats model id as reviewed-but-unmapped so it is not prompted again."""
    _add_mapping(llmstats_name, UNMAPPABLE, path)


# --- Benchmark-name side (llm-stats label -> llm.json benchmark key) -----------


def fetch_llmstats_benchmark_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(LLMSTATS_SCRIPT), "--format", "names", "--names", "benchmarks"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_llmstats.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def load_llmstats_benchmark_to_key_mapping(
    path: Path = LLMSTATS_BENCHMARK_MAPPING,
) -> dict[str, str]:
    """Real llm-stats benchmark label -> llm.json key mappings (excludes unmappable)."""
    return {k: v for k, v in _load_raw_mapping(path).items() if v != UNMAPPABLE}


def load_reviewed_llmstats_benchmarks(
    path: Path = LLMSTATS_BENCHMARK_MAPPING,
) -> set[str]:
    """All llm-stats benchmark labels already reviewed, including unmappable entries."""
    return set(_load_raw_mapping(path))


def write_llmstats_benchmark_mapping(
    mapping: dict[str, str], path: Path = LLMSTATS_BENCHMARK_MAPPING
) -> None:
    _write_mapping(mapping, path)


def add_llmstats_benchmark_mapping(
    label: str, key: str, path: Path = LLMSTATS_BENCHMARK_MAPPING
) -> None:
    _add_mapping(label, key, path)


def add_llmstats_benchmark_unmappable(
    label: str, path: Path = LLMSTATS_BENCHMARK_MAPPING
) -> None:
    """Record an llm-stats benchmark label as reviewed-but-unmapped."""
    _add_mapping(label, UNMAPPABLE, path)
