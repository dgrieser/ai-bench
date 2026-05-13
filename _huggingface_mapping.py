"""Helper module for managing Hugging Face benchmark name mappings."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HF_SCRIPT = Path(__file__).resolve().with_name("fetch_huggingface.py")
HF_MAPPING = Path(__file__).resolve().with_name("huggingface-benchmark-name-mapping.json")


def fetch_huggingface_benchmark_names() -> list[str]:
    proc = subprocess.run(
        [sys.executable, str(HF_SCRIPT), "--all-models", "--format", "names"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fetch_huggingface.py failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def load_hf_to_key_mapping(path: Path = HF_MAPPING) -> dict[str, str]:
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def write_hf_to_key_mapping(mapping: dict[str, str], path: Path = HF_MAPPING) -> None:
    path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_hf_mapping(hf_label: str, key: str, path: Path = HF_MAPPING) -> None:
    mapping = load_hf_to_key_mapping(path)
    if mapping.get(hf_label) == key:
        return
    mapping[hf_label] = key
    write_hf_to_key_mapping(mapping, path)
