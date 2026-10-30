"""Hand-entered VRAM figures that hold while Spheron keeps reporting a known-wrong one.

Spheron is the only writer of ``vram`` and it overwrites on every refresh, so
a figure typed into llm.json by hand would be gone by the next run. Most of the
time that is right: the recommender is the source and a hand figure would go
stale. Sometimes the recommender is plainly wrong -- it sizes
``mistralai/Mistral-Large-4-1T-A52B`` as a 52B model, reading the active count
in the repo name as the total, and asks 113 GB at FP16 for a 1.05T checkpoint.

``spheron-vram-overrides.json`` records, per slug, the reading being ruled
out::

    {"mistral-large-4": {"spheron": {"fp16": 113, "int8": 57, "int4": 28},
                         "reason": "..."}}

``update.update_spheron_vram`` leaves a quant alone while Spheron still
reports exactly that value, so the hand figure and its ``vram_source`` stand.
The moment Spheron reports anything else -- its fix, or a re-estimate -- the
pin no longer matches and Spheron writes as usual; the refresh says so, and the
entry can then be deleted. Pinning the *wrong value* rather than the slug is
what makes the override expire by itself instead of outliving the bug.

``edit.py --vram-fp16 … --vram-url …`` writes the hand figure and records the
Spheron reading it replaces here, so the two are never entered separately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OVERRIDES_PATH = Path(__file__).resolve().with_name("spheron-vram-overrides.json")
QUANTS = ("fp16", "int8", "int4")


def is_spheron_url(url: Any) -> bool:
    return isinstance(url, str) and "spheron.network/" in url


def load(path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = path or OVERRIDES_PATH
    if not path.exists():
        return {}
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, dict)}


def save(overrides: dict[str, dict[str, Any]], path: Path | None = None) -> None:
    path = path or OVERRIDES_PATH
    ordered = {slug: overrides[slug] for slug in sorted(overrides)}
    path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ignored_value(overrides: dict[str, dict[str, Any]], slug: str, quant: str) -> Any:
    """The Spheron reading ruled out for this slug and quant, or None."""
    entry = overrides.get(slug)
    if not isinstance(entry, dict):
        return None
    spheron = entry.get("spheron")
    if not isinstance(spheron, dict):
        return None
    return spheron.get(quant)


def record(
    overrides: dict[str, dict[str, Any]],
    slug: str,
    model: dict[str, Any],
    quants: list[str],
    reason: str | None = None,
) -> list[str]:
    """Rule out the Spheron readings a hand edit of ``quants`` is replacing.

    Only a stored value credited to a Spheron page is recorded: anything else
    is not a reading the refresh would bring back. Returns the quants recorded.
    """
    vram = model.get("vram") if isinstance(model.get("vram"), dict) else {}
    sources = model.get("vram_source") if isinstance(model.get("vram_source"), dict) else {}
    recorded: list[str] = []
    for quant in quants:
        value = vram.get(quant)
        if value is None or not is_spheron_url(sources.get(quant)):
            continue
        entry = overrides.setdefault(slug, {})
        entry.setdefault("spheron", {})[quant] = value
        recorded.append(quant)
    if recorded and reason:
        overrides[slug]["reason"] = reason
    return recorded
