#!/usr/bin/env python3
"""Tests for llm.html's GPU filter presets. Run with ./test_gpu_presets.py

Two strips lead the GPU filter: the mittwald hosting tiers, and the device
presets below them. Both are shortcuts into the same filter, and both name
cards out of gpu.json rather than carrying their own totals -- so a card
renamed or dropped there leaves a chip pointing at nothing. The page handles
that quietly (a preset whose card is missing is simply not offered), which is
right for a browser and wrong for a repository: nobody would notice the chip
had gone. These tests notice.

The device presets that carry a flat `vram` figure instead of a card -- the
unified-memory machines gpu.json has no entry for -- are checked for shape
only, since there is nothing to pin them to.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LLM_HTML = ROOT / "llm.html"
GPU_JSON = ROOT / "gpu.json"


def page_source() -> str:
    return LLM_HTML.read_text(encoding="utf-8")


def hosting_card() -> tuple[str, str]:
    """(vendor, model) the tier chips point at."""
    source = page_source()
    vendor = re.search(r'const HOSTING_PRESET_VENDOR = "([^"]+)";', source).group(1)
    model = re.search(r'const HOSTING_PRESET_GPU = "([^"]+)";', source).group(1)
    return vendor, model


def device_presets() -> list[dict[str, str | int]]:
    """The DEVICE_PRESETS entries, read one object at a time out of llm.html.

    One entry at a time, rather than one sweep of the whole literal, is what
    keeps a preset without a `gpu:` of its own from being paired with the next
    one's card.
    """
    source = page_source()
    block = re.search(r"const DEVICE_PRESETS = \[(.*?)\n      \];", source, re.S).group(1)

    presets: list[dict[str, str | int]] = []
    for entry in re.finditer(r'key:\s*"([A-Za-z0-9_-]+)"(.*?)(?=key:\s*"|\Z)', block, re.S):
        key, body = entry.group(1), entry.group(2)
        preset: dict[str, str | int] = {"key": key}
        for field in ("name", "vendor", "gpu", "note", "tip"):
            found = re.search(rf'{field}:\s*"([^"]*)"', body)
            if found:
                preset[field] = found.group(1)
        for field in ("cards", "vram"):
            found = re.search(rf"{field}:\s*(\d+)", body)
            if found:
                preset[field] = int(found.group(1))
        presets.append(preset)
    return presets


class TestGpuPresets(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalogue = {
            (gpu["manufacturer"], gpu["model"]): gpu
            for gpu in json.loads(GPU_JSON.read_text(encoding="utf-8"))
        }
        cls.presets = device_presets()

    def test_hosting_card_is_in_the_catalogue(self) -> None:
        card = hosting_card()
        self.assertIn(card, self.catalogue, f"{card} is not in gpu.json — the tier strip would stay hidden")

    def test_the_strip_is_not_empty(self) -> None:
        self.assertTrue(self.presets, "no device presets parsed out of llm.html")

    def test_every_card_backed_preset_is_in_the_catalogue(self) -> None:
        for preset in self.presets:
            if "gpu" not in preset:
                continue
            with self.subTest(preset=preset["key"]):
                card = (preset["vendor"], preset["gpu"])
                self.assertIn(
                    card, self.catalogue,
                    f"{card} is not in gpu.json — that chip would silently drop off the strip",
                )

    def test_every_preset_sets_a_budget(self) -> None:
        """A card and a count, or a flat VRAM figure. Never both, never neither:
        applying a preset has to leave the filter holding something."""
        for preset in self.presets:
            with self.subTest(preset=preset["key"]):
                self.assertNotEqual(
                    "gpu" in preset, "vram" in preset,
                    "a preset names a card or carries a vram figure, not both and not neither",
                )
                if "gpu" in preset:
                    self.assertIn("vendor", preset, "a card-backed preset needs the card's maker")
                    self.assertGreaterEqual(int(preset.get("cards", 1)), 1)
                else:
                    self.assertGreater(int(preset["vram"]), 0)
                    self.assertIn("note", preset, "a card-less preset says in its note what it stands for")

    def test_presets_are_labelled_and_explained(self) -> None:
        for preset in self.presets:
            with self.subTest(preset=preset["key"]):
                self.assertTrue(preset.get("name"), "every chip carries a name")
                self.assertTrue(preset.get("tip"), "every chip carries a tooltip")

    def test_keys_and_names_are_unique(self) -> None:
        keys = [preset["key"] for preset in self.presets]
        names = [preset.get("name") for preset in self.presets]
        self.assertEqual(len(keys), len(set(keys)), "a chip is applied by its key, so keys cannot repeat")
        self.assertEqual(len(names), len(set(names)))

    def test_the_strip_reads_high_to_low(self) -> None:
        """The chips are a ladder, and a ladder out of order is a list."""
        budgets = []
        for preset in self.presets:
            if "gpu" in preset:
                card = self.catalogue.get((preset["vendor"], preset["gpu"]))
                if not card:
                    continue
                budgets.append(card["vram_gb"] * int(preset.get("cards", 1)))
            else:
                budgets.append(int(preset["vram"]))
        self.assertEqual(budgets, sorted(budgets, reverse=True), f"out of order: {budgets}")

    def test_no_device_preset_repeats_the_hosting_card(self) -> None:
        """Two chips that arm on the same filter state would light up together."""
        vendor, model = hosting_card()
        for preset in self.presets:
            with self.subTest(preset=preset["key"]):
                self.assertNotEqual((preset.get("vendor"), preset.get("gpu")), (vendor, model))

    def test_no_two_presets_arm_on_the_same_state(self) -> None:
        seen: set[tuple[str, ...]] = set()
        for preset in self.presets:
            arm = (
                ("card", str(preset.get("vendor")), str(preset.get("gpu")), str(preset.get("cards", 1)))
                if "gpu" in preset
                else ("vram", str(preset["vram"]))
            )
            with self.subTest(preset=preset["key"]):
                self.assertNotIn(arm, seen, "two chips would arm on the same filter state")
            seen.add(arm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
