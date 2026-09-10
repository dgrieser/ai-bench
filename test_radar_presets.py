#!/usr/bin/env python3
"""Tests for llm.html's radar axis presets. Run with ./test_radar_presets.py

The radar's preset dropdown charts either the five composite indexes or the
benchmarks one index aggregates. Those contributor lists live in
derive_indexes.py's INDEXES -- that is where they are computed from -- but the
page cannot read them: llm.json publishes each index's label, description and
icon, never its contributors, so RADAR_PRESETS in llm.html restates them.

A restated list drifts. These tests pin it to the source: same benchmarks, same
order, and every key actually a column in llm.json, so adding or dropping a
contributor in derive_indexes.py fails here until the page is updated too.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

import derive_indexes as di

ROOT = Path(__file__).resolve().parent
LLM_HTML = ROOT / "llm.html"
LLM_JSON = ROOT / "llm.json"


def parse_presets() -> dict[str, list[str]]:
    """id -> axis keys, read out of llm.html's RADAR_PRESETS literal.

    The "default" preset names RADAR_DEFAULT_AXES instead of spelling an array,
    so that constant is resolved separately and the rest are read one object at
    a time (an object at a time, rather than one sweep, is what keeps a preset
    without its own array from being paired with the next one's).
    """
    source = LLM_HTML.read_text(encoding="utf-8")

    def keys_of(literal: str) -> list[str]:
        return re.findall(r'"([A-Za-z0-9_]+)"', literal)

    defaults = keys_of(
        re.search(r"const RADAR_DEFAULT_AXES = (\[[^\]]*\]);", source).group(1)
    )
    block = re.search(r"const RADAR_PRESETS = \[(.*?)\n      \];", source, re.S).group(1)

    presets: dict[str, list[str]] = {}
    for entry in re.finditer(r'id:\s*"([A-Za-z0-9_]+)"(.*?)(?=id:\s*"|\Z)', block, re.S):
        pid, body = entry.group(1), entry.group(2)
        axes = re.search(r"axes:\s*(\[[^\]]*\]|RADAR_DEFAULT_AXES)", body, re.S)
        assert axes, f"preset {pid} declares no axes"
        literal = axes.group(1)
        presets[pid] = list(defaults) if literal == "RADAR_DEFAULT_AXES" else keys_of(literal)
    return presets


class TestRadarPresets(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.presets = parse_presets()
        cls.benchmarks = json.loads(LLM_JSON.read_text(encoding="utf-8"))["benchmarks"]

    def test_every_index_has_a_preset(self) -> None:
        for index in di.INDEXES:
            pid = index.key.removesuffix("_index")
            self.assertIn(pid, self.presets, f"no radar preset for {index.key}")

    def test_default_preset_is_the_indexes(self) -> None:
        # As a set: the chart lays its axes out clockwise in this order, which
        # is a reading decision (Trust before Vision) and not derive_indexes.py's
        # computation order.
        self.assertEqual(
            sorted(self.presets["default"]), sorted(index.key for index in di.INDEXES)
        )

    def test_preset_matches_its_index(self) -> None:
        for index in di.INDEXES:
            pid = index.key.removesuffix("_index")
            with self.subTest(preset=pid):
                self.assertEqual(
                    self.presets[pid],
                    [key for key, _ in index.contributing],
                    "llm.html's RADAR_PRESETS no longer matches derive_indexes.py",
                )

    def test_every_axis_is_a_column(self) -> None:
        for pid, axes in self.presets.items():
            with self.subTest(preset=pid):
                missing = [key for key in axes if key not in self.benchmarks]
                self.assertEqual(missing, [], f"not columns in llm.json: {missing}")

    def test_presets_fit_the_charts_bounds(self) -> None:
        source = LLM_HTML.read_text(encoding="utf-8")
        low = int(re.search(r"const RADAR_MIN_AXES = (\d+);", source).group(1))
        high = int(re.search(r"const RADAR_MAX_AXES = (\d+);", source).group(1))
        for pid, axes in self.presets.items():
            with self.subTest(preset=pid):
                self.assertGreaterEqual(len(axes), low)
                self.assertLessEqual(
                    len(axes), high,
                    "raise RADAR_MAX_AXES or the picker cannot reproduce this preset",
                )

    def test_no_preset_repeats_an_axis(self) -> None:
        for pid, axes in self.presets.items():
            with self.subTest(preset=pid):
                self.assertEqual(len(axes), len(set(axes)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
