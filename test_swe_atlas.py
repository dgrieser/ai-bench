#!/usr/bin/env python3
"""Tests for the SWE Atlas label normalizer. Run with ./test_swe_atlas.py

Scale writes a run's reasoning effort as a word of its own after the name --
"Opus 5 (Claude Code) xHigh", "Gpt 5.4 xHigh (Mini-SWE-Agent)" -- and the
normalizer drops it so one model gets one mapping key whatever effort it ran
at. "Max" is also part of some model names, though, and there it is joined by
a hyphen: "Qwen3.8-Max". Stripping every \\bmax\\b folded such a model onto its
base ("qwen3.8"), so the -Max row's score was credited to the smaller model
(issue #233). The rule under test: only a free-standing word is an effort.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import fetch_swe_atlas as atlas

MAPPING = Path(__file__).with_name("model-name-mapping-swe-atlas-to-artificialanalysis.json")


class Efforts(unittest.TestCase):
    def test_a_free_standing_effort_is_dropped(self):
        for raw, expected in [
            ("Opus 5 (Claude Code) xHigh", "opus 5"),
            ("Gpt 5.4 xHigh (Mini-SWE-Agent)", "gpt 5.4"),
            ("Opus 4.8 (Claude Code) xhigh", "opus 4.8"),
            ("Opus 5 (Claude Code) Max", "opus 5"),
            ("GPT-5.5 x-high", "gpt 5.5"),
            ("Fable-5.1 (Claude Code) xHigh*", "fable 5.1 *"),
        ]:
            self.assertEqual(atlas.normalize_model(raw), expected, raw)

    def test_a_hyphenated_max_is_part_of_the_name(self):
        self.assertEqual(atlas.normalize_model("Qwen3.8-Max (Mini-SWE-Agent)"), "qwen3.8 max")
        self.assertEqual(atlas.normalize_model("Kimi-K3-Max xHigh"), "kimi k3 max")
        self.assertNotEqual(
            atlas.normalize_model("Qwen3.8-Max (Mini-SWE-Agent)"),
            atlas.normalize_model("Qwen3.8 (Mini-SWE-Agent)"),
        )


class KeepsTheMapping(unittest.TestCase):
    """Labels from the live board (2026-09-24) still land on the keys the
    mapping file holds, so the fix unmaps nothing."""

    LABELS = [
        "Opus 5 (Claude Code) xHigh\n",
        "Fable-5.1 (Claude Code) xHigh*",
        "GPT 6 Astra (Codex) xHigh*",
        "GLM 5.2 (Mini-SWE-Agent)",
        "GPT-5.6-Sol (Codex) xHigh*",
        "DeepSeek V4 Pro (Mini-SWE-Agent)",
        "Kimi K2.5 (Mini-SWE-Agent)",
        "Minimax-M2.5 (Mini-SWE)",
        "Muse Spark 1.1 (Mini-SWE-Agent) xHigh",
        "Gemini-3.1-Pro (Gemini CLI)",
    ]

    def test_live_labels_hit_existing_keys(self):
        keys = set(json.loads(MAPPING.read_text()))
        for raw in self.LABELS:
            self.assertIn(atlas.normalize_model(raw), keys, raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
