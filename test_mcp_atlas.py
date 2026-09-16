#!/usr/bin/env python3
"""Tests for the MCP-Atlas label normalizer. Run with ./test_mcp_atlas.py

Scale puts two different things in the same parenthetical. "(xHigh)" and
"(max)" are how a model was run, and folding those into the mapping key would
give one model a key per effort. "(sol)" is *which* model was run: GPT-5.6 Sol,
GPT-5.6 Luna and GPT-5.6 Terra are three models, each with its own slug in
llm.json. Dropping that collapsed all three onto "gpt 5.6", a key that cannot
be mapped to any one of them and so was answered __unmappable__ -- which is why
the board's Sol row reached llm.json only by hand.

So the rule under test is: a parenthetical is dropped when every token in it is
a run setting, and kept otherwise. The second test is the one that keeps the
first honest -- every key the live board produces today must still normalize to
the key the mapping file already holds, or the fix silently unmaps the field.
"""

from __future__ import annotations

import unittest

import fetch_mcp_atlas as mcp


class DropsRunSettings(unittest.TestCase):
    def test_effort_parentheticals_are_dropped(self):
        for raw, expected in [
            ("claude-opus-5 (xhigh)", "claude opus 5"),
            ("Inkling (xHigh)", "inkling"),
            ("kimi-k3 (max)", "kimi k3"),
            ("gemini-3.5-flash (high)", "gemini 3.5 flash"),
            ("claude-sonnet-4-5 (thinking)", "claude sonnet 4 5"),
            ("GPT-5.2 (medium reasoning)", "gpt 5.2"),
        ]:
            self.assertEqual(mcp.normalize_model(raw), expected, raw)

    def test_labels_without_a_parenthetical_are_unchanged(self):
        self.assertEqual(mcp.normalize_model("Muse Spark 1.1"), "muse spark 1.1")
        self.assertEqual(mcp.normalize_model("claude-fable-5-1"), "claude fable 5 1")

    def test_scales_p_decimal_spelling_is_restored(self):
        self.assertEqual(mcp.normalize_model("glm-5p2"), "glm 5.2")
        self.assertEqual(mcp.normalize_model("kimi-k2p5"), "kimi k2.5")


class KeepsModelNames(unittest.TestCase):
    def test_a_variant_parenthetical_stays_in_the_key(self):
        self.assertEqual(mcp.normalize_model("gpt-5.6 (sol)"), "gpt 5.6 sol")
        self.assertEqual(mcp.normalize_model("gpt-5.6 (luna)"), "gpt 5.6 luna")
        self.assertEqual(mcp.normalize_model("gpt-5.6 (terra)"), "gpt 5.6 terra")

    def test_the_three_variants_do_not_share_a_key(self):
        keys = {mcp.normalize_model(f"gpt-5.6 ({v})") for v in ("sol", "luna", "terra")}
        self.assertEqual(len(keys), 3)

    def test_a_mixed_parenthetical_keeps_only_the_model_token(self):
        # One unrecognised token is enough to keep the parenthetical; the
        # effort beside it is then stripped like any other, and the comma that
        # separated them goes with it rather than trailing the key.
        self.assertEqual(mcp.normalize_model("gpt-5.6 (sol, max)"), "gpt 5.6 sol")
        self.assertEqual(mcp.normalize_model("gpt-5.6 (sol/high)"), "gpt 5.6 sol")

    def test_is_run_setting_rejects_the_empty_parenthetical(self):
        # "model ()" says nothing; treat it as a name we do not recognise
        # rather than silently dropping it.
        self.assertFalse(mcp.is_run_setting(""))
        self.assertFalse(mcp.is_run_setting(None))
        self.assertTrue(mcp.is_run_setting("xHigh"))


class KeepsTheMappingFileValid(unittest.TestCase):
    """Every label the board carries today, and the key it must produce.

    Taken from a live fetch. If a change here moves any of these off the key the
    mapping file already holds, the field silently loses its mapping.
    """

    LIVE = {
        "Muse Spark 1.1": "muse spark 1.1",
        "claude-fable-5-1": "claude fable 5 1",
        "claude-opus-5 (xhigh)": "claude opus 5",
        "gemini-3.5-flash (high)": "gemini 3.5 flash",
        "Claude Fable 5": "claude fable 5",
        "kimi-k3 (max)": "kimi k3",
        "claude-opus-4-8 (max)": "claude opus 4 8",
        "Muse Spark": "muse spark",
        "Inkling-small": "inkling small",
        "claude-opus-4-7 (max)": "claude opus 4 7",
        "gemini-3.1-pro-preview (high)": "gemini 3.1 pro preview",
        "glm-5p2": "glm 5.2",
        "claude-opus-4-6 (max)": "claude opus 4 6",
        "Inkling (xHigh)": "inkling",
        "glm-5p1": "glm 5.1",
        "gpt-5.5 (xhigh)": "gpt 5.5",
        "gpt-5.4 (xhigh)": "gpt 5.4",
        "gemini-3-pro-preview": "gemini 3 pro preview",
        "claude-opus-4-5 (high)": "claude opus 4 5",
        "claude-sonnet-4-6": "claude sonnet 4 6",
        "gpt-5.2 (xhigh)": "gpt 5.2",
        "kimi-k2p5": "kimi k2.5",
        "gemini-3-flash-preview": "gemini 3 flash preview",
        "claude-sonnet-4-5 (thinking)": "claude sonnet 4 5",
        "glm-4p7": "glm 4.7",
        "gemini-3.1-flash-lite (high)": "gemini 3.1 flash lite",
        "gpt-5.4-mini (xhigh)": "gpt 5.4 mini",
        "gpt-5.1 (high)": "gpt 5.1",
        "o3-pro": "o3 pro",
        "claude-haiku-4-5": "claude haiku 4 5",
    }

    def test_every_live_label_keeps_its_key(self):
        for raw, expected in self.LIVE.items():
            self.assertEqual(mcp.normalize_model(raw), expected, raw)

    def test_the_only_key_the_fix_moves_is_the_gpt_5_6_one(self):
        # "gpt-5.6 (sol)" is deliberately absent from LIVE: it is the one label
        # whose key changes, from "gpt 5.6" to "gpt 5.6 sol".
        self.assertNotIn("gpt-5.6 (sol)", self.LIVE)
        self.assertEqual(mcp.normalize_model("gpt-5.6 (sol)"), "gpt 5.6 sol")


class SplitsTheLabel(unittest.TestCase):
    def test_harness_is_reported_separately(self):
        self.assertEqual(mcp.split_harness("gpt-5.6 (sol)"), ("gpt-5.6", "sol"))
        self.assertEqual(mcp.split_harness("Inkling (xHigh)"), ("Inkling", "xHigh"))
        self.assertEqual(mcp.split_harness("Muse Spark 1.1"), ("Muse Spark 1.1", None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
