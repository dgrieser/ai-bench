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

import contextlib
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import _fetch_warnings

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


def _row(model: str, score: float) -> dict:
    return {"model": model, "score": score}


class OneTrackDown(unittest.TestCase):
    """sweatlas-tw 404'd for hours at a time (runs 2026-09-27/28) while qna and
    refactoring served; that cost all three columns and two warnings a run."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.warnings = Path(tmp.name) / "w.jsonl"
        env = mock.patch.dict(os.environ, {_fetch_warnings.ENV_VAR: str(self.warnings)})
        env.start()
        self.addCleanup(env.stop)

    def scores(self, down: dict[str, Exception]) -> list[dict]:
        def fetch_board_rows(_fetch, url, slug):
            track = slug.removeprefix("sweatlas-")
            if track in down:
                raise down[track]
            return [_row(f"Opus 5 ({track})", 50.0)]

        with mock.patch.object(atlas, "fetch_board_rows", fetch_board_rows), \
                contextlib.redirect_stderr(io.StringIO()):
            return atlas.get_scores(list(atlas.TRACKS))

    def test_the_other_tracks_still_land(self):
        gone = urllib.error.HTTPError("u", 404, "Not Found", None, None)
        rows = self.scores({"tw": gone})
        self.assertEqual(sorted({r["track"] for r in rows}), ["qna", "refactoring"])
        [entry] = _fetch_warnings.load(self.warnings)
        self.assertEqual(entry["source"], "swe_atlas (tw)")
        self.assertIn("sweatlas-tw", entry["message"])
        self.assertIn("404", entry["message"])

    def test_an_empty_board_is_skipped_too(self):
        rows = self.scores({"tw": atlas.NoRowsError("no rows")})
        self.assertEqual(len(rows), 2)

    def test_every_track_down_still_fails_the_source(self):
        gone = urllib.error.URLError("down")
        with self.assertRaises(urllib.error.URLError):
            self.scores({t: gone for t in atlas.TRACKS})

    def test_another_board_is_still_refused(self):
        with self.assertRaises(ValueError):
            self.scores({"tw": ValueError("the page does not identify itself as board")})


class Retries(unittest.TestCase):
    def test_a_404_is_not_retried(self):
        gone = urllib.error.HTTPError("u", 404, "Not Found", None, None)
        with mock.patch.object(atlas.urllib.request, "urlopen", side_effect=gone) as opened, \
                mock.patch.object(atlas.time, "sleep") as slept:
            with self.assertRaises(urllib.error.HTTPError):
                atlas.fetch_html("https://example.invalid")
        self.assertEqual(opened.call_count, 1)
        slept.assert_not_called()

    def test_a_server_error_is(self):
        busy = urllib.error.HTTPError("u", 503, "Unavailable", None, None)
        with mock.patch.object(atlas.urllib.request, "urlopen", side_effect=busy) as opened, \
                mock.patch.object(atlas.time, "sleep"), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(urllib.error.HTTPError):
                atlas.fetch_html("https://example.invalid")
        self.assertEqual(opened.call_count, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
