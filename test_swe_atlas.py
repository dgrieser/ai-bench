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
                mock.patch.object(atlas.time, "sleep"), \
                contextlib.redirect_stderr(io.StringIO()):
            return atlas.get_scores(list(atlas.LEGACY_TRACKS))

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
            self.scores({t: gone for t in atlas.LEGACY_TRACKS})

    def test_another_board_is_still_refused(self):
        with self.assertRaises(ValueError):
            self.scores({"tw": ValueError("the page does not identify itself as board")})


class SecondPass(unittest.TestCase):
    """The 404s and empty pages pass in minutes (2026-10-01/02, 10-05), so a
    failed track is read once more before it is skipped."""

    def run_with(self, outcomes: dict[str, list]) -> tuple[list[dict], mock.Mock]:
        def fetch_board_rows(_fetch, url, slug):
            track = slug.removeprefix("sweatlas-")
            queue = outcomes.get(track)
            result = queue.pop(0) if queue else None
            if isinstance(result, Exception):
                raise result
            return [_row(f"Opus 5 ({track})", 50.0)]

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {_fetch_warnings.ENV_VAR: str(Path(tmp) / "w.jsonl")}), \
                mock.patch.object(atlas, "fetch_board_rows", fetch_board_rows), \
                mock.patch.object(atlas.time, "sleep") as slept, \
                contextlib.redirect_stderr(io.StringIO()):
            return atlas.get_scores(list(atlas.LEGACY_TRACKS)), slept

    def test_a_track_back_on_the_second_pass_lands(self):
        gone = urllib.error.HTTPError("u", 404, "Not Found", None, None)
        rows, slept = self.run_with({"tw": [gone]})
        self.assertEqual(sorted({r["track"] for r in rows}), ["qna", "refactoring", "tw"])
        slept.assert_called_once_with(atlas.SECOND_PASS_DELAY)

    def test_an_empty_page_gets_the_second_pass_too(self):
        rows, _ = self.run_with({"refactoring": [atlas.NoRowsError("no rows")]})
        self.assertEqual(len(rows), 3)

    def test_no_failure_no_wait(self):
        _, slept = self.run_with({})
        slept.assert_not_called()

    def test_still_down_is_skipped(self):
        gone = urllib.error.HTTPError("u", 404, "Not Found", None, None)
        rows, _ = self.run_with({"tw": [gone, gone]})
        self.assertEqual(sorted({r["track"] for r in rows}), ["qna", "refactoring"])

    def test_another_board_is_not_given_a_second_pass(self):
        with self.assertRaises(ValueError):
            self.run_with({"tw": [ValueError("the page does not identify itself as board")]})


def _flight(*chunks: str) -> str:
    return "".join(
        f"<script>self.__next_f.push([1,{json.dumps(chunk)}])</script>" for chunk in chunks
    )


def _current_page(table: str | None = "sweatlas_1.1 Overall", variants=None) -> str:
    """The v1.1 page as Scale serves it: its slug, the site's board list naming
    the table behind each board, and one variant table per track."""
    variants = variants or {v: [_row(f"Opus 5 ({v}) max", 50.0)] for v in atlas.CURRENT_VARIANTS}
    boards = [{"name": "SWE Atlas - Codebase QnA", "id": "sweatlas-qna",
               "airtableTableName": "sweatlas-qna", "deprecated": True}]
    if table is not None:
        boards.append({"name": "SWE Atlas", "id": "swe-atlas",
                       "airtableTableName": table, "deprecated": False})
    items = [{"key": k, "label": k, "entries": rows} for k, rows in variants.items()]
    return _flight(
        '0:{"query":{"slug":"swe-atlas"}}\n',
        "2:" + json.dumps({"boards": boards}, separators=(",", ":")) + "\n",
        '1c:["$","div",null,{"variants":' + json.dumps(items) + "}]\n",
    )


class Revisions(unittest.TestCase):
    """v1.1 (2026-11-06) re-ran every model with a new judge on one page; the
    v1.0 track boards stay up as Legacy. Each revision has its own columns."""

    def current_rows(self, html: str) -> tuple[list[dict], Path]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        warnings = Path(tmp.name) / "w.jsonl"
        with mock.patch.dict(os.environ, {_fetch_warnings.ENV_VAR: str(warnings)}), \
                mock.patch.object(atlas, "fetch_html", return_value=html), \
                contextlib.redirect_stderr(io.StringIO()):
            return atlas.get_scores([atlas.CURRENT_BOARD]), warnings

    def test_legacy_boards_are_v1_0(self):
        with mock.patch.object(atlas, "fetch_board_rows", return_value=[_row("Opus 5", 50.0)]), \
                contextlib.redirect_stderr(io.StringIO()):
            rows = atlas.get_scores(list(atlas.LEGACY_TRACKS))
        self.assertEqual(
            sorted(r["key"] for r in rows),
            ["swe_atlas_qna_1_0", "swe_atlas_rf_1_0", "swe_atlas_tw_1_0"],
        )

    def test_the_current_page_names_its_revision(self):
        self.assertEqual(atlas.current_revision(_current_page()), "1.1")
        self.assertIsNone(atlas.current_revision(_current_page(table=None)))

    def test_current_page_rows_land_in_the_1_1_columns(self):
        rows, _ = self.current_rows(_current_page())
        self.assertEqual(
            sorted(r["key"] for r in rows),
            ["swe_atlas_1_1", "swe_atlas_qna_1_1", "swe_atlas_rf_1_1", "swe_atlas_tw_1_1"],
        )
        self.assertEqual({r["model"] for r in rows}, {"opus 5"})
        self.assertEqual({r["url"] for r in rows}, {atlas.CURRENT_URL})

    def test_an_unknown_or_missing_revision_is_not_filed(self):
        for table in ("sweatlas_1.2 Overall", None):
            with self.subTest(table=table):
                rows, warnings = self.current_rows(_current_page(table=table))
                self.assertTrue(rows)
                self.assertEqual({r["key"] for r in rows}, {None})
                [entry] = _fetch_warnings.load(warnings)
                self.assertEqual(entry["source"], "swe_atlas (current)")

    def test_a_missing_track_refuses_the_page(self):
        variants = {"overall": [_row("Opus 5", 50.0)], "qna": [_row("Opus 5", 50.0)]}
        with mock.patch.object(atlas, "fetch_html", return_value=_current_page(variants=variants)):
            with self.assertRaises(ValueError):
                atlas.fetch_current()

    def test_every_column_has_a_page_and_a_benchmark(self):
        benchmarks = json.loads(
            Path(__file__).with_name("llm.json").read_text(encoding="utf-8")
        )["benchmarks"]
        self.assertEqual(
            set(atlas.COLUMN_URLS),
            {"swe_atlas_1_1", "swe_atlas_qna_1_1", "swe_atlas_rf_1_1", "swe_atlas_tw_1_1",
             "swe_atlas_qna_1_0", "swe_atlas_rf_1_0", "swe_atlas_tw_1_0"},
        )
        for key in atlas.COLUMN_URLS:
            self.assertIn(key, benchmarks)
        for old in ("swe_atlas_qna", "swe_atlas_rf", "swe_atlas_tw"):
            self.assertNotIn(old, benchmarks)

    def test_a_max_in_the_name_survives_a_trailing_effort(self):
        # v1.1 writes the name with a space and the effort after the harness.
        self.assertEqual(atlas.normalize_model("Qwen 3.8 Max (mini-swe-agent) max"), "qwen 3.8 max")
        self.assertEqual(atlas.normalize_model("Claude Fable 5.2 (Claude Code) max"), "claude fable 5.2")
        self.assertEqual(atlas.normalize_model("Gemini 4 Carbon (Antigravity CLI) high"), "gemini 4 carbon")


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
