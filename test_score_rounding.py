#!/usr/bin/env python3
"""Tests for score rounding. Run with ./test_score_rounding.py

Every score is stored on the grid its benchmark declares: one printed digit by
default, or the coarser "round_to" step a benchmark whose numbers drift on
their own asks for. The point is that a refresh only rewrites llm.json when the
number a reader sees actually moved -- a leaderboard reporting 43.78 where 43.8
is stored must not restamp the score's date for a change the site cannot show.
These tests pin the grid rules and their use in apply_score.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import update
from _scores import round_score, score_decimals, score_step

URL = "https://example.com/leaderboard"

DOC = {
    "benchmarks": {
        # A plain percentage: no metadata, so one digit.
        "pct": {"name": "Percent"},
        # An Elo that re-anchors as the judging pool grows.
        "elo": {"name": "Elo", "decimals": 0, "round_to": 5},
        # A whole-point index, like the derived Coding column.
        "points": {"name": "Points", "decimals": 0},
    }
}


def model_with(score=None, key="pct") -> dict:
    return {
        "name": "m",
        "scores": {key: score},
        "scores_updated": {key: None},
        "scores_source": {key: None},
    }


class TestScoreGrid(unittest.TestCase):
    def test_unknown_benchmark_gets_the_printed_default(self) -> None:
        self.assertEqual(score_decimals(DOC, "nope"), 1)
        self.assertEqual(float(score_step(DOC, "nope")), 0.1)

    def test_decimals_alone_sets_the_step(self) -> None:
        self.assertEqual(float(score_step(DOC, "points")), 1.0)

    def test_round_to_overrides_the_step_from_decimals(self) -> None:
        self.assertEqual(float(score_step(DOC, "elo")), 5.0)

    def test_malformed_metadata_falls_back_instead_of_raising(self) -> None:
        doc = {"benchmarks": {"b": {"decimals": "two", "round_to": -1}}}
        self.assertEqual(float(score_step(doc, "b")), 0.1)


class TestRoundScore(unittest.TestCase):
    def test_none_passes_through(self) -> None:
        self.assertIsNone(round_score(DOC, "pct", None))

    def test_extra_digits_collapse_onto_the_printed_one(self) -> None:
        self.assertEqual(round_score(DOC, "pct", 43.78), 43.8)
        self.assertEqual(round_score(DOC, "pct", 48.77), 48.8)

    def test_an_integral_result_is_stored_as_an_int(self) -> None:
        value = round_score(DOC, "pct", 34.04)
        self.assertEqual(value, 34)
        self.assertIsInstance(value, int)

    def test_float_dust_from_scaling_does_not_survive(self) -> None:
        # 0.4378 * 100 is 43.78000000000001 in binary floating point.
        self.assertEqual(round_score(DOC, "pct", update.to_percent(0.4378)), 43.8)

    def test_halves_round_away_from_zero_both_ways(self) -> None:
        self.assertEqual(round_score(DOC, "pct", 43.75), 43.8)
        self.assertEqual(round_score(DOC, "pct", 43.85), 43.9)
        self.assertEqual(round_score(DOC, "pct", -43.75), -43.8)

    def test_a_coarse_step_snaps_to_its_multiples(self) -> None:
        self.assertEqual(round_score(DOC, "elo", 743.6), 745)
        self.assertEqual(round_score(DOC, "elo", 716.8), 715)
        self.assertEqual(round_score(DOC, "elo", -121.4), -120)

    def test_rounding_is_idempotent(self) -> None:
        for key, value in (("pct", 43.78), ("elo", 743.6), ("points", 50374.4)):
            once = round_score(DOC, key, value)
            self.assertEqual(round_score(DOC, key, once), once)

    def test_a_non_number_is_an_error_rather_than_a_silent_write(self) -> None:
        with self.assertRaises(TypeError):
            round_score(DOC, "pct", "43.8")


class TestApplyScoreRounds(unittest.TestCase):
    def test_a_finer_reading_of_the_stored_value_is_not_a_change(self) -> None:
        model = model_with(score=43.8)
        changes: list = []
        n = update.apply_score(DOC, model, "m", "pct", 43.78, URL, changes)
        self.assertEqual(n, 0)
        self.assertEqual(model["scores"]["pct"], 43.8)
        # The date is what a spurious rewrite would burn.
        self.assertIsNone(model["scores_updated"]["pct"])
        self.assertEqual(changes, [])

    def test_a_real_move_still_lands_rounded(self) -> None:
        model = model_with(score=43.8)
        n = update.apply_score(DOC, model, "m", "pct", 44.44, URL, [])
        self.assertEqual(n, 1)
        self.assertEqual(model["scores"]["pct"], 44.4)

    def test_elo_drift_below_the_step_is_not_a_change(self) -> None:
        model = model_with(score=745, key="elo")
        n = update.apply_score(DOC, model, "m", "elo", 743.6, URL, [])
        self.assertEqual(n, 0)
        self.assertEqual(model["scores"]["elo"], 745)

    def test_fill_source_urls_matches_a_stored_rounded_score(self) -> None:
        # The backfill attributes a stored score to the source whose current
        # value equals it; an unrounded reading must not miss its own write.
        model = model_with(score=43.8)
        n = update.apply_score(DOC, model, "m", "pct", 43.78, URL, [], fill_urls_only=True)
        self.assertEqual(n, 1)
        self.assertEqual(model["scores_source"]["pct"], URL)


class TestLlmJsonIsOnItsGrid(unittest.TestCase):
    def test_every_stored_score_is_already_rounded(self) -> None:
        doc = json.loads(
            (Path(__file__).resolve().with_name("llm.json")).read_text(encoding="utf-8")
        )
        off_grid = [
            (model["name"], key, value)
            for model in doc["models"]
            for key, value in (model.get("scores") or {}).items()
            if value is not None and round_score(doc, key, value) != value
        ]
        self.assertEqual(off_grid, [])


if __name__ == "__main__":
    unittest.main()
