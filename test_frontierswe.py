#!/usr/bin/env python3
"""Tests for the FrontierSWE leaderboard reader. Run with ./test_frontierswe.py

V2 keys the flight payload by score mode ("abs") and then by trial aggregation
("mean", "best", "worst"), and scores each model as a percentage in `overall`.
V1's pairwise `dominance` -- a win rate on an entirely different scale, still
served at /v1 -- must never be what reaches llm.json, so these tests pin the
shape the reader accepts and the number it takes out of it.
"""

from __future__ import annotations

import unittest
from unittest import mock

import fetch_frontierswe as fs


def row(model: str, overall: float, harness: str = "proximus") -> dict:
    return {
        "model": model,
        "harness": harness,
        "vendor": "vendor",
        "overall": overall,
        "implementation": overall + 1,
        "performance": overall - 1,
        "research": overall,
        "avgCostUsd": 12.5,
    }


ENTRIES = {
    "abs": {
        "mean": [row("Alpha", 56.29), row("Beta", 32.204), row("Gamma", 4.117)],
        "best": [row("Alpha", 66.65), row("Beta", 44.17), row("Gamma", 10.79)],
        "worst": [row("Alpha", 44.47), row("Beta", 22.2), row("Gamma", 0.13)],
    }
}


def scores(**kwargs) -> list[dict]:
    with mock.patch.object(fs, "fetch_html", return_value=""), mock.patch.object(
        fs, "extract_entries", return_value=ENTRIES
    ):
        return fs.get_scores(**kwargs)


class TestSelectGroup(unittest.TestCase):
    def test_reads_through_the_score_mode(self):
        self.assertEqual(fs.select_group(ENTRIES, "mean"), ENTRIES["abs"]["mean"])

    def test_missing_score_mode_names_what_is_there(self):
        with self.assertRaises(ValueError) as ctx:
            fs.select_group({"rel": {"mean": []}}, "mean")
        self.assertIn("'abs'", str(ctx.exception))
        self.assertIn("rel", str(ctx.exception))

    def test_missing_group_names_the_views(self):
        with self.assertRaises(ValueError) as ctx:
            fs.select_group(ENTRIES, "median")
        self.assertIn("'median'", str(ctx.exception))
        self.assertIn("worst", str(ctx.exception))

    def test_v1_payload_is_not_read_as_v2(self):
        # /v1 still serves the un-nested, dominance-scored shape; taking its
        # numbers for V2's would silently mix two scales in one column.
        v1 = {"best": [{"model": "Alpha", "dominance": 0.90625, "overall": 2.5}]}
        with self.assertRaises(ValueError):
            fs.select_group(v1, "best")


class TestGetScores(unittest.TestCase):
    def test_mean_is_the_default_view(self):
        self.assertEqual([r["score"] for r in scores()], [56.3, 32.2, 4.1])

    def test_group_selects_the_view(self):
        self.assertEqual([r["score"] for r in scores(group="best")], [66.7, 44.2, 10.8])
        self.assertEqual([r["score"] for r in scores(group="worst")], [44.5, 22.2, 0.1])

    def test_keeps_the_unrounded_percentage_alongside(self):
        self.assertEqual(scores()[1]["overall"], 32.204)

    def test_ranks_by_descending_score(self):
        self.assertEqual([r["rank"] for r in scores()], [1, 2, 3])
        self.assertEqual([r["model"] for r in scores()], ["Alpha", "Beta", "Gamma"])

    def test_carries_the_harness(self):
        self.assertEqual(scores()[0]["harness"], "proximus")

    def test_drops_rows_without_a_usable_score(self):
        entries = {"abs": {"mean": [row("Alpha", 10.0), {"model": "NoScore"}, "junk"]}}
        with mock.patch.object(fs, "fetch_html", return_value=""), mock.patch.object(
            fs, "extract_entries", return_value=entries
        ):
            self.assertEqual([r["model"] for r in fs.get_scores()], ["Alpha"])


class TestExtractEntries(unittest.TestCase):
    def test_pulls_entries_out_of_the_flight_chunks(self):
        import json

        payload = '17:["$","$L24",null,{"entries":{"abs":{"mean":[{"model":"Alpha"}]}},"tail":1}]'
        halves = [payload[:20], payload[20:]]
        html = "".join(
            f"<script>self.__next_f.push([1,{json.dumps(half)}])</script>" for half in halves
        )
        self.assertEqual(fs.extract_entries(html), {"abs": {"mean": [{"model": "Alpha"}]}})

    def test_missing_entries_raises(self):
        with self.assertRaises(ValueError):
            fs.extract_entries("<html></html>")


if __name__ == "__main__":
    unittest.main()
