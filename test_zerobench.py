#!/usr/bin/env python3
"""Tests for the ZeroBench leaderboard reader. Run with ./test_zerobench.py

zerobench.github.io puts three tables on one page and two of them are traps.
``sortableTableProvider`` is externally reported numbers -- model cards and
release posts, many of them "w/ tools", which on a benchmark whose best row is
22% roughly doubles the score -- and ``sortableTableRelease`` is the retired
single-greedy-sampling cohort. Both open with a "Model | pass@1 | pass@5"
header, so nothing about the shape of a table says which one it is.

What these tests pin is therefore everything that could silently read the wrong
number: that the table is chosen by id and never by position, that a renamed id
or a changed header stops the read instead of falling through to a neighbour,
that the effort parenthetical is dropped from the mapping key while a *model*
parenthetical is kept, and that the ● / ○ sampling marks never leak into the
score.
"""

from __future__ import annotations

import unittest
from unittest import mock

import fetch_zerobench as zb


def row(model: str, pass_1: str, rest: str = "26.0") -> str:
    return (
        f"<tr><td>{model}</td><td>{pass_1}</td><td>{rest}</td>"
        "<td>11.0</td><td>$0.68</td><td>30.0k</td></tr>"
    )


HEADER = (
    "<tr><th>Main questions (100)</th><th>Mean per question</th></tr>"
    "<tr><th>Model</th><th>pass@1</th><th>pass@5</th><th>pass^5</th>"
    "<th>cost</th><th>tokens</th></tr>"
)

OFFICIAL = (
    f'<table id="{zb.OFFICIAL_TABLE_ID}">'
    + HEADER
    + row("Claude Opus 5 (max)", "17.2●")
    + row("GPT-5.6 Sol (max)", "22.0●")
    + row("o1", "0.0○")
    + "</table>"
)

# Same header shape, different numbers -- the "w/ tools" rows this reader must
# never reach.
PROVIDER = (
    '<table id="sortableTableProvider">'
    "<tr><th>Model</th><th>pass@1</th><th>pass@5</th><th>pass@1</th>"
    "<th>Source</th><th>Notes</th></tr>"
    + row("GPT-5.6 Sol (max, w/ tools)", "54.6")
    + "</table>"
)

RELEASE = (
    '<table id="sortableTableRelease">'
    "<tr><th>Model</th><th>pass@1</th><th>pass@5</th><th>pass^5</th>"
    "<th>pass@1</th><th>n correct</th><th>cost</th><th>tokens</th></tr>"
    + row("o1 pro", "0.0")
    + "</table>"
)

PAGE = f"<html><body>{PROVIDER}{OFFICIAL}{RELEASE}</body></html>"


def scores(page: str) -> list[dict]:
    with mock.patch.object(zb, "fetch_html", return_value=page):
        return zb.get_scores()


class SelectsTheOfficialTable(unittest.TestCase):
    def test_reads_the_official_board_not_its_neighbours(self):
        by_model = {entry["model"]: entry["score"] for entry in scores(PAGE)}
        self.assertEqual(by_model, {"claude opus 5": 17.2, "gpt 5.6 sol": 22.0, "o1": 0.0})

    def test_the_with_tools_number_never_lands(self):
        # 54.6 is the provider table's Sol row; 22.0 is the board's own.
        self.assertEqual(
            [e["score"] for e in scores(PAGE) if e["model"] == "gpt 5.6 sol"], [22.0]
        )

    def test_position_on_the_page_is_not_what_selects_it(self):
        # The official table is second here and first in reordered; same result.
        reordered = f"<html><body>{OFFICIAL}{PROVIDER}{RELEASE}</body></html>"
        self.assertEqual(
            [e["model"] for e in scores(reordered)],
            [e["model"] for e in scores(PAGE)],
        )

    def test_a_renamed_id_stops_the_read(self):
        renamed = PAGE.replace(f'id="{zb.OFFICIAL_TABLE_ID}"', 'id="sortableTableV2"')
        with self.assertRaises(ValueError):
            scores(renamed)

    def test_a_changed_header_stops_the_read(self):
        # pass@1 renamed: the column's metric is no longer known, so refuse it
        # rather than read whatever now sits in that position.
        changed = OFFICIAL.replace("<th>pass@1</th>", "<th>accuracy</th>", 1)
        with self.assertRaises(ValueError):
            scores(f"<html><body>{changed}</body></html>")

    def test_an_empty_board_is_refused(self):
        empty = f'<table id="{zb.OFFICIAL_TABLE_ID}">{HEADER}</table>'
        with self.assertRaises(ValueError):
            scores(f"<html><body>{empty}</body></html>")


class NormalizesLabels(unittest.TestCase):
    def test_effort_parentheticals_are_dropped(self):
        for raw, expected in [
            ("Claude Opus 5 (max)", "claude opus 5"),
            ("GPT-5.2 (medium reasoning)", "gpt 5.2"),
            ("Grok 4.20 (non-reasoning)", "grok 4.20"),
            ("Claude Sonnet 4.5 (thinking)", "claude sonnet 4.5"),
            ("Gemini 3.5 Flash-Lite (high)", "gemini 3.5 flash lite"),
            ("Claude Opus 4.8", "claude opus 4.8"),
        ]:
            self.assertEqual(zb.normalize_model(raw), expected, raw)

    def test_a_model_parenthetical_is_kept(self):
        # Sol, Luna and Terra are three models, not three efforts, and llm.json
        # tracks each under its own slug.
        self.assertEqual(zb.normalize_model("gpt-5.6 (sol)"), "gpt 5.6 sol")
        self.assertEqual(zb.normalize_model("gpt-5.6 (luna)"), "gpt 5.6 luna")

    def test_the_raw_label_survives_normalization(self):
        entry = next(e for e in scores(PAGE) if e["model"] == "claude opus 5")
        self.assertEqual(entry["raw"], "Claude Opus 5 (max)")
        self.assertEqual(entry["effort"], "max")


class ReadsTheMetric(unittest.TestCase):
    def test_sampling_marks_are_reported_not_scored(self):
        by_model = {e["model"]: e for e in scores(PAGE)}
        self.assertEqual(by_model["claude opus 5"]["sampling"], "mean")
        self.assertEqual(by_model["o1"]["sampling"], "greedy")
        # The mark is not digits, so it can never reach the score.
        self.assertEqual(by_model["claude opus 5"]["score"], 17.2)

    def test_rows_without_a_score_are_skipped(self):
        board = (
            f'<table id="{zb.OFFICIAL_TABLE_ID}">'
            + HEADER
            + row("Scored", "5.0●")
            + row("Unscored", "-")
            + "</table>"
        )
        self.assertEqual([e["model"] for e in scores(f"<body>{board}</body>")], ["scored"])

    def test_rank_follows_the_score(self):
        self.assertEqual(
            [(e["model"], e["rank"]) for e in scores(PAGE)],
            [("gpt 5.6 sol", 1), ("claude opus 5", 2), ("o1", 3)],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
