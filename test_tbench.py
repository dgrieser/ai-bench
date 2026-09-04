#!/usr/bin/env python3
"""Tests for the Terminal-Bench 4.0 leaderboard reader. Run with ./test_tbench.py

tbench.ai serves only the *current* board: the version picker is client-side,
so ``?version=2.1`` returns the same payload the homepage does, and a new
release simply takes the slot 4.0 occupies today. Reading that as 4.0 would
look like every model moving at once, so the load-bearing test here is
test_a_later_release_is_not_read_as_4_0 -- the reader has to refuse a payload
whose leaderboard is not the one it was written for.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

import fetch_tbench as tb


def row(model: str, accuracy: float, agent: str = "Claude Code", effort: str = "max") -> dict:
    return {
        "id": f"row-{model}-{effort}",
        "rank": 1,
        "metadata": {
            "date": "2026-09-01",
            "agent_org": {"url": "https://example.test", "label": "Example"},
            "model_org": {"url": "https://example.test", "label": "Example"},
            "display_date": "Sep 1, 2026",
            "agent_display": {"url": "https://example.test/agent", "label": agent},
            "model_display": {"url": "https://example.test/model", "label": model},
            "reasoning_effort": effort,
        },
        "metrics": {
            "accuracy": accuracy,
            "n_trials": 330,
            "display_accuracy": f"**{accuracy}%** ± 3.0%",
            "accuracy_ci95_half_width": 3.0,
        },
        "status": "display",
    }


BOARD = {
    "leaderboard": {
        "id": "board-id",
        "name": tb.LEADERBOARD_NAME,
        "package": tb.PACKAGE,
        "title": "Terminal-Bench 4.0",
    },
    "rows": [
        row("Alpha", 57.88),
        row("Alpha", 44.55, effort="high"),
        row("Beta", 41.82, agent="Codex"),
    ],
}


def payload(board: dict) -> str:
    """A flight payload with the board embedded the way tbench.ai embeds it."""
    return (
        '3a:["$","$L3c",null,{"state":{"queries":[{"state":{"data":'
        + json.dumps(board)
        + ',"dataUpdateCount":1}}]}}]\n'
    )


def scores(board: dict = BOARD) -> list[dict]:
    with mock.patch.object(tb, "fetch_payload", return_value=payload(board)):
        return tb.get_scores()


class TestExtractBoard(unittest.TestCase):
    def test_reads_the_board_out_of_the_flight_payload(self):
        board = tb.extract_board(payload(BOARD))
        self.assertEqual(board["leaderboard"]["name"], tb.LEADERBOARD_NAME)
        self.assertEqual(len(board["rows"]), 3)

    def test_reads_the_board_out_of_an_html_shell(self):
        # A server that answers the RSC request with HTML embeds the same
        # payload, escaped inside self.__next_f.push chunks.
        chunk = json.dumps(payload(BOARD))
        html = f"<script>self.__next_f.push([1,{chunk}])</script>"
        self.assertEqual(len(tb.extract_board(html)["rows"]), 3)

    def test_a_payload_without_the_board_is_an_error(self):
        with self.assertRaises(ValueError) as ctx:
            tb.extract_board('1:["$","div",null,{}]')
        self.assertIn("leaderboard payload", str(ctx.exception))


class TestVersionGuard(unittest.TestCase):
    def test_accepts_the_board_it_was_written_for(self):
        self.assertEqual(tb.check_version(BOARD)["name"], tb.LEADERBOARD_NAME)

    def test_a_later_release_is_not_read_as_4_0(self):
        later = {**BOARD, "leaderboard": {**BOARD["leaderboard"], "name": "5-0-0"}}
        with self.assertRaises(ValueError) as ctx:
            tb.check_version(later)
        self.assertIn("5-0-0", str(ctx.exception))
        self.assertIn(tb.LEADERBOARD_NAME, str(ctx.exception))

    def test_another_dataset_is_not_read_as_terminal_bench(self):
        other = {**BOARD, "leaderboard": {**BOARD["leaderboard"], "package": "other/bench"}}
        with self.assertRaises(ValueError):
            tb.check_version(other)

    def test_a_payload_without_a_descriptor_is_an_error(self):
        with self.assertRaises(ValueError):
            tb.check_version({"rows": []})

    def test_get_scores_refuses_a_later_release(self):
        later = {**BOARD, "leaderboard": {**BOARD["leaderboard"], "name": "5-0-0"}}
        with self.assertRaises(ValueError):
            scores(later)


class TestGetScores(unittest.TestCase):
    def test_reads_accuracy_as_the_score(self):
        self.assertEqual(scores()[0]["score"], 57.88)

    def test_every_agent_and_effort_row_is_kept(self):
        # update.py folds them onto one slug; the reader must not pre-empt that.
        alpha = [e for e in scores() if e["model"] == "Alpha"]
        self.assertEqual(sorted(e["effort"] for e in alpha), ["high", "max"])

    def test_the_model_label_is_the_mapping_key(self):
        self.assertEqual({e["model"] for e in scores()}, {"Alpha", "Beta"})

    def test_agent_and_effort_are_kept_beside_the_score(self):
        beta = next(e for e in scores() if e["model"] == "Beta")
        self.assertEqual(beta["agent"], "Codex")
        self.assertEqual(beta["effort"], "max")

    def test_rows_are_ranked_best_first(self):
        got = scores()
        self.assertEqual([e["rank"] for e in got], [1, 2, 3])
        self.assertEqual([e["score"] for e in got], sorted((e["score"] for e in got), reverse=True))

    def test_a_row_without_an_accuracy_is_dropped(self):
        broken = row("Gamma", 0.0)
        broken["metrics"].pop("accuracy")
        board = {**BOARD, "rows": [*BOARD["rows"], broken]}
        self.assertNotIn("Gamma", {e["model"] for e in scores(board)})

    def test_a_boolean_is_not_an_accuracy(self):
        broken = row("Gamma", 0.0)
        broken["metrics"]["accuracy"] = True
        board = {**BOARD, "rows": [*BOARD["rows"], broken]}
        self.assertNotIn("Gamma", {e["model"] for e in scores(board)})

    def test_a_row_without_a_model_label_is_dropped(self):
        broken = row("Gamma", 12.0)
        broken["metadata"]["model_display"] = {"url": "https://example.test"}
        board = {**BOARD, "rows": [*BOARD["rows"], broken]}
        self.assertEqual(len(scores(board)), len(scores()))


if __name__ == "__main__":
    unittest.main()
