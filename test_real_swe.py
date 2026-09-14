#!/usr/bin/env python3
"""Tests for the Real-SWE leaderboard reader. Run with ./test_real_swe.py

Real-SWE publishes no API: the board is markup on a one-page Next.js site that
renders the same eight rows twice (an <ol> for phones, a <table> for everything
else) and carries two further tables further down -- a per-task pass matrix and
an effort table -- neither of which is a list of models with one score each.

So what these tests pin is everything that could silently read the wrong
numbers: that the table comes from the leaderboard container rather than from a
position on the page, that each row is keyed by the table's own header rather
than by column index, that the phone list never doubles a model, that a tie the
board prints as "=5" stays a tie, and that the confidence whisker -- which
exists only as CSS -- can never stand in for, or cost, the score beside it.
"""

from __future__ import annotations

import unittest
from unittest import mock

import fetch_real_swe as rs


def bar(score: float, ci: tuple[float, float] | None = (10.0, 20.0)) -> str:
    """The score cell's innards: the fill bar, optionally with a whisker."""
    whisker = ""
    if ci is not None:
        low, high = ci
        whisker = (
            '<div data-confidence-whisker="true" class="absolute" '
            f'style="left:{low}%;width:{round(high - low, 4)}%"></div>'
        )
    return (
        f'<div class="h-full" style="width:{score}%;background-color:#d97757"></div>'
        f"{whisker}"
        f'<span class="w-14 tabular-nums">{score}%</span>'
    )


def row(rank: str, model: str, harness: str, score: float, **kwargs) -> str:
    return (
        "<tr>"
        f'<td class="tabular-nums">{rank}</td>'
        f'<th scope="row"><span><img alt="" src="/logos/anthropic.svg"/>{model}</span></th>'
        f"<td>{harness}</td>"
        f"<td>{bar(score, **kwargs)}</td>"
        "</tr>"
    )


HEAD = (
    "<thead><tr>"
    '<th scope="col">#</th><th scope="col">Model</th>'
    '<th scope="col">Harness</th><th scope="col">Resolution rate</th>'
    "</tr></thead>"
)

# The phone rendering of the same models, inside the same figure. Reading it as
# well would double every row.
PHONE_LIST = (
    "<ol><li><div>Alpha 2"
    '<span class="sr-only">Resolution rate: </span>38.8%</div></li></ol>'
)

# A page in the shape the site serves: the board first, then an unrelated table
# whose rows are tasks rather than models.
PAGE = (
    "<div><p>prose</p></div>"
    '<div id="leaderboard"><figure aria-label="Model leaderboard">'
    f"{PHONE_LIST}"
    f"<table>{HEAD}<tbody>"
    + row("1", "Alpha 2", "Claude Code", 38.8, ci=(32.1, 45.4))
    + row("2", "Beta 1.5", "Codex CLI", 33.8)
    + row("=3", "Gamma", "Gemini CLI", 23.8, ci=None)
    + row("=3", "Delta", "Delta Code", 23.8)
    + "</tbody></table></figure></div>"
    '<figure aria-label="Task results"><table><thead><tr>'
    '<th scope="col">Task</th><th scope="col">Alpha 2</th></tr></thead>'
    "<tbody><tr><td>Tax jurisdiction</td><td>3.1%</td></tr></tbody></table></figure>"
)


def scores(page: str = PAGE) -> list[dict]:
    with mock.patch.object(rs, "fetch_html", return_value=page):
        return rs.get_scores()


def by_model(page: str = PAGE) -> dict[str, dict]:
    return {entry["model"]: entry for entry in scores(page)}


class TestTableSelection(unittest.TestCase):
    def test_reads_the_board_and_not_the_task_table(self) -> None:
        self.assertEqual(
            [e["model"] for e in scores()], ["Alpha 2", "Beta 1.5", "Gamma", "Delta"]
        )

    def test_the_phone_list_does_not_double_a_model(self) -> None:
        models = [e["model"] for e in scores()]
        self.assertEqual(len(models), len(set(models)))

    def test_the_label_is_enough_when_the_id_is_gone(self) -> None:
        page = PAGE.replace('id="leaderboard"', 'id="board"')
        self.assertEqual(len(scores(page)), 4)

    def test_a_missing_container_raises_rather_than_guessing(self) -> None:
        page = PAGE.replace('id="leaderboard"', 'id="board"').replace(
            'aria-label="Model leaderboard"', 'aria-label="Models"'
        )
        with self.assertRaises(ValueError):
            scores(page)

    def test_a_renamed_score_column_raises(self) -> None:
        """A renamed column could mean a new metric, so it is never read blind."""
        page = PAGE.replace("Resolution rate", "Weighted score")
        with self.assertRaises(ValueError):
            scores(page)


class TestRows(unittest.TestCase):
    def test_the_printed_percentage_is_the_score(self) -> None:
        self.assertEqual(by_model()["Alpha 2"]["score"], 38.8)

    def test_the_vendor_logo_is_not_part_of_the_model_name(self) -> None:
        self.assertEqual(by_model()["Beta 1.5"]["model"], "Beta 1.5")

    def test_every_row_carries_the_harness_it_was_run_under(self) -> None:
        """A row is a model-and-harness pair, which is how the board is run."""
        self.assertEqual(
            {e["model"]: e["harness"] for e in scores()},
            {
                "Alpha 2": "Claude Code",
                "Beta 1.5": "Codex CLI",
                "Gamma": "Gemini CLI",
                "Delta": "Delta Code",
            },
        )

    def test_cells_are_keyed_by_header_not_by_position(self) -> None:
        """A column inserted before the score must not shift what is read."""
        page = PAGE.replace(
            '<th scope="col">Harness</th>',
            '<th scope="col">Vendor</th><th scope="col">Harness</th>',
        ).replace('<td>Claude Code</td>', "<td>Anthropic</td><td>Claude Code</td>")
        entry = by_model(page)["Alpha 2"]
        self.assertEqual((entry["score"], entry["harness"]), (38.8, "Claude Code"))

    def test_a_published_tie_stays_a_tie(self) -> None:
        gamma, delta = by_model()["Gamma"], by_model()["Delta"]
        self.assertEqual((gamma["rank"], gamma["tied"]), (3, True))
        self.assertEqual((delta["rank"], delta["tied"]), (3, True))

    def test_a_clear_position_is_not_marked_tied(self) -> None:
        self.assertEqual((by_model()["Alpha 2"]["rank"], by_model()["Alpha 2"]["tied"]), (1, False))


class TestConfidenceInterval(unittest.TestCase):
    def test_the_whisker_reads_as_the_interval(self) -> None:
        entry = by_model()["Alpha 2"]
        self.assertEqual((entry["ci_low"], entry["ci_high"]), (32.1, 45.4))

    def test_the_interval_brackets_the_score(self) -> None:
        entry = by_model()["Alpha 2"]
        self.assertLessEqual(entry["ci_low"], entry["score"])
        self.assertGreaterEqual(entry["ci_high"], entry["score"])

    def test_a_missing_whisker_costs_nothing_but_the_interval(self) -> None:
        entry = by_model()["Gamma"]
        self.assertEqual(entry["score"], 23.8)
        self.assertEqual((entry["ci_low"], entry["ci_high"]), (None, None))

    def test_the_interval_is_never_read_as_the_score(self) -> None:
        """The bar's own width is the score's scale; the whisker is only drawn
        against it. Swapping the two would report a low end as a result."""
        page = PAGE.replace("left:32.1%", "left:0.0%")
        self.assertEqual(by_model(page)["Alpha 2"]["score"], 38.8)


class TestOutput(unittest.TestCase):
    def test_names_are_unique_and_sorted(self) -> None:
        with mock.patch.object(rs, "fetch_html", return_value=PAGE):
            names = sorted({e["model"] for e in rs.get_scores()})
        self.assertEqual(names, ["Alpha 2", "Beta 1.5", "Delta", "Gamma"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
