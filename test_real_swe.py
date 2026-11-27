#!/usr/bin/env python3
"""Tests for the Real-SWE leaderboard reader. Run with ./test_real_swe.py

Real-SWE publishes no API: the board is markup on a one-page Next.js site that
renders the same rows twice (an <ol> for phones, a <table> for everything
else) and carries two further tables further down -- a per-task pass matrix and
an effort table -- neither of which is a list of models with one score each.

So what these tests pin is everything that could silently read the wrong
numbers: that the table comes from the leaderboard container rather than from a
position on the page, that each row is keyed by the table's own header rather
than by column index, that the phone list never doubles a model, that a tie the
board prints as "=5" stays a tie, that the screen-reader-only copy of a name
("Opus 5.5" + "Claude Opus 5.5") or rank ("=8" + "Tied for 8") is never read
into the visible one, and that the one-standard-error bar -- stated as
screen-reader text and drawn as CSS -- can never stand in for, or cost, the
score beside it.
"""

from __future__ import annotations

import unittest
from unittest import mock

import fetch_real_swe as rs


def sr(text: str) -> str:
    return f'<span class="sr-only">{text}</span>'


def bar(
    score: float,
    se: tuple[float, float] | None = (10.0, 20.0),
    stated: bool = True,
) -> str:
    """The score cell's innards, in the shape the site serves since November
    2026: the fill bar, optionally a drawn `data-error-bar`, and the printed
    percentage followed by the error bar as screen-reader text."""
    drawn = said = ""
    if se is not None:
        low, high = se
        drawn = (
            '<div data-error-bar="true" class="absolute" '
            f'style="left:{low}%;width:{round(high - low, 4)}%"></div>'
        )
        if stated:
            said = sr(f", <!-- -->one standard error either side: {low}% to {high}%")
    return (
        '<div aria-hidden="true">'
        f'<div class="h-full" style="width:{score}%;background-color:#d97757"></div>'
        f"{drawn}</div>"
        f'<div><span class="percentage">{score:.2f}%{said}</span></div>'
    )


def rank_cell(rank: str) -> str:
    if rank.startswith("="):
        return f'<span aria-hidden="true">{rank}</span>' + sr(f"Tied for {rank[1:]}")
    return rank


def row(rank: str, model: str, score: float, full_name: str | None = None, **kwargs) -> str:
    name = model
    if full_name is not None:
        name = (
            f'<span title="{full_name}"><span aria-hidden="true">{model}</span>'
            f"{sr(full_name)}</span>"
        )
    return (
        "<tr>"
        f'<td class="tabular-nums">{rank_cell(rank)}</td>'
        f'<th scope="row"><span><img alt="" src="/logos/anthropic.svg"/>'
        f'<a href="/models/x">{name}</a></span></th>'
        f"<td>{bar(score, **kwargs)}</td>"
        "</tr>"
    )


HEAD = (
    "<thead><tr>"
    f'<th scope="col"><span aria-hidden="true">#</span>{sr("Rank")}</th>'
    '<th scope="col">Model</th><th scope="col">Resolution rate</th>'
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
    "<html><head><title>Real-SWE 1.1 — by Specific</title></head>"
    "<div><p>prose</p></div>"
    '<div id="leaderboard"><figure aria-label="Model leaderboard">'
    f"{PHONE_LIST}"
    f"<table>{HEAD}<tbody>"
    + row("1", "Alpha 2", 38.8, full_name="Claude Alpha 2", se=(32.1, 45.4))
    + row("2", "Beta 1.5", 33.8)
    + row("=3", "Gamma", 23.8, se=None)
    + row("=3", "Delta", 23.8, se=(20.1, 27.5), stated=False)
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

    def test_the_screen_reader_full_name_is_not_appended(self) -> None:
        """The cell prints "Alpha 2" and hides "Claude Alpha 2" for screen
        readers; reading both gave "Alpha 2 Claude Alpha 2"."""
        self.assertIn("Alpha 2", by_model())
        self.assertNotIn("Claude", " ".join(by_model()))

    def test_no_harness_column_means_no_harness(self) -> None:
        """The table stopped naming the harness; rows say None, not a guess."""
        self.assertEqual({e["harness"] for e in scores()}, {None})

    def test_a_harness_column_is_read_if_it_returns(self) -> None:
        page = PAGE.replace(
            '<th scope="col">Model</th>',
            '<th scope="col">Model</th><th scope="col">Harness</th>',
        ).replace("</a></span></th>", "</a></span></th><td>Claude Code</td>")
        self.assertEqual({e["harness"] for e in scores(page)}, {"Claude Code"})

    def test_cells_are_keyed_by_header_not_by_position(self) -> None:
        """A column inserted before the score must not shift what is read."""
        page = PAGE.replace(
            '<th scope="col">Model</th>',
            '<th scope="col">Model</th><th scope="col">Vendor</th>',
        ).replace("</a></span></th>", "</a></span></th><td>Anthropic 99%</td>")
        self.assertEqual(by_model(page)["Alpha 2"]["score"], 38.8)

    def test_a_published_tie_stays_a_tie(self) -> None:
        gamma, delta = by_model()["Gamma"], by_model()["Delta"]
        self.assertEqual((gamma["rank"], gamma["tied"]), (3, True))
        self.assertEqual((delta["rank"], delta["tied"]), (3, True))

    def test_a_clear_position_is_not_marked_tied(self) -> None:
        self.assertEqual((by_model()["Alpha 2"]["rank"], by_model()["Alpha 2"]["tied"]), (1, False))


class TestStandardError(unittest.TestCase):
    def test_the_stated_error_bar_is_read(self) -> None:
        entry = by_model()["Alpha 2"]
        self.assertEqual((entry["se_low"], entry["se_high"]), (32.1, 45.4))

    def test_the_drawn_bar_is_the_fallback(self) -> None:
        entry = by_model()["Delta"]
        self.assertEqual((entry["se_low"], entry["se_high"]), (20.1, 27.5))

    def test_the_stated_text_wins_over_the_drawing(self) -> None:
        page = PAGE.replace("left:32.1%", "left:30.0%")
        entry = by_model(page)["Alpha 2"]
        self.assertEqual((entry["se_low"], entry["se_high"]), (32.1, 45.4))

    def test_the_interval_brackets_the_score(self) -> None:
        entry = by_model()["Alpha 2"]
        self.assertLessEqual(entry["se_low"], entry["score"])
        self.assertGreaterEqual(entry["se_high"], entry["score"])

    def test_a_missing_error_bar_costs_nothing_but_the_interval(self) -> None:
        entry = by_model()["Gamma"]
        self.assertEqual(entry["score"], 23.8)
        self.assertEqual((entry["se_low"], entry["se_high"]), (None, None))

    def test_the_interval_is_never_read_as_the_score(self) -> None:
        """The error bar's ends are percentages too; the score is the first
        percentage the board prints, never one of them."""
        page = PAGE.replace("either side: 32.1%", "either side: 0.0%")
        self.assertEqual(by_model(page)["Alpha 2"]["score"], 38.8)

    def test_the_old_confidence_whisker_is_not_read_as_a_standard_error(self) -> None:
        page = PAGE.replace("data-error-bar", "data-confidence-whisker")
        self.assertEqual(by_model(page)["Delta"]["se_low"], None)


class TestRevision(unittest.TestCase):
    def test_the_title_names_the_release(self) -> None:
        self.assertEqual(rs.page_revision(PAGE), "1.1")
        self.assertEqual(
            rs.page_revision("<title>Real-SWE 2 Benchmark</title>"), "2.0"
        )

    def test_a_page_naming_no_release_is_refused(self) -> None:
        # 1.1 replaced 1.0 at the same address once already; a page that stops
        # saying which release it is must not be filed under the last one.
        with self.assertRaisesRegex(ValueError, "names no Real-SWE release"):
            rs.page_revision("<title>Specific Labs</title>")

    def test_every_row_names_the_release(self) -> None:
        with mock.patch.object(rs, "fetch_html", return_value=PAGE):
            rows = rs.get_scores()
        self.assertEqual({r["revision"] for r in rows}, {"1.1"})


class TestOutput(unittest.TestCase):
    def test_names_are_unique_and_sorted(self) -> None:
        with mock.patch.object(rs, "fetch_html", return_value=PAGE):
            names = sorted({e["model"] for e in rs.get_scores()})
        self.assertEqual(names, ["Alpha 2", "Beta 1.5", "Delta", "Gamma"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
