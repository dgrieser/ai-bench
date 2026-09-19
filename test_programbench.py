#!/usr/bin/env python3
"""Tests for the ProgramBench leaderboard reader. Run with ./test_programbench.py

Three things carry the weight here.

``test_the_score_is_almost_not_resolved``: the board prints Almost Resolved
beside Resolved and the two are far apart -- 37.0 against 4.5 for the leader --
so reading the wrong cell would move every value in the file by the width of
that gap while still looking like a plausible benchmark score, the kind of error
nothing downstream would catch. The column stores Almost, and is named for it;
Resolved is reported on every row and never scored.

``test_a_zero_is_a_score_not_a_gap``: several rows are almost-resolving nothing,
and a reader that treated 0% as "no result" would drop them and leave the column
reporting only the models above the floor.

``test_a_moved_leaderboard_is_refused``: the table is picked by its class, so a
layout change raises instead of silently reading the first table on the page.
"""

from __future__ import annotations

import unittest
from unittest import mock

import fetch_programbench as pb


def row(
    rank: int,
    model: str,
    resolved: str,
    almost: str,
    agent: str = "mini-SWE-agent",
    cost: str = "$1.00",
    calls: str = "42",
) -> str:
    return f"""
    <tr class="clickable-row" data-href="/run/x/" data-provider="Example">
      <td class="col-rank rank-{rank}">{rank}</td>
      <td class="col-logo"><img src="/static/images/example.svg" class="org-logo"></td>
      <td class="col-model"><span class="model-name">{model}</span></td>
      <td class="col-agent muted">{agent}</td>
      <td class="col-num resolved-highlight">{resolved}</td>
      <td class="col-num col-almost">{almost}</td>
      <td class="col-num col-cost">{cost}</td>
      <td class="col-num col-calls">{calls}</td>
    </tr>"""


HEADER = """
    <thead><tr>
      <th class="col-rank">Rank</th>
      <th class="col-logo"></th>
      <th>Model</th>
      <th class="col-agent">Agent</th>
      <th class="col-num col-resolved-header">
        <span class="th-info"><span class="th-full">Resolved</span><span class="th-mobile">Res.</span>
          <span class="material-icons-outlined info-icon">help_outline</span>
          <span class="info-popover">The number of fully solved instances.</span>
        </span>
      </th>
      <th class="col-num col-almost-header">
        <span class="th-info">Almost
          <span class="material-icons-outlined info-icon">help_outline</span>
          <span class="info-popover">Instances where the solution solves &ge; 95% of tests.</span>
        </span>
      </th>
      <th class="col-num"><span class="th-info">Cost
        <span class="info-popover">Average API cost in USD per task instance.</span></span></th>
      <th class="col-num"><span class="th-info">Calls
        <span class="info-popover">Average number of LLM calls per task instance.</span></span></th>
    </tr></thead>"""


def page(
    rows: str,
    subtitle: str = "Evaluated with mini-SWE-agent &middot; 200 tasks",
    cls: str = "lb-table",
) -> str:
    return f"""<html><body>
      <p class="lb-sub">{subtitle}</p>
      <table class="{cls}">{HEADER}<tbody>{rows}</tbody></table>
    </body></html>"""


def stub(body: str):
    return mock.patch.object(pb, "fetch_html", return_value=body)


class TestParsing(unittest.TestCase):
    def test_the_score_is_almost_not_resolved(self) -> None:
        with stub(page(row(1, "Claude Opus 5 (xhigh)", "4.5%", "37.0%"))):
            entries = pb.get_scores()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["score"], 37.0)
        self.assertEqual(entries[0]["resolved"], 4.5)

    def test_a_row_is_kept_on_its_almost_cell_alone(self) -> None:
        # Resolved sitting at zero is the normal case on this board and must
        # never cost the row its Almost score.
        with stub(page(row(1, "A", "0%", "16.5%"))):
            entries = pb.get_scores()
        self.assertEqual([(e["score"], e["resolved"]) for e in entries], [(16.5, 0.0)])

    def test_a_zero_is_a_score_not_a_gap(self) -> None:
        with stub(page(row(1, "A", "0.5%", "5.0%") + row(2, "B", "0%", "0.0%"))):
            entries = pb.get_scores()
        self.assertEqual([e["model"] for e in entries], ["A", "B"])
        self.assertEqual(entries[1]["score"], 0.0)

    def test_a_moved_leaderboard_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            pb.parse_rows(
                pb.select_leaderboard_table(page(row(1, "A", "1%", "2%"), cls="other-table"))
            )

    def test_a_renamed_score_column_is_refused(self) -> None:
        # Silently losing the Almost column would empty the ingest; the header
        # is checked so that shows up as a failure instead. Resolved is checked
        # too: it is reported on every row, and a board that stopped printing it
        # is a board whose meaning has changed.
        for label, missing in (("Resolved", "resolved"), ("Almost", "almost")):
            with self.subTest(column=label):
                body = page(row(1, "A", "1%", "2%")).replace(f">{label}", ">Renamed", 1)
                with self.assertRaises(ValueError) as caught:
                    pb.parse_rows(pb.select_leaderboard_table(body))
                self.assertIn(missing, str(caught.exception))

    def test_the_mobile_label_is_not_read_as_the_column_name(self) -> None:
        rows = pb.parse_rows(pb.select_leaderboard_table(page(row(1, "A", "1%", "2%"))))
        self.assertIn("resolved", rows[0])
        self.assertIn("almost", rows[0])

    def test_the_logo_column_does_not_displace_the_rank(self) -> None:
        with stub(page(row(3, "A", "1%", "2%"))):
            entries = pb.get_scores()
        self.assertEqual(entries[0]["rank"], 3)

    def test_the_effort_suffix_is_kept_and_reported(self) -> None:
        # Two efforts of one model are two runs; the ingest collapses them onto
        # a slug with "best reported run wins", which it can only do while the
        # published labels stay distinct.
        with stub(
            page(row(1, "GPT 5.5 (xhigh)", "0.5%", "13.5%") + row(2, "GPT 5.5", "0%", "1.5%"))
        ):
            entries = pb.get_scores()
        self.assertEqual([e["model"] for e in entries], ["GPT 5.5 (xhigh)", "GPT 5.5"])
        self.assertEqual([e["effort"] for e in entries], ["xhigh", None])

    def test_cost_and_calls_are_read_when_present(self) -> None:
        with stub(page(row(1, "A", "4.5%", "37.0%", cost="$51.04", calls="259"))):
            entry = pb.get_scores()[0]
        self.assertEqual(entry["cost_per_task"], 51.04)
        self.assertEqual(entry["calls_per_task"], 259)

    def test_the_task_count_is_read_from_the_subtitle(self) -> None:
        with stub(page(row(1, "A", "4.5%", "37.0%"))):
            self.assertEqual(pb.get_scores()[0]["tasks"], 200)

    def test_a_missing_task_count_is_not_fatal(self) -> None:
        with stub(page(row(1, "A", "4.5%", "37.0%"), subtitle="Evaluated with mini-SWE-agent")):
            self.assertIsNone(pb.get_scores()[0]["tasks"])

    def test_the_root_pages_percent_markup_is_read_too(self) -> None:
        # The site root wraps the sign ("37.0<span class="pct">%</span>") where
        # /extended/ prints it inline; one reader has to handle both.
        body = page(
            row(1, "A", '4.5<span class="pct">%</span>', '37.0<span class="pct">%</span>')
        )
        with stub(body):
            self.assertEqual(pb.get_scores()[0]["score"], 37.0)


class TestSourceIdentity(unittest.TestCase):
    def test_the_leaderboard_url_is_the_complete_table(self) -> None:
        # The root shows the top ten of the same runs, so reading it as well
        # would double every row it carries.
        self.assertTrue(pb.LEADERBOARD_URL.startswith(pb.SITE_URL))
        self.assertTrue(pb.LEADERBOARD_URL.endswith("/extended/"))

    def test_the_board_ranks_as_the_benchmarks_own_leaderboard(self) -> None:
        from _precedence import PROGRAMBENCH_SOURCE_URL, RANK_BENCHMARK_SITE, source_rank

        self.assertEqual(source_rank(PROGRAMBENCH_SOURCE_URL), RANK_BENCHMARK_SITE)

    def test_the_vals_mirror_ranks_below_it(self) -> None:
        from _precedence import PROGRAMBENCH_SOURCE_URL, VALS_RERUN_KEY_URLS, may_overwrite

        mirror = VALS_RERUN_KEY_URLS["programbench_almost"]
        self.assertTrue(may_overwrite(PROGRAMBENCH_SOURCE_URL, mirror))
        self.assertFalse(may_overwrite(mirror, PROGRAMBENCH_SOURCE_URL))


class TestValsBoard(unittest.TestCase):
    def test_the_vals_board_is_read_at_the_almost_task(self) -> None:
        """Which task is read *is* which metric the column stores, and the Vals
        board mirrors all three of ProgramBench's."""
        import fetch_vals

        self.assertEqual(fetch_vals.BENCHMARKS["programbench"], "programbench_almost")
        self.assertEqual(fetch_vals.task_of("programbench"), "almost")

    def test_the_column_names_the_reading_it_stores(self) -> None:
        # "programbench" alone would read as the authors' primary metric, which
        # is the one number this column is not.
        import fetch_vals

        self.assertTrue(fetch_vals.BENCHMARKS["programbench"].endswith("_almost"))

    def test_programbench_is_not_a_vals_owned_board(self) -> None:
        import fetch_vals

        self.assertNotIn("programbench", fetch_vals.VALS_OWN_BENCHMARKS)


if __name__ == "__main__":
    unittest.main()
