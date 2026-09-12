#!/usr/bin/env python3
"""Tests for the FrontierSWE leaderboard reader. Run with ./test_frontierswe.py

FrontierSWE publishes two boards and they do not share a metric: V2 (the root
page) keys its flight payload by score mode ("abs") and then by trial
aggregation, and scores each model as a percentage in `overall`; V1 (/v1,
"preserved as published") keys the aggregations directly and ranks by
`dominance`, a pairwise win rate, with `overall` holding an average per-task
rank instead. llm.json keeps a column per revision, so these tests pin what
reaches each one: which page, which field, which scale -- and that neither
board can ever be read as the other.
"""

from __future__ import annotations

import unittest
from unittest import mock

import fetch_frontierswe as fs


def v2(model: str, overall: float, harness: str = "proximus") -> dict:
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


def v1(model: str, dominance: float, mean_rank: float, harness: str = "Claude Code") -> dict:
    return {
        "model": model,
        "harness": harness,
        "overall": mean_rank,
        "dominance": dominance,
        "implementation": mean_rank,
        "performance": mean_rank,
        "research": mean_rank,
    }


V2_ENTRIES = {
    "abs": {
        "mean": [v2("Alpha", 56.29), v2("Beta", 32.204), v2("Gamma", 4.117)],
        "best": [v2("Alpha", 66.65), v2("Beta", 44.17), v2("Gamma", 10.79)],
        "worst": [v2("Alpha", 44.47), v2("Beta", 22.2), v2("Gamma", 0.13)],
    }
}

V1_ENTRIES = {
    # As the site serves it: ordered by dominance, and with no "worst" view.
    "mean": [
        v1("Alpha", 0.8823529411764706, 2.88),
        v1("Retired", 0.24632352941176472, 13.06, harness="Kimi CLI"),
    ],
    "best": [v1("Alpha", 0.90625, 2.5), v1("Retired", 0.3, 11.0, harness="Kimi CLI")],
}

PAGES = {fs.URL: V2_ENTRIES, fs.V1_URL: V1_ENTRIES}


def scores(**kwargs) -> list[dict]:
    """get_scores() over both stubbed boards, addressed by the URL each reads."""
    with mock.patch.object(fs, "fetch_html", side_effect=lambda url: url), mock.patch.object(
        fs, "extract_entries", side_effect=lambda url: PAGES[url]
    ):
        return fs.get_scores(**kwargs)


def by_model(revision: str, **kwargs) -> dict[str, dict]:
    return {row["model"]: row for row in scores(revision=revision, **kwargs)}


class TestBoards(unittest.TestCase):
    def test_each_revision_has_its_own_page(self) -> None:
        # V1 is not a view of the current board: it is a page of its own, and
        # the column it fills cites that page.
        self.assertEqual(fs.BOARD_URLS["2.0"], fs.URL)
        self.assertEqual(fs.BOARD_URLS["1.0"], fs.V1_URL)
        self.assertTrue(fs.V1_URL.endswith("/v1"))

    def test_newest_first(self) -> None:
        self.assertEqual(fs.revisions_newest_first(), ["2.0", "1.0"])

    def test_reporting_every_revision_is_the_default(self) -> None:
        self.assertEqual(fs.DEFAULT_REVISION, fs.ALL_REVISIONS)

    def test_the_sites_own_spelling_resolves(self) -> None:
        self.assertEqual(fs.resolve_revision("v1"), ["1.0"])
        self.assertEqual(fs.resolve_revision("2.0"), ["2.0"])

    def test_a_revision_with_no_board_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            fs.resolve_revision("3.0")
        self.assertIn("2.0", str(ctx.exception))


class TestSelectGroup(unittest.TestCase):
    def test_v2_reads_through_the_score_mode(self) -> None:
        self.assertEqual(fs.select_group(V2_ENTRIES, "mean"), V2_ENTRIES["abs"]["mean"])

    def test_v1_reads_the_group_directly(self) -> None:
        self.assertEqual(fs.select_group(V1_ENTRIES, "mean", "1.0"), V1_ENTRIES["mean"])

    def test_missing_score_mode_names_what_is_there(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            fs.select_group({"rel": {"mean": []}}, "mean")
        self.assertIn("'abs'", str(ctx.exception))
        self.assertIn("rel", str(ctx.exception))

    def test_missing_group_names_the_views(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            fs.select_group(V2_ENTRIES, "median")
        self.assertIn("'median'", str(ctx.exception))
        self.assertIn("worst", str(ctx.exception))

    def test_v1_payload_is_not_read_as_v2(self) -> None:
        # Taking the un-nested, dominance-scored board for the current one
        # would put a win rate in the 2.0 column.
        with self.assertRaises(ValueError):
            fs.select_group(V1_ENTRIES, "best")

    def test_v2_payload_is_not_read_as_v1(self) -> None:
        # And the other direction: if /v1 ever starts serving the current
        # shape, its scale has to be looked at before anything is read.
        with self.assertRaises(ValueError) as ctx:
            fs.select_group(V2_ENTRIES, "mean", "1.0")
        self.assertIn("abs", str(ctx.exception))

    def test_a_view_v1_never_published_raises_rather_than_dropping_the_board(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            fs.select_group(V1_ENTRIES, "worst", "1.0")
        self.assertIn("V1.0", str(ctx.exception))


class TestV2Scores(unittest.TestCase):
    def test_mean_is_the_default_view(self) -> None:
        self.assertEqual([r["score"] for r in scores(revision="2.0")], [56.3, 32.2, 4.1])

    def test_group_selects_the_view(self) -> None:
        self.assertEqual(
            [r["score"] for r in scores(revision="2.0", group="best")], [66.7, 44.2, 10.8]
        )
        self.assertEqual(
            [r["score"] for r in scores(revision="2.0", group="worst")], [44.5, 22.2, 0.1]
        )

    def test_keeps_the_unrounded_percentage_alongside(self) -> None:
        self.assertEqual(by_model("2.0")["Beta"]["overall"], 32.204)

    def test_carries_the_harness(self) -> None:
        self.assertEqual(by_model("2.0")["Alpha"]["harness"], "proximus")

    def test_drops_rows_without_a_usable_score(self) -> None:
        entries = {"abs": {"mean": [v2("Alpha", 10.0), {"model": "NoScore"}, "junk"]}}
        with mock.patch.object(fs, "fetch_html", return_value=""), mock.patch.object(
            fs, "extract_entries", return_value=entries
        ):
            rows = fs.get_scores(revision="2.0")
        self.assertEqual([r["model"] for r in rows], ["Alpha"])


class TestV1Scores(unittest.TestCase):
    def test_score_is_dominance_as_a_percentage(self) -> None:
        self.assertEqual(by_model("1.0")["Alpha"]["score"], 88.2)

    def test_the_average_rank_is_reported_but_never_scored(self) -> None:
        """V1's `overall` is a position in its field, not a measurement."""
        row = by_model("1.0")["Alpha"]
        self.assertEqual(row["mean_rank"], 2.88)
        self.assertNotIn("overall", row)
        self.assertNotEqual(row["score"], 2.88)

    def test_keeps_the_raw_win_rate_alongside(self) -> None:
        self.assertEqual(by_model("1.0")["Retired"]["dominance"], 0.24632352941176472)

    def test_carries_the_per_model_harness(self) -> None:
        self.assertEqual(by_model("1.0")["Retired"]["harness"], "Kimi CLI")

    def test_a_row_without_dominance_is_dropped(self) -> None:
        entries = {"mean": [v1("Alpha", 0.5, 2.0), {"model": "NoScore", "overall": 3.0}]}
        with mock.patch.object(fs, "fetch_html", return_value=""), mock.patch.object(
            fs, "extract_entries", return_value=entries
        ):
            rows = fs.get_scores(revision="1.0")
        self.assertEqual([r["model"] for r in rows], ["Alpha"])


class TestRevisionSplit(unittest.TestCase):
    def test_every_revision_is_reported(self) -> None:
        self.assertEqual(
            {(r["revision"], r["model"]) for r in scores()},
            {
                ("2.0", "Alpha"), ("2.0", "Beta"), ("2.0", "Gamma"),
                ("1.0", "Alpha"), ("1.0", "Retired"),
            },
        )

    def test_a_model_on_both_boards_reports_both_numbers(self) -> None:
        """The heart of the split: 56.3 and 88.2 are two measurements.

        Merging them -- the old behaviour, which read the root page alone --
        left V1's board unrepresented and its retired models unscored.
        """
        both = {r["revision"]: r["score"] for r in scores() if r["model"] == "Alpha"}
        self.assertEqual(both, {"2.0": 56.3, "1.0": 88.2})

    def test_newest_revision_comes_first(self) -> None:
        self.assertEqual(scores()[0]["revision"], "2.0")

    def test_a_model_dropped_from_the_current_board_still_reports(self) -> None:
        row = by_model("1.0")["Retired"]
        self.assertEqual((row["revision"], row["score"]), ("1.0", 24.6))

    def test_rows_are_ranked_within_their_own_revision(self) -> None:
        """Ranking across revisions would compare a percentage with a win rate."""
        by_revision: dict[str, list[dict]] = {}
        for row in scores():
            by_revision.setdefault(row["revision"], []).append(row)
        self.assertEqual([r["rank"] for r in by_revision["2.0"]], [1, 2, 3])
        self.assertEqual([r["rank"] for r in by_revision["1.0"]], [1, 2])

    def test_a_pinned_revision_reads_only_its_own_page(self) -> None:
        seen: list[str] = []
        with mock.patch.object(fs, "fetch_html", side_effect=lambda url: seen.append(url) or url), \
                mock.patch.object(fs, "extract_entries", side_effect=lambda url: PAGES[url]):
            fs.get_scores(revision="1.0")
        self.assertEqual(seen, [fs.V1_URL])


class TestExtractEntries(unittest.TestCase):
    def test_pulls_entries_out_of_the_flight_chunks(self) -> None:
        import json

        payload = '17:["$","$L24",null,{"entries":{"abs":{"mean":[{"model":"Alpha"}]}},"tail":1}]'
        halves = [payload[:20], payload[20:]]
        html = "".join(
            f"<script>self.__next_f.push([1,{json.dumps(half)}])</script>" for half in halves
        )
        self.assertEqual(fs.extract_entries(html), {"abs": {"mean": [{"model": "Alpha"}]}})

    def test_missing_entries_raises(self) -> None:
        with self.assertRaises(ValueError):
            fs.extract_entries("<html></html>")


if __name__ == "__main__":
    unittest.main()
