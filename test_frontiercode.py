#!/usr/bin/env python3
"""Tests for the FrontierCode leaderboard reader. Run with ./test_frontiercode.py

The payload nests model -> effort -> subset, carries two metrics per leaf
(`new_score`, the site's Score column, and `correct`, the raw pass rate) and
keeps every past revision alongside the current one. These tests pin the choices
that decide which number reaches llm.json: subset, metric, which effort
represents a model published at several, and the revision split -- every
revision reported and labelled, never merged, because llm.json keeps a column
per revision and a 1.0 number is not a 1.1 number. Ordering is read off the key
so a future revision needs no code change.
"""

from __future__ import annotations

import unittest
from unittest import mock

import fetch_frontiercode as fc


def leaf(new_score: float, correct: float | None = None) -> dict:
    return {
        "correct": correct if correct is not None else new_score + 0.03,
        "new_score": new_score,
        "tokens": 1000.0,
        "cost": 1.5,
    }


PAYLOAD = {
    "v1": {
        "models": ["Single Effort", "Retired"],
        "harness": {},
        "subsets": {"main": 100, "extended": 150},
        "data": {
            "Single Effort": {"none": {"main": leaf(0.19), "extended": leaf(0.22)}},
            # Not re-run in 1.1, so 1.0 stays its only published number.
            "Retired": {"none": {"main": leaf(0.07), "extended": leaf(0.09)}},
        },
    },
    "v1_1": {
        "models": ["Multi Effort", "Single Effort", "Partial"],
        "harness": {"Multi Effort": "codex", "Single Effort": "mini-swe-agent"},
        "subsets": {"main": 100, "extended": 150},
        "data": {
            "Multi Effort": {
                "low": {"main": leaf(0.30), "extended": leaf(0.40)},
                "high": {"main": leaf(0.45), "extended": leaf(0.35)},
            },
            "Single Effort": {"none": {"main": leaf(0.245), "extended": leaf(0.30)}},
            # Published for extended only; must not surface as a main-subset row.
            "Partial": {"none": {"extended": leaf(0.11)}},
        },
    },
}


def scores(**kwargs) -> list[dict]:
    with mock.patch.object(fc, "fetch_json", return_value=PAYLOAD):
        return fc.get_scores(**kwargs)


class TestRevisionOrdering(unittest.TestCase):
    def test_numbers_in_the_key_order_the_revisions(self) -> None:
        keys = ["v2", "v1_10", "v1_2", "v1_1", "v1"]
        self.assertEqual(
            sorted(keys, key=fc.revision_rank, reverse=True),
            ["v2", "v1_10", "v1_2", "v1_1", "v1"],
        )

    def test_a_future_revision_outranks_todays(self) -> None:
        self.assertGreater(fc.revision_rank("v1_2"), fc.revision_rank("v1_1"))
        self.assertGreater(fc.revision_rank("v2"), fc.revision_rank("v1_9"))

    def test_key_without_digits_is_treated_as_oldest(self) -> None:
        self.assertLess(fc.revision_rank("draft"), fc.revision_rank("v1"))

    def test_labels_match_what_the_site_prints(self) -> None:
        self.assertEqual(fc.revision_label("v1"), "1.0")
        self.assertEqual(fc.revision_label("v1_1"), "1.1")
        self.assertEqual(fc.revision_label("v2_3"), "2.3")

    def test_newest_first_regardless_of_payload_order(self) -> None:
        self.assertEqual(
            [key for key, _ in fc.revisions_newest_first(PAYLOAD)], ["v1_1", "v1"]
        )


class TestRevisionBlock(unittest.TestCase):
    def test_label_and_payload_key_both_resolve(self) -> None:
        self.assertEqual(fc.revision_block(PAYLOAD, "1.1"), ("v1_1", PAYLOAD["v1_1"]))
        self.assertEqual(fc.revision_block(PAYLOAD, "v1_1"), ("v1_1", PAYLOAD["v1_1"]))

    def test_merging_every_revision_is_the_default(self) -> None:
        self.assertEqual(fc.DEFAULT_REVISION, fc.ALL_REVISIONS)

    def test_unknown_revision_raises(self) -> None:
        with self.assertRaises(ValueError):
            fc.revision_block(PAYLOAD, "2.0")


class TestBestEffort(unittest.TestCase):
    def test_highest_metric_wins(self) -> None:
        efforts = PAYLOAD["v1_1"]["data"]["Multi Effort"]
        self.assertEqual(fc.best_effort(efforts, "main", "new_score")[0], "high")
        # The ranking is per subset, and this model is stronger at low there.
        self.assertEqual(fc.best_effort(efforts, "extended", "new_score")[0], "low")

    def test_missing_subset_is_no_row(self) -> None:
        efforts = PAYLOAD["v1_1"]["data"]["Partial"]
        self.assertEqual(fc.best_effort(efforts, "main", "new_score"), (None, None))

    def test_leaf_without_the_metric_never_wins(self) -> None:
        efforts = {
            "low": {"main": {"new_score": 0.2}},
            "high": {"main": {"new_score": None}},
            "max": {"main": {"new_score": True}},  # a bool is not a score
        }
        self.assertEqual(fc.best_effort(efforts, "main", "new_score")[0], "low")


class TestGetScores(unittest.TestCase):
    def by_model(self, revision: str = "1.1", **kwargs) -> dict[str, dict]:
        """Rows of one revision, keyed by model.

        Defaults to the current revision: every revision is reported now, so a
        model published in both appears twice and a flat model index would be
        ambiguous.
        """
        return {row["model"]: row for row in scores(revision=revision, **kwargs)}

    def test_score_is_the_percentage_of_new_score(self) -> None:
        # llm.json stores the site's Score column as a percentage; the payload
        # keeps fractions.
        self.assertEqual(self.by_model()["Single Effort"]["score"], 24.5)

    def test_correct_is_reported_but_not_used_as_the_score(self) -> None:
        row = self.by_model()["Single Effort"]
        self.assertEqual(row["new_score"], 0.245)
        self.assertEqual(row["correct"], 0.275)
        self.assertNotEqual(row["score"], round(row["correct"] * 100, 2))

    def test_metric_switch_reports_the_pass_rate(self) -> None:
        self.assertEqual(self.by_model(metric="correct")["Single Effort"]["score"], 27.5)

    def test_best_effort_row_is_the_one_reported(self) -> None:
        row = self.by_model()["Multi Effort"]
        self.assertEqual((row["effort"], row["score"]), ("high", 45.0))
        self.assertEqual(row["efforts"], ["high", "low"])

    def test_subset_switch_changes_the_chosen_effort(self) -> None:
        row = self.by_model(subset="extended")["Multi Effort"]
        self.assertEqual((row["effort"], row["score"]), ("low", 40.0))

    def test_model_absent_from_the_subset_is_dropped(self) -> None:
        self.assertNotIn("Partial", self.by_model())
        self.assertIn("Partial", self.by_model(subset="extended"))

    def test_pinned_older_revision_reports_its_own_numbers(self) -> None:
        rows = self.by_model(revision="1.0")
        self.assertEqual(rows["Single Effort"]["score"], 19.0)
        self.assertNotIn("Multi Effort", rows)

    def test_a_model_in_both_revisions_reports_both_numbers(self) -> None:
        """The heart of the split: 19.0 and 24.5 are two measurements, not one.

        Merging them -- the old behaviour -- silently dropped whichever
        revision lost, so a column mixed 1.0 and 1.1 numbers with nothing
        recording which was which.
        """
        both = {r["revision"]: r["score"] for r in scores() if r["model"] == "Single Effort"}
        self.assertEqual(both, {"1.0": 19.0, "1.1": 24.5})

    def test_model_dropped_from_the_newest_revision_still_reports(self) -> None:
        row = self.by_model(revision="1.0")["Retired"]
        self.assertEqual((row["revision"], row["score"]), ("1.0", 7.0))

    def test_every_revision_is_reported(self) -> None:
        self.assertEqual({r["revision"] for r in scores()}, {"1.0", "1.1"})
        self.assertEqual(
            {(r["revision"], r["model"]) for r in scores()},
            {
                ("1.1", "Multi Effort"), ("1.1", "Single Effort"),
                ("1.0", "Single Effort"), ("1.0", "Retired"),
            },
        )

    def test_a_future_revision_is_reported_beside_the_others(self) -> None:
        payload = {**PAYLOAD, "v1_2": {
            "models": ["Single Effort"],
            "subsets": {"main": 100},
            "data": {"Single Effort": {"none": {"main": leaf(0.31)}}},
        }}
        with mock.patch.object(fc, "fetch_json", return_value=payload):
            rows = fc.get_scores()
        self.assertEqual({r["revision"] for r in rows}, {"1.0", "1.1", "1.2"})
        # Newest first, so a consumer reading in order meets the current run first.
        self.assertEqual(rows[0]["revision"], "1.2")

    def test_rows_are_ranked_within_their_own_revision(self) -> None:
        """Ranking across revisions would compare two different task sets."""
        rows = scores()
        by_revision: dict[str, list[dict]] = {}
        for row in rows:
            by_revision.setdefault(row["revision"], []).append(row)
        for revision, group in by_revision.items():
            self.assertEqual(
                [r["rank"] for r in group], list(range(1, len(group) + 1)), revision
            )
        self.assertEqual(rows[0]["model"], "Multi Effort")

    def test_harness_is_carried_through(self) -> None:
        self.assertEqual(self.by_model()["Single Effort"]["harness"], "mini-swe-agent")

    def test_unknown_subset_raises(self) -> None:
        with self.assertRaises(ValueError):
            scores(subset="nope")

    def test_unknown_revision_raises(self) -> None:
        with self.assertRaises(ValueError):
            scores(revision="9.9")


class TestSourceUrls(unittest.TestCase):
    def test_scraped_url_and_reader_facing_page_differ(self) -> None:
        # The score's source is the leaderboard page a reader can open, not the
        # JSON the page happens to load.
        self.assertTrue(fc.URL.endswith(".json"))
        self.assertEqual(fc.LEADERBOARD_URL, "https://cognition.com/frontiercode")


if __name__ == "__main__":
    unittest.main()
