#!/usr/bin/env python3
"""Tests for the index's Bradley-Terry math. Run with ./test_index_math.py

The index estimates one ability per model from head-to-head comparisons drawn
from the benchmarks two models have both run. These tests pin the properties
that make that worth doing, and in particular the two the percentile-average it
replaced could not have:

  * a benchmark's field is priced in, so losing to strong opponents costs less
    than losing to weak ones; and
  * a gap is never filled, so a model is never scored on a stand-in value.
"""

from __future__ import annotations

import unittest

import derive_indexes as di


def model(name: str, **scores) -> dict:
    return {"name": name, "scores": dict(scores)}


DOC: dict = {"benchmarks": {}}
LOWER = {"benchmarks": {"b": {"lower_is_better": True}}}


def index(*contributing, ratio: float = 0.0) -> di.IndexDef:
    return di.IndexDef(
        key="idx",
        fallback_source_url="u",
        contributing=list(contributing),
        transfer_ratio=ratio,
    )


class TestComparisons(unittest.TestCase):
    def test_a_pair_is_worth_the_benchmark_weight(self) -> None:
        record = di.comparisons(
            [model("hi", b=90.0), model("lo", b=10.0)], DOC, index(("b", 0.7))
        )
        self.assertEqual(record.wins[("hi", "lo")], 0.7)
        self.assertEqual(record.wins[("lo", "hi")], 0.0)
        self.assertEqual(record.pairs[("hi", "lo")], 0.7)
        self.assertEqual(record.pairs[("lo", "hi")], 0.7)

    def test_equal_values_split_the_comparison(self) -> None:
        record = di.comparisons(
            [model("a", b=50.0), model("c", b=50.0)], DOC, index(("b", 1.0))
        )
        self.assertEqual(record.wins[("a", "c")], 0.5)
        self.assertEqual(record.wins[("c", "a")], 0.5)

    def test_a_lower_is_better_column_compares_the_other_way(self) -> None:
        record = di.comparisons(
            [model("clean", b=5.0), model("noisy", b=80.0)], LOWER, index(("b", 1.0))
        )
        self.assertEqual(record.wins[("clean", "noisy")], 1.0)
        self.assertEqual(record.wins[("noisy", "clean")], 0.0)

    def test_a_benchmark_with_one_scored_model_supports_nothing(self) -> None:
        """No comparison, and its weight is out of the total so it dilutes
        nobody."""
        record = di.comparisons(
            [model("a", b=50.0, lonely=10.0), model("c", b=40.0)],
            DOC,
            index(("b", 1.0), ("lonely", 1.0)),
        )
        self.assertEqual(record.weight, {"b": 1.0})

    def test_a_missing_score_produces_no_comparison(self) -> None:
        """The whole of how a gap is handled: it is absent, not filled."""
        record = di.comparisons(
            [model("both", b=50.0, c=50.0), model("one", b=40.0)],
            DOC,
            index(("b", 1.0), ("c", 1.0)),
        )
        self.assertEqual(record.pairs[("both", "one")], 1.0)  # the shared b only


class TestFieldStrength(unittest.TestCase):
    """The defect that prompted the rewrite.

    Coming last among five near-frontier models and coming last among a hundred
    ordinary ones were the same percentile, 0.0, and were averaged as if they
    were the same evidence. A comparison knows the difference.
    """

    def setUp(self) -> None:
        # Two side benchmarks of equal weight, one contested by strong models
        # and one by weak ones. X and Y each come last on theirs, and are
        # identical everywhere else.
        self.models = [
            model("strong-1", wide=90.0, elite=80.0),
            model("strong-2", wide=89.0, elite=79.0),
            model("strong-3", wide=88.0, elite=78.0),
            model("strong-4", wide=87.0, elite=77.0),
            model("weak-1", wide=40.0, ordinary=30.0),
            model("weak-2", wide=39.0, ordinary=29.0),
            model("weak-3", wide=38.0, ordinary=28.0),
            model("weak-4", wide=37.0, ordinary=27.0),
            model("x-last-among-strong", wide=60.0, elite=10.0),
            model("y-last-among-weak", wide=60.0, ordinary=1.0),
        ]
        self.index = index(("wide", 1.0), ("elite", 1.0), ("ordinary", 1.0))

    def test_losing_to_strong_opponents_costs_less(self) -> None:
        values = di.compute_index(self.models, DOC, self.index)
        self.assertGreater(
            values["x-last-among-strong"],
            values["y-last-among-weak"],
            "coming last in a strong field must beat coming last in a weak one",
        )

    def test_both_models_are_measured_on_the_same_weight(self) -> None:
        """So the difference above is the opponents, not the coverage: the
        shrinkage cannot be what separates them."""
        record = di.comparisons(self.models, DOC, self.index)
        by_name = {m["name"]: m for m in self.models}
        covered = {
            name: sum(
                weight
                for key, weight in record.weight.items()
                if key in by_name[name]["scores"]
            )
            for name in ("x-last-among-strong", "y-last-among-weak")
        }
        self.assertEqual(covered["x-last-among-strong"], 2.0)
        self.assertEqual(covered["y-last-among-weak"], 2.0)


class TestCoverageReliability(unittest.TestCase):
    def test_an_uncalibrated_index_shrinks_nobody(self) -> None:
        self.assertEqual(di.coverage_reliability([0.15], 0.0), 1.0)

    def test_it_is_covered_weight_against_itself_plus_the_ratio(self) -> None:
        self.assertAlmostEqual(di.coverage_reliability([1.0, 0.5], 0.3), 1.5 / 1.8)

    def test_more_weight_is_believed_more(self) -> None:
        thin = di.coverage_reliability([0.3], 0.5)
        thick = di.coverage_reliability([1.0, 0.9, 0.8], 0.5)
        self.assertLess(thin, thick)
        self.assertLess(thick, 1.0)

    def test_one_heavy_benchmark_can_beat_several_light_ones(self) -> None:
        """Weights are precisions, so what counts is how much weight was
        measured rather than how many columns."""
        self.assertGreater(
            di.coverage_reliability([1.0], 0.5),
            di.coverage_reliability([0.2, 0.2, 0.2], 0.5),
        )

    def test_no_weight_is_not_a_division_by_zero(self) -> None:
        self.assertEqual(di.coverage_reliability([], 0.5), 1.0)


class TestShrinkage(unittest.TestCase):
    def test_a_thinly_covered_model_is_pulled_toward_the_field(self) -> None:
        models = [
            model("top", a=90.0, b=90.0, c=90.0),
            model("mid", a=50.0, b=50.0, c=50.0),
            model("low", a=10.0, b=10.0, c=10.0),
            model("thin-and-good", a=95.0),
        ]
        spec = [("a", 1.0), ("b", 1.0), ("c", 1.0)]
        loose = di.compute_index(models, DOC, index(*spec, ratio=0.0))
        tight = di.compute_index(models, DOC, index(*spec, ratio=2.0))
        self.assertLess(tight["thin-and-good"], loose["thin-and-good"])
        # The fully measured model is barely touched by the same ratio.
        self.assertLess(
            abs(tight["top"] - loose["top"]),
            abs(tight["thin-and-good"] - loose["thin-and-good"]),
        )


class TestUnranked(unittest.TestCase):
    def test_too_little_measured_weight_is_null(self) -> None:
        models = [
            model("full", a=90.0, b=90.0, c=90.0, d=90.0, e=90.0),
            model("also-full", a=50.0, b=50.0, c=50.0, d=50.0, e=50.0),
            model("sliver", e=60.0),
        ]
        values = di.compute_index(
            models,
            DOC,
            index(("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0), ("e", 0.1)),
        )
        self.assertIsNone(values["sliver"])
        self.assertIsNotNone(values["full"])

    def test_an_unranked_model_is_still_a_valid_opponent(self) -> None:
        """Dropping it from the ranking must not drop the evidence it carries:
        a model too thin to score is still a perfectly good thing to be
        compared against, and the links it supplies place everyone else."""
        spec = index(
            ("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0), ("e", 0.1)
        )
        models = [
            model("full", a=90.0, b=90.0, c=90.0, d=90.0, e=90.0),
            model("also-full", a=50.0, b=50.0, c=50.0, d=50.0, e=50.0),
            model("sliver", e=70.0),
        ]
        self.assertIsNone(di.compute_index(models, DOC, spec)["sliver"])

        record = di.comparisons(models, DOC, spec)
        self.assertEqual(record.pairs[("sliver", "full")], 0.1)
        self.assertEqual(record.wins[("sliver", "full")], 0.0)
        self.assertEqual(record.wins[("sliver", "also-full")], 0.1)
        self.assertIn("sliver", di.bradley_terry(record))


class TestDeterminism(unittest.TestCase):
    def test_the_row_order_does_not_change_the_result(self) -> None:
        models = [
            model("a", x=10.0, y=80.0),
            model("b", x=50.0, y=20.0),
            model("c", x=90.0),
            model("d", y=55.0),
        ]
        spec = index(("x", 1.0), ("y", 0.4), ratio=0.2)
        forward = di.compute_index(models, DOC, spec)
        backward = di.compute_index(list(reversed(models)), DOC, spec)
        self.assertEqual(forward, backward)

    def test_identical_evidence_ties(self) -> None:
        models = [
            model("twin-a", x=40.0),
            model("twin-b", x=40.0),
            model("other", x=90.0),
        ]
        values = di.compute_index(models, DOC, index(("x", 1.0)))
        self.assertEqual(values["twin-a"], values["twin-b"])
        self.assertGreater(values["other"], values["twin-a"])


class TestScale(unittest.TestCase):
    def test_a_value_is_a_win_rate_in_points(self) -> None:
        models = [
            model("best", x=90.0),
            model("mid", x=50.0),
            model("worst", x=10.0),
        ]
        values = di.compute_index(models, DOC, index(("x", 1.0)))
        for name, value in values.items():
            self.assertGreaterEqual(value, 0, name)
            self.assertLessEqual(value, di.SCALE, name)
        # The middle of a symmetric three-model field wins about half of its
        # head-to-heads.
        self.assertAlmostEqual(values["mid"] / di.SCALE, 0.5, places=2)


class TestLiveIndexes(unittest.TestCase):
    """The configured indexes, rather than a synthetic one."""

    def test_every_index_is_calibrated(self) -> None:
        """A 0.0 ratio silently shrinks nobody, which is a configuration
        oversight rather than a decision."""
        for spec in di.INDEXES:
            with self.subTest(index=spec.key):
                self.assertGreater(spec.transfer_ratio, 0.0)

    def test_the_prior_is_small_enough_to_leave_the_data_in_charge(self) -> None:
        self.assertGreater(di.BT_PRIOR, 0.0)
        self.assertLessEqual(di.BT_PRIOR, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
