#!/usr/bin/env python3
"""Tests for the index's revision fallback. Run with ./test_index_revision_fallback.py

A benchmark that has re-run itself keeps a column per revision, because the two
are not comparable as published. For the index that leaves a hole: a model
measured only on the retired board contributes nothing to the benchmark it was
actually measured on, so nothing it did there can count -- which flatters a model
that scored near zero there.

REVISION_FALLBACKS closes the hole with a scale conversion, used inside the
index only and fitted from the models on both boards once there are enough of
them. These tests pin the properties that make that safe: the factor is fitted
rather than typed in and needs MIN_FALLBACK_OVERLAP models, the conversion never
touches llm.json's columns, it never displaces a number the current board did
publish, a converted model is ranked against the current field rather than its
own retired one, and every converted value is named in index_coverage.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import derive_indexes as di
from _revisions import KNOWN_REVISIONS, revision_key, revision_rank


FIRST_HAND = "https://artificialanalysis.ai/models/fixture"


def model(name: str, **scores) -> dict:
    return {
        "name": name,
        "scores": dict(scores),
        "scores_source": {key: FIRST_HAND for key in scores},
    }


DOC = {"benchmarks": {"b_1_1": {}, "b_1_0": {}}}
FACTORS = {"b_1_1": di.RevisionFactor("b_1_0", 2.0, 3)}

# Three models on both boards, each reading exactly twice as high on 1.1: the
# overlap a factor of 2.0 is fitted from.
OVERLAP = [
    model("both-1", b_1_0=10.0, b_1_1=20.0),
    model("both-2", b_1_0=20.0, b_1_1=40.0),
    model("both-3", b_1_0=40.0, b_1_1=80.0),
]


class Registry(unittest.TestCase):
    """Swaps REVISION_FALLBACKS for a single synthetic b_1_1 <- b_1_0 entry."""

    def setUp(self) -> None:
        self._saved = dict(di.REVISION_FALLBACKS)
        di.REVISION_FALLBACKS.clear()
        di.REVISION_FALLBACKS["b_1_1"] = "b_1_0"

    def tearDown(self) -> None:
        di.REVISION_FALLBACKS.clear()
        di.REVISION_FALLBACKS.update(self._saved)


class TestIndexScore(unittest.TestCase):
    def test_a_current_revision_score_is_used_as_is(self) -> None:
        self.assertEqual(di.index_score(model("m", b_1_1=40.0), "b_1_1", FACTORS), 40.0)

    def test_an_older_score_is_converted_onto_the_current_scale(self) -> None:
        self.assertEqual(di.index_score(model("m", b_1_0=10.0), "b_1_1", FACTORS), 20.0)

    def test_without_a_fitted_factor_nothing_is_converted(self) -> None:
        self.assertFalse(math.isfinite(di.index_score(model("m", b_1_0=10.0), "b_1_1")))

    def test_the_current_revision_always_wins_when_both_exist(self) -> None:
        """The conversion fills a hole; it never overrides a real measurement."""
        m = model("m", b_1_1=5.0, b_1_0=99.0)
        self.assertEqual(di.index_score(m, "b_1_1", FACTORS), 5.0)
        self.assertFalse(di.is_converted(m, "b_1_1", FACTORS))

    def test_a_model_on_neither_revision_stays_unmeasured(self) -> None:
        self.assertFalse(math.isfinite(di.index_score(model("m"), "b_1_1", FACTORS)))

    def test_a_benchmark_without_a_fallback_is_untouched(self) -> None:
        m = model("m", other_1_0=10.0)
        self.assertFalse(math.isfinite(di.index_score(m, "other_1_1", FACTORS)))

    def test_the_older_column_is_never_ranked_in_its_own_right(self) -> None:
        """Only the current key carries the fallback; the archive key is literal."""
        self.assertEqual(di.index_score(model("m", b_1_0=10.0), "b_1_0", FACTORS), 10.0)

    def test_a_converted_value_is_judged_by_its_archived_source(self) -> None:
        m = model("m", b_1_0=10.0)
        self.assertTrue(di.is_converted(m, "b_1_1", FACTORS))
        self.assertFalse(di.is_second_hand(m, "b_1_1", FACTORS))
        m["scores_source"]["b_1_0"] = "https://huggingface.co/org/m"
        self.assertTrue(di.is_second_hand(m, "b_1_1", FACTORS))


class TestFittedFactor(Registry):
    """The factor is fitted from llm.json, and only on enough overlap."""

    def test_it_is_fitted_from_the_models_on_both_boards(self) -> None:
        fitted = di.revision_factors(OVERLAP)["b_1_1"]
        self.assertEqual(fitted.older, "b_1_0")
        self.assertAlmostEqual(fitted.factor, 2.0)
        self.assertEqual(fitted.overlap, 3)

    def test_too_little_overlap_fits_nothing(self) -> None:
        """Two models cannot tell a scale from a coincidence."""
        self.assertEqual(di.revision_factors(OVERLAP[: di.MIN_FALLBACK_OVERLAP - 1]), {})

    def test_it_is_a_geometric_mean(self) -> None:
        """A model that doubled and one that halved cancel, as ratios should."""
        models = [
            model("up", b_1_0=10.0, b_1_1=20.0),
            model("down", b_1_0=20.0, b_1_1=10.0),
            model("flat", b_1_0=30.0, b_1_1=30.0),
        ]
        self.assertAlmostEqual(di.revision_factors(models)["b_1_1"].factor, 1.0)

    def test_a_zero_says_nothing_about_a_ratio(self) -> None:
        models = OVERLAP + [model("zero", b_1_0=0.0, b_1_1=5.0)]
        fitted = di.revision_factors(models)["b_1_1"]
        self.assertEqual(fitted.overlap, 3)
        self.assertAlmostEqual(fitted.factor, 2.0)


class TestConvertedPopulation(Registry):
    INDEX = di.IndexDef(
        key="idx", fallback_source_url="u", contributing=[("b_1_1", 1.0)]
    )

    def test_converted_models_join_the_current_population(self) -> None:
        """A converted score is compared against the current field like any
        other, and lands where its converted value puts it."""
        models = OVERLAP + [
            model("on-current-hi", b_1_1=90.0),
            model("on-current-lo", b_1_1=10.0),
            model("on-archive", b_1_0=25.0),      # -> 50.0, between the two
        ]
        values = di.compute_index(models, DOC, self.INDEX)
        order = sorted(values, key=lambda n: -values[n])
        self.assertLess(order.index("on-current-hi"), order.index("on-archive"))
        self.assertLess(order.index("on-archive"), order.index("on-current-lo"))

    def test_a_converted_model_is_compared_not_imputed(self) -> None:
        """It takes part in real head-to-heads rather than sitting out."""
        models = OVERLAP + [
            model("on-current", b_1_1=90.0),
            model("on-archive", b_1_0=25.0),
        ]
        record = di.comparisons(models, DOC, self.INDEX)
        self.assertGreater(record.pairs[("on-archive", "on-current")], 0.0)
        self.assertEqual(record.wins[("on-archive", "on-current")], 0.0)

    def test_without_enough_overlap_the_archive_model_sits_out(self) -> None:
        models = OVERLAP[:2] + [model("on-archive", b_1_0=25.0)]
        record = di.comparisons(models, DOC, self.INDEX)
        self.assertNotIn(("on-archive", "both-1"), record.pairs)

    def test_a_weak_archive_score_ranks_last_rather_than_at_the_median(self) -> None:
        """The point of the whole mechanism: near-zero is evidence, not a gap."""
        models = OVERLAP + [
            model("strong", b_1_1=90.0),
            model("weak-on-archive", b_1_0=0.1),
        ]
        values = di.compute_index(models, DOC, self.INDEX)
        self.assertEqual(min(values, key=lambda n: values[n]), "weak-on-archive")

    def test_the_coverage_record_names_the_conversion(self) -> None:
        models = OVERLAP + [model("on-archive", b_1_0=25.0)]
        coverage = di.index_coverage(models, self.INDEX)
        self.assertEqual(coverage["on-archive"]["converted"], ["b_1_1"])
        self.assertNotIn("converted", coverage["both-1"])

    def test_scored_count_counts_a_converted_benchmark_as_measured(self) -> None:
        index = di.IndexDef(key="i", fallback_source_url="u", contributing=[("b_1_1", 1.0)])
        self.assertEqual(di.scored_count(model("m", b_1_0=10.0), index, FACTORS), 1)
        self.assertEqual(di.scored_count(model("m"), index, FACTORS), 0)


class TestLiveRegistry(unittest.TestCase):
    """The registry as configured against the real llm.json."""

    def setUp(self) -> None:
        self.doc = json.loads(Path(__file__).resolve().with_name("llm.json").read_text())

    def test_every_fallback_names_columns_that_exist(self) -> None:
        for key, older in di.REVISION_FALLBACKS.items():
            self.assertIn(key, self.doc["benchmarks"], key)
            self.assertIn(older, self.doc["benchmarks"], older)

    def test_every_fallback_key_contributes_to_an_index(self) -> None:
        """A factor on a column no index reads would silently do nothing."""
        contributing = {k for index in di.INDEXES for k, _ in index.contributing}
        for key in di.REVISION_FALLBACKS:
            self.assertIn(key, contributing, key)

    def test_no_archived_column_contributes_to_an_index_directly(self) -> None:
        """The archive reaches an index only through the conversion, never twice."""
        contributing = {k for index in di.INDEXES for k, _ in index.contributing}
        for older in di.REVISION_FALLBACKS.values():
            self.assertNotIn(older, contributing, older)

    def test_a_split_benchmark_is_aggregated_through_its_current_board_only(self) -> None:
        """No archived revision reaches an index directly.

        An archived revision's score ranks a model against a field that no
        longer exists, and admitting both columns would count the benchmark
        twice for whoever was re-run and once for everyone else. The current
        board may or may not be aggregated -- a column can be carried without
        being voted on, and frontiercode_1_1 is (see the next test) -- but an
        archived one never is.
        """
        contributing = {k for index in di.INDEXES for k, _ in index.contributing}
        for base, labels in KNOWN_REVISIONS.items():
            current = revision_key(base, max(labels, key=revision_rank))
            for label in labels:
                key = revision_key(base, label)
                if key != current:
                    self.assertNotIn(key, contributing, key)

    def test_frontiercode_is_aggregated_through_extended_and_not_main(self) -> None:
        """One benchmark, one slot -- and for FrontierCode the slot is Extended.

        Main and Extended are the same 1.1 runs scored over nested task sets
        (150 tasks against the hardest 100), so they rank the same field to
        Spearman 0.99. Aggregating both would spend two of the coding group's
        slots on one measurement; Extended is kept because it is the larger
        sample. Main stays a column in llm.json that no index reads.
        """
        contributing = {k for index in di.INDEXES for k, _ in index.contributing}
        self.assertIn("frontiercode_extended_1_1", contributing)
        self.assertNotIn("frontiercode_1_1", contributing)
        # And the archive converts onto the scale of the board actually read.
        self.assertEqual(
            di.REVISION_FALLBACKS["frontiercode_extended_1_1"], "frontiercode_1_0"
        )
        self.assertNotIn("frontiercode_1_1", di.REVISION_FALLBACKS)

    def test_frontierswe_is_deliberately_left_without_a_fallback(self) -> None:
        """There is no scale to convert from: 1.0 published a pairwise win rate.

        Every other archive measures the same kind of thing as its current
        board, so an overlap gives a factor. FrontierSWE 1.0 ranked a different
        task set by dominance over a 17-model field, so a factor fitted on the
        models on both boards would be fitting the two metrics to each other.
        """
        self.assertNotIn("frontierswe_2_0", di.REVISION_FALLBACKS)

    def test_the_conversion_does_not_write_to_llm_json(self) -> None:
        """Columns keep exactly what each board published: a converted value
        exists only inside the index, and a model it answers for still has no
        current-revision score of its own."""
        doc = json.loads(json.dumps(self.doc))
        before = {m["name"]: dict(m.get("scores") or {}) for m in doc["models"]}
        di.refresh(doc)
        for m in doc["models"]:
            for key in di.REVISION_FALLBACKS:
                self.assertEqual(
                    (m.get("scores") or {}).get(key), before[m["name"]].get(key),
                    f"{m['name']}/{key}",
                )

    def test_every_converted_value_is_named_on_the_page(self) -> None:
        """index_coverage lists exactly the archive-only models a fitted
        factor answers for, so the page can say which inputs are conversions."""
        models = self.doc["models"]
        factors = di.revision_factors(models)
        for spec in di.INDEXES:
            coverage = di.index_coverage(models, spec, factors)
            for m in models:
                named = set((coverage.get(m["name"]) or {}).get("converted", []))
                expected = {
                    key for key, _ in spec.contributing
                    if di.is_converted(m, key, factors)
                }
                self.assertEqual(named, expected, f"{spec.key}/{m['name']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
