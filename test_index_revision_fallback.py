#!/usr/bin/env python3
"""Tests for the index's revision fallback. Run with ./test_index_revision_fallback.py

A benchmark that has re-run itself keeps a column per revision, because the two
are not comparable as published. For the index that leaves a hole: a model
measured only on the retired board contributes nothing to the benchmark it was
actually measured on, so it is imputed at the median -- which flatters a model
that scored near zero there.

REVISION_FALLBACKS closes the hole with a scale conversion, used inside the
index only. These tests pin the three properties that make that safe: the
conversion never touches llm.json's columns, it never displaces a number the
current board did publish, and a converted model is ranked against the current
field rather than its own retired one.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import derive_indexes as di


def model(name: str, **scores) -> dict:
    return {"name": name, "scores": dict(scores)}


DOC = {"benchmarks": {"b_1_1": {}, "b_1_0": {}}}


class TestIndexScore(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = dict(di.REVISION_FALLBACKS)
        di.REVISION_FALLBACKS.clear()
        di.REVISION_FALLBACKS["b_1_1"] = ("b_1_0", 2.0)

    def tearDown(self) -> None:
        di.REVISION_FALLBACKS.clear()
        di.REVISION_FALLBACKS.update(self._saved)

    def test_a_current_revision_score_is_used_as_is(self) -> None:
        self.assertEqual(di.index_score(model("m", b_1_1=40.0), "b_1_1"), 40.0)

    def test_an_older_score_is_converted_onto_the_current_scale(self) -> None:
        self.assertEqual(di.index_score(model("m", b_1_0=10.0), "b_1_1"), 20.0)

    def test_the_current_revision_always_wins_when_both_exist(self) -> None:
        """The conversion fills a hole; it never overrides a real measurement."""
        m = model("m", b_1_1=5.0, b_1_0=99.0)
        self.assertEqual(di.index_score(m, "b_1_1"), 5.0)

    def test_a_model_on_neither_revision_stays_unmeasured(self) -> None:
        self.assertFalse(math.isfinite(di.index_score(model("m"), "b_1_1")))

    def test_a_benchmark_without_a_fallback_is_untouched(self) -> None:
        m = model("m", other_1_0=10.0)
        self.assertFalse(math.isfinite(di.index_score(m, "other_1_1")))

    def test_the_older_column_is_never_ranked_in_its_own_right(self) -> None:
        """Only the current key carries the fallback; the archive key is literal."""
        self.assertEqual(di.index_score(model("m", b_1_0=10.0), "b_1_0"), 10.0)


class TestPercentilePopulation(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = dict(di.REVISION_FALLBACKS)
        di.REVISION_FALLBACKS.clear()
        di.REVISION_FALLBACKS["b_1_1"] = ("b_1_0", 2.0)

    def tearDown(self) -> None:
        di.REVISION_FALLBACKS.clear()
        di.REVISION_FALLBACKS.update(self._saved)

    def test_converted_models_join_the_current_population(self) -> None:
        models = [
            model("on-current-hi", b_1_1=90.0),
            model("on-current-lo", b_1_1=10.0),
            model("on-archive", b_1_0=25.0),      # -> 50.0, between the two
        ]
        pct = di.percentile_map(models, "b_1_1", DOC)
        self.assertEqual(sorted(pct), ["on-archive", "on-current-hi", "on-current-lo"])
        self.assertEqual(pct["on-current-lo"], 0.0)
        self.assertEqual(pct["on-archive"], 0.5)
        self.assertEqual(pct["on-current-hi"], 1.0)

    def test_a_weak_archive_score_ranks_last_rather_than_at_the_median(self) -> None:
        """The point of the whole mechanism: near-zero is evidence, not a gap."""
        models = [
            model("strong", b_1_1=90.0),
            model("mid", b_1_1=50.0),
            model("weak-on-archive", b_1_0=0.1),
        ]
        pct = di.percentile_map(models, "b_1_1", DOC)
        self.assertEqual(pct["weak-on-archive"], 0.0)

    def test_scored_count_counts_a_converted_benchmark_as_measured(self) -> None:
        index = di.IndexDef(key="i", fallback_source_url="u", contributing=[("b_1_1", 1.0)])
        self.assertEqual(di.scored_count(model("m", b_1_0=10.0), index), 1)
        self.assertEqual(di.scored_count(model("m"), index), 0)


class TestLiveRegistry(unittest.TestCase):
    """The registry as configured against the real llm.json."""

    def setUp(self) -> None:
        self.doc = json.loads(Path(__file__).resolve().with_name("llm.json").read_text())

    def test_every_fallback_names_columns_that_exist(self) -> None:
        for key, (older, factor) in di.REVISION_FALLBACKS.items():
            self.assertIn(key, self.doc["benchmarks"], key)
            self.assertIn(older, self.doc["benchmarks"], older)
            self.assertGreater(factor, 0)

    def test_every_fallback_key_contributes_to_an_index(self) -> None:
        """A factor on a column no index reads would silently do nothing."""
        contributing = {k for index in di.INDEXES for k, _ in index.contributing}
        for key in di.REVISION_FALLBACKS:
            self.assertIn(key, contributing, key)

    def test_no_archived_column_contributes_to_an_index_directly(self) -> None:
        """The archive reaches an index only through the conversion, never twice."""
        contributing = {k for index in di.INDEXES for k, _ in index.contributing}
        for _key, (older, _f) in di.REVISION_FALLBACKS.items():
            self.assertNotIn(older, contributing, older)

    def test_the_conversion_does_not_write_to_llm_json(self) -> None:
        """Columns keep exactly what each board published."""
        for key, (older, factor) in di.REVISION_FALLBACKS.items():
            for m in self.doc["models"]:
                stored = (m.get("scores") or {}).get(key)
                if stored is None:
                    continue
                # A stored current-revision value is never the converted one.
                old = (m.get("scores") or {}).get(older)
                if old is not None:
                    self.assertNotAlmostEqual(
                        stored, old * factor, places=6,
                        msg=f"{m['name']}/{key} looks like a converted value",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
