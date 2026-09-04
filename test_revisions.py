#!/usr/bin/env python3
"""Tests for the benchmark-revision naming helpers. Run with ./test_revisions.py

Three sources spell the same revision three different ways, and every one of
them has to land on the same llm.json column or the split leaks: a row filed
under the wrong revision is the blend the columns exist to end, and one filed
under no revision at all must be refused rather than guessed at.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import _revisions as rev


class TestRevisionLabel(unittest.TestCase):
    def test_every_source_spelling_reduces_to_one_label(self) -> None:
        # Cognition's payload key, DeepSWE's artifact directory, SWE-Marathon's
        # board name, and the label itself.
        for spelling in ("v1_1", "v1.1", "1.1", "V1.1"):
            self.assertEqual(rev.revision_label(spelling), "1.1", spelling)

    def test_a_bare_major_is_filled_out_to_dot_zero(self) -> None:
        # Cognition calls it "v1" and prints "1.0"; DeepSWE serves /artifacts/v1/.
        self.assertEqual(rev.revision_label("v1"), "1.0")
        self.assertEqual(rev.revision_label("v1.0"), "1.0")

    def test_a_name_without_digits_is_left_alone(self) -> None:
        # It will match no column, which is the point.
        self.assertEqual(rev.revision_label("nightly"), "nightly")


class TestRevisionRank(unittest.TestCase):
    def test_revisions_order_oldest_to_newest(self) -> None:
        names = ["v2", "v1_2", "v1", "v1_1"]
        self.assertEqual(sorted(names, key=rev.revision_rank), ["v1", "v1_1", "v1_2", "v2"])

    def test_double_digit_minor_sorts_above_single(self) -> None:
        # String ordering would put v1.10 before v1.2.
        self.assertGreater(rev.revision_rank("v1.10"), rev.revision_rank("v1.2"))

    def test_a_name_without_digits_sorts_oldest(self) -> None:
        self.assertLess(rev.revision_rank("nightly"), rev.revision_rank("v1"))


class TestRevisionKey(unittest.TestCase):
    def test_key_matches_the_spelling_llm_json_already_uses(self) -> None:
        # terminal_bench_2_1 set the pattern for a versioned column.
        self.assertEqual(rev.revision_key("deepswe", "1.1"), "deepswe_1_1")
        self.assertEqual(rev.revision_key("swe_marathon", "v1"), "swe_marathon_1_0")


class TestKnownRevisionKey(unittest.TestCase):
    def test_a_tracked_revision_resolves_to_its_column(self) -> None:
        self.assertEqual(rev.known_revision_key("frontiercode", "v1_1"), "frontiercode_1_1")

    def test_an_unnamed_revision_is_refused(self) -> None:
        self.assertIsNone(rev.known_revision_key("deepswe", None))
        self.assertIsNone(rev.known_revision_key("deepswe", ""))

    def test_an_untracked_revision_is_refused_rather_than_rounded(self) -> None:
        # A 2.0 release must not quietly land in the 1.1 column.
        self.assertIsNone(rev.known_revision_key("deepswe", "2.0"))

    def test_an_unversioned_benchmark_has_no_revision_column(self) -> None:
        self.assertIsNone(rev.known_revision_key("livecodebench", "1.1"))


class TestColumnsExist(unittest.TestCase):
    """Every revision declared here needs the matching llm.json column.

    KNOWN_REVISIONS is what the ingests check a row against, so a revision
    listed without a column would let rows through to a key nothing renders.
    """

    def test_llm_json_carries_a_column_for_every_known_revision(self) -> None:
        doc = json.loads((Path(__file__).resolve().with_name("llm.json")).read_text())
        for base, labels in rev.KNOWN_REVISIONS.items():
            for label in labels:
                key = rev.revision_key(base, label)
                self.assertIn(key, doc["benchmarks"], key)

    def test_the_unsplit_key_is_gone_from_llm_json(self) -> None:
        doc = json.loads((Path(__file__).resolve().with_name("llm.json")).read_text())
        for base in rev.KNOWN_REVISIONS:
            self.assertNotIn(base, doc["benchmarks"])
            for model in doc["models"]:
                self.assertNotIn(base, model["scores"], f"{model['name']}/{base}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
