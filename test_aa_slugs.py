#!/usr/bin/env python3
"""Tests for reading a model from several Artificial Analysis slugs.

Run with ./test_aa_slugs.py

AA sometimes tracks one model under more than one slug, each carrying a
different slice of the benchmarks. A list value in the llm -> AA mapping reads
them all; these tests pin the two halves of that: the priority order the list
resolves to, and the per-benchmark merge that fills gaps without ever letting a
later slug override a measured value.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import _artificialanalysis_mapping as aa_mapping
import update


class MappingFileTestCase(unittest.TestCase):
    def write_mapping(self, mapping: dict) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "mapping.json"
        tmp.write_text(json.dumps(mapping), encoding="utf-8")
        return tmp


class TestMappingFile(MappingFileTestCase):
    def test_reads_a_string_and_a_list_alike(self) -> None:
        path = self.write_mapping({"one": "one-aa", "two": ["two-new", "two-old"]})
        self.assertEqual(
            aa_mapping.load_llm_to_aa_slugs(path),
            {"one": ["one-aa"], "two": ["two-new", "two-old"]},
        )

    def test_leading_slug_stays_the_single_slug_view(self) -> None:
        path = self.write_mapping({"two": ["two-new", "two-old"]})
        self.assertEqual(aa_mapping.load_llm_to_aa_mapping(path), {"two": "two-new"})

    def test_mapped_slugs_include_the_trailing_ones(self) -> None:
        path = self.write_mapping({"one": "one-aa", "two": ["two-new", "two-old"]})
        self.assertEqual(
            aa_mapping.mapped_aa_slugs(path), {"one-aa", "two-new", "two-old"}
        )

    def test_junk_values_are_dropped(self) -> None:
        path = self.write_mapping({"a": None, "b": [], "c": [""], "d": ["ok", 7]})
        self.assertEqual(aa_mapping.load_llm_to_aa_slugs(path), {"d": ["ok"]})

    def test_writing_a_new_pick_keeps_the_rest_of_the_list(self) -> None:
        path = self.write_mapping({"two": ["two-new", "two-old"]})
        aa_mapping.add_aa_mapping("two", "two-newest", path)
        self.assertEqual(
            aa_mapping.load_llm_to_aa_slugs(path),
            {"two": ["two-newest", "two-new", "two-old"]},
        )

    def test_writing_a_slug_already_listed_changes_nothing(self) -> None:
        path = self.write_mapping({"two": ["two-new", "two-old"]})
        aa_mapping.add_aa_mapping("two", "two-old", path)
        self.assertEqual(
            aa_mapping.load_llm_to_aa_slugs(path), {"two": ["two-new", "two-old"]}
        )


class TestSentinels(MappingFileTestCase):
    """A sentinel is a marker, never a slug -- and only one of them is an answer.

    propose.py parks a name it could not resolve as __pending__. Read back as a
    slug, that line posed as a mapping: update_artificialanalysis_mapping.py saw
    a truthy value, printed "mapped already" and skipped the model, so a parked
    name was never asked about again. Every other source drops sentinels in its
    load_*_to_slug_mapping and excludes only PENDING from its reviewed set;
    these pin the same two halves here.
    """

    def test_sentinels_are_not_slugs(self) -> None:
        path = self.write_mapping(
            {
                "parked": "__pending__",
                "declined": "__unmappable__",
                "closed": "__closed_weights__",
                "real": "real-aa",
            }
        )
        self.assertEqual(aa_mapping.load_llm_to_aa_slugs(path), {"real": ["real-aa"]})
        self.assertEqual(aa_mapping.mapped_aa_slugs(path), {"real-aa"})

    def test_a_sentinel_beside_a_slug_leaves_the_slug(self) -> None:
        path = self.write_mapping({"two": ["__pending__", "two-old"]})
        self.assertEqual(aa_mapping.load_llm_to_aa_slugs(path), {"two": ["two-old"]})

    def test_a_parked_name_is_not_reviewed(self) -> None:
        path = self.write_mapping(
            {
                "parked": "__pending__",
                "declined": "__unmappable__",
                "closed": "__closed_weights__",
                "real": "real-aa",
            }
        )
        self.assertEqual(
            aa_mapping.load_reviewed_llm_names(path),
            {"declined", "closed", "real"},
        )

    def test_closed_weights_can_be_rechecked(self) -> None:
        path = self.write_mapping({"declined": "__unmappable__", "closed": "__closed_weights__"})
        self.assertEqual(
            aa_mapping.load_reviewed_llm_names(path, include_closed=False),
            {"declined"},
        )


class TestResolveAaSlugs(MappingFileTestCase):
    def test_unmapped_model_reads_its_own_slug(self) -> None:
        path = self.write_mapping({})
        self.assertEqual(
            update.resolve_aa_slugs(["m"], {"m"}, path), {"m": ["m"]}
        )

    def test_model_absent_from_aa_is_left_out(self) -> None:
        path = self.write_mapping({})
        self.assertEqual(update.resolve_aa_slugs(["m"], set(), path), {})

    def test_list_resolves_in_its_own_order(self) -> None:
        path = self.write_mapping({"m": ["m-2", "m-1"]})
        self.assertEqual(
            update.resolve_aa_slugs(["m"], {"m-1", "m-2"}, path), {"m": ["m-2", "m-1"]}
        )

    def test_slugs_missing_from_aa_are_skipped(self) -> None:
        path = self.write_mapping({"m": ["m-2", "m-1"]})
        self.assertEqual(update.resolve_aa_slugs(["m"], {"m-1"}, path), {"m": ["m-1"]})

    def test_own_slug_leads_unless_the_list_places_it(self) -> None:
        path = self.write_mapping({"m": ["m-old"]})
        self.assertEqual(
            update.resolve_aa_slugs(["m"], {"m", "m-old"}, path), {"m": ["m", "m-old"]}
        )

        path = self.write_mapping({"m": ["m-old", "m"]})
        self.assertEqual(
            update.resolve_aa_slugs(["m"], {"m", "m-old"}, path), {"m": ["m-old", "m"]}
        )


class TestMergeAaModels(unittest.TestCase):
    def test_leading_record_wins_every_value_it_has(self) -> None:
        merged = update.merge_aa_models(
            [
                {"slug": "m-2", "evaluations": {"hle": 0.4}},
                {"slug": "m-1", "evaluations": {"hle": 0.9}},
            ]
        )
        self.assertEqual(merged["slug"], "m-2")
        self.assertEqual(merged["evaluations"]["hle"], 0.4)

    def test_later_records_fill_gaps_per_benchmark(self) -> None:
        merged = update.merge_aa_models(
            [
                {"evaluations": {"hle": 0.4, "gpqa": None}},
                {"evaluations": {"gpqa": 0.7, "tau2": 0.5}},
            ]
        )
        self.assertEqual(
            merged["evaluations"], {"hle": 0.4, "gpqa": 0.7, "tau2": 0.5}
        )

    def test_zero_counts_as_untested(self) -> None:
        # AA reports some untested benchmarks as 0, the same reading
        # normalize_aa_value() applies before a score is written.
        merged = update.merge_aa_models(
            [{"evaluations": {"hle": 0}}, {"evaluations": {"hle": 0.6}}]
        )
        self.assertEqual(merged["evaluations"]["hle"], 0.6)

    def test_false_is_a_value_not_a_gap(self) -> None:
        merged = update.merge_aa_models(
            [{"microevals_enabled": False}, {"microevals_enabled": True}]
        )
        self.assertIs(merged["microevals_enabled"], False)

    def test_top_level_fields_fill_too(self) -> None:
        merged = update.merge_aa_models(
            [
                {"context": "", "params": "70B"},
                {"context": "128k", "params": "8B"},
            ]
        )
        self.assertEqual(merged["context"], "128k")
        self.assertEqual(merged["params"], "70B")

    def test_records_are_not_mutated(self) -> None:
        first = {"evaluations": {"hle": None}}
        update.merge_aa_models([first, {"evaluations": {"hle": 0.6}}])
        self.assertIsNone(first["evaluations"]["hle"])


if __name__ == "__main__":
    unittest.main()
