#!/usr/bin/env python3
"""SWE-bench Pro's public split has its own column; reports naming no split stay in swe_bench_pro."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import fetch_llmstats

HERE = Path(__file__).resolve().parent


class TestLlmStatsNotes(unittest.TestCase):
    def kept(self, note):
        return (fetch_llmstats.board_row_kept("swe_bench_pro", note),
                fetch_llmstats.board_row_kept("swe_bench_pro_public", note))

    def test_a_note_naming_the_public_split_goes_to_the_public_column(self):
        self.assertEqual(self.kept("SWE-Bench Pro (Public). Reasoning effort xhigh."), (False, True))
        self.assertEqual(self.kept("Public set, single attempt"), (False, True))

    def test_a_note_naming_no_split_stays_unrecorded(self):
        self.assertEqual(self.kept("OpenHands"), (True, False))
        self.assertEqual(self.kept(None), (True, False))

    def test_the_refined_set_is_neither(self):
        self.assertEqual(self.kept("Refined public set (problematic tasks corrected)"), (False, False))

    def test_the_flat_field_is_not_read(self):
        self.assertIn("swe_bench_pro", fetch_llmstats.BOARD_FIELDS)


class TestMappings(unittest.TestCase):
    def test_the_dataset_id_is_the_public_split(self):
        mapping = json.loads((HERE / "huggingface-benchmark-name-mapping.json").read_text(encoding="utf-8"))
        self.assertEqual(mapping["ScaleAI/SWE-bench_Pro"], "swe_bench_pro_public")
        self.assertEqual(mapping["SWEBench Pro (public)"], "swe_bench_pro_public")
        self.assertEqual(mapping["SWE-bench Pro"], "swe_bench_pro")

    def test_the_index_reads_the_public_split(self):
        import derive_indexes
        weighted = {key for index in derive_indexes.INDEXES for key, _ in index.contributing}
        self.assertIn("swe_bench_pro_public", weighted)
        self.assertNotIn("swe_bench_pro", weighted)
        self.assertEqual(derive_indexes.REVISION_FALLBACKS["swe_bench_pro_public"], "swe_bench_pro")


if __name__ == "__main__":
    unittest.main()
