#!/usr/bin/env python3
"""APEX-Agents 1.0 and 1.1 are separate columns; these pin how a row is placed.

Mercor re-graded its board as 1.1 on 2026-09-08 under the same name. Epoch's
copy follows the board (1.1); llm-stats' rows are split by note and
announcement date (fetch_llmstats.apex_revision); a model card's bare label is
placed by date and dropped when too recent to tell (update.dated_revision_key).
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import fetch_epoch
import fetch_llmstats
import update

HERE = Path(__file__).resolve().parent


class TestLlmStatsRows(unittest.TestCase):
    def test_a_note_naming_the_revision_decides(self):
        self.assertEqual(fetch_llmstats.apex_revision("APEX-Agents v1.1 score 72.1", "2026-01-01"), "1.1")
        self.assertEqual(fetch_llmstats.apex_revision("APEX-Agents 1.1 64.2.", None), "1.1")
        self.assertEqual(fetch_llmstats.apex_revision("APEX-Agents v1 pass@1", "2026-11-01"), "1.0")

    def test_otherwise_the_announcement_date_decides(self):
        self.assertEqual(fetch_llmstats.apex_revision("Max reasoning effort", "2026-07-16"), "1.0")
        self.assertEqual(fetch_llmstats.apex_revision("Cited from Mercor APEX-Agents leaderboard", "2026-10-23"), "1.1")
        self.assertEqual(fetch_llmstats.apex_revision("", "2026-09-08"), "1.1")

    def test_no_note_and_no_date_is_unknowable(self):
        self.assertIsNone(fetch_llmstats.apex_revision(None, None))

    def test_pass_at_1_is_not_a_revision(self):
        self.assertEqual(fetch_llmstats.apex_revision("pass@1", "2026-08-28"), "1.0")
        self.assertEqual(fetch_llmstats.apex_revision("pass@1", "2026-10-28"), "1.1")

    def test_each_label_keeps_only_its_revision(self):
        row = {"analysis_method": "APEX-Agents v1.1", "announcement_date": "2026-11-16"}
        self.assertTrue(fetch_llmstats.board_row_kept("apex_agents_1_1", row["analysis_method"], row))
        self.assertFalse(fetch_llmstats.board_row_kept("apex_agents_1_0", row["analysis_method"], row))

    def test_the_unsplit_flat_field_is_never_offered(self):
        self.assertIn("apex_agents", fetch_llmstats.SPLIT_FLAT_FIELDS)
        mapping = json.loads((HERE / "llmstats-benchmark-name-mapping.json").read_text(encoding="utf-8"))
        self.assertNotIn("apex_agents", mapping)
        self.assertEqual(mapping["apex_agents_1_1"], "apex_agents_1_1")
        self.assertEqual(mapping["apex_agents_1_0"], "apex_agents_1_0")


class TestModelCards(unittest.TestCase):
    def test_a_bare_label_before_the_switch_is_1_0(self):
        self.assertEqual(update.dated_revision_key("apex_agents_1_0@dated", {"date_added": "2026-02-13"}), "apex_agents_1_0")

    def test_a_bare_label_after_the_switch_is_dropped(self):
        self.assertIsNone(update.dated_revision_key("apex_agents_1_0@dated", {"date_added": "2026-10-03"}))
        self.assertIsNone(update.dated_revision_key("apex_agents_1_0@dated", {}))

    def test_a_versioned_label_is_untouched(self):
        self.assertEqual(update.dated_revision_key("apex_agents_1_1", {"date_added": "2025-01-01"}), "apex_agents_1_1")

    def test_which_labels_name_a_revision(self):
        names = update._NAMES_A_REVISION_RE.search
        for label in ("APEX-Agents 1.1", "APEX-Agents v1.1", "APEX-Agents v1", "mercor/apex-agents-v1.1"):
            self.assertTrue(names(label), label)
        for label in ("APEX-Agents", "Apex-Agents (pass@1) Long-horizon professional workflows", "mercor/apex-agents"):
            self.assertFalse(names(label), label)

    def test_no_card_label_maps_to_the_retired_column(self):
        mapping = json.loads((HERE / "huggingface-benchmark-name-mapping.json").read_text(encoding="utf-8"))
        self.assertNotIn("apex_agents", set(mapping.values()))


class TestColumns(unittest.TestCase):
    def test_epoch_feeds_1_1(self):
        self.assertEqual(fetch_epoch.BENCHMARKS["apex-agents"].key, "apex_agents_1_1")

    def test_both_columns_exist_and_the_old_one_is_gone(self):
        doc = json.loads((HERE / "llm.json").read_text(encoding="utf-8"))
        self.assertIn("apex_agents_1_0", doc["benchmarks"])
        self.assertIn("apex_agents_1_1", doc["benchmarks"])
        self.assertNotIn("apex_agents", doc["benchmarks"])
        for model in doc["models"]:
            self.assertNotIn("apex_agents", model.get("scores") or {}, model["name"])

    def test_1_1_is_aggregated_and_1_0_falls_back_onto_it(self):
        import derive_indexes
        weighted = {key for index in derive_indexes.INDEXES for key, _ in index.contributing}
        self.assertIn("apex_agents_1_1", weighted)
        self.assertNotIn("apex_agents_1_0", weighted)
        self.assertEqual(derive_indexes.REVISION_FALLBACKS["apex_agents_1_1"], "apex_agents_1_0")


if __name__ == "__main__":
    unittest.main()
