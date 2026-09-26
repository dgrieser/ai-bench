#!/usr/bin/env python3
"""Tests for the cybergym.io reader (CyberGym and ExploitGym). Run with ./test_cybergym.py

Both boards put the rows the columns must not hold right next to the ones they
do, and mark the difference in fields rather than in the layout:

  * ``focus: "agent"`` rows are a security vendor's pipeline around a model,
    scoring ten or more points above the same model in a plain coding agent;
  * CyberGym's ``score_10`` is pass@10 on rows that also carry ``score_x1``,
    and a row run over thirty trials sits in the same field;
  * ExploitGym mixes 2-hour and 6-hour budgets, a subset run and the retired
    v0 task set on one board, and prints counts rather than percentages.

What these tests pin is that each of those is dropped by its field, that the
ExploitGym count is divided by the stated task total, and that a file whose
shape changed stops the read instead of reporting an empty board.
"""

from __future__ import annotations

import unittest

import fetch_cybergym as cg


CYBERGYM = {
    "level1": [
        {"agent": "Claude Code", "model": "GLM-5.3", "score_10": 0.845, "trials": 1, "focus": "model"},
        # The same model inside a security vendor's pipeline.
        {"agent": "Sangfor AI", "model": "GLM-5.3", "score_10": 0.972, "trials": 1, "focus": "agent"},
        {"agent": "MDASH", "model": "Multi-model (GPT-5.4, Claude Opus 4.6)", "score_10": 0.91, "trials": 1},
        # pass@10 beside its own pass@1.
        {"agent": "OpenHands", "model": "Claude Sonnet 4", "score_10": 0.179,
         "score_x1": 0.0199, "trials": 1, "focus": "model"},
        # Thirty trials in the same field.
        {"agent": "Anthropic Agent", "model": "Claude Sonnet 4.5", "score_10": 0.667,
         "trials": 30, "focus": "model"},
        {"agent": "Kimi Agent", "model": "Kimi K2.5", "score_10": 0.413, "trials": 1, "focus": "model"},
        {"agent": "Meta Agent", "model": "Muse Spark 1.1 (helpful-only)", "score_10": 0.59, "trials": 1},
    ],
    "level0": [{"model": "GPT-4.1", "score_10": 0.5}],
}

EXPLOITGYM = {
    "instances": {"total": 869},
    "results": [
        {"model": "GPT-5.6 Sol (reasoning max)", "on_target": 293, "eval_note": "6h timeout",
         "version": "v1", "subRows": [{"on_target": 216, "eval_note": "2h timeout cutoff"}]},
        {"model": "GLM-5.3", "on_target": 130, "eval_note": "rescaled 6h timeout"},
        # The maintainers' 2-hour run of an older model.
        {"model": "GPT-5.5", "on_target": 129, "eval_note": "2h timeout", "version": "v1"},
        {"model": "Muse Spark 1.1", "on_target": 7, "eval_note": "4h timeout", "version": "v1"},
        {"model": "GLM-5.2", "on_target": 79, "eval_note": "selected subset"},
        {"model": "GLM-5.2", "on_target": 79, "eval_note": "6h timeout", "focus": "agent"},
        {"model": "Claude Mythos Preview", "on_target": 157, "eval_note": "6h timeout",
         "version": "v0"},
        {"model": "Hidden Model", "on_target": 300, "eval_note": "6h timeout", "hidden": True},
    ],
}


class CyberGymRows(unittest.TestCase):
    def rows(self):
        return cg.cybergym_rows(CYBERGYM)

    def test_keeps_only_model_focused_single_trial_pass_at_1(self):
        rows, _ = self.rows()
        self.assertEqual(
            {r["model"]: r["score"] for r in rows},
            {"GLM-5.3": 84.5, "Kimi K2.5": 41.3, "Muse Spark 1.1": 59.0},
        )

    def test_reports_what_it_dropped(self):
        _, dropped = self.rows()
        self.assertEqual(dropped, {"agent": 2, "pass@k": 1, "trials": 1})

    def test_the_label_note_is_not_part_of_the_name(self):
        rows, _ = self.rows()
        muse = next(r for r in rows if r["raw"].startswith("Muse"))
        self.assertEqual(muse["model"], "Muse Spark 1.1")
        self.assertEqual(muse["raw"], "Muse Spark 1.1 (helpful-only)")

    def test_a_file_without_level1_stops_the_read(self):
        with self.assertRaises(ValueError):
            cg.cybergym_rows({"level0": []})


class ExploitGymRows(unittest.TestCase):
    def rows(self):
        return cg.exploitgym_rows(EXPLOITGYM)

    def test_six_hour_full_set_rows_as_a_share_of_the_total(self):
        rows, _ = self.rows()
        self.assertEqual(
            {r["model"]: r["score"] for r in rows},
            {"GPT-5.6 Sol": round(293 / 869 * 100, 2), "GLM-5.3": round(130 / 869 * 100, 2)},
        )

    def test_the_two_hour_cutoff_of_a_six_hour_run_is_not_read(self):
        rows, _ = self.rows()
        sol = next(r for r in rows if r["model"] == "GPT-5.6 Sol")
        self.assertEqual(sol["solved"], 293)

    def test_reports_what_it_dropped(self):
        _, dropped = self.rows()
        self.assertEqual(dropped, {"agent": 1, "hidden": 1, "v0": 1, "subset": 1, "budget": 2})

    def test_no_denominator_stops_the_read(self):
        with self.assertRaises(ValueError):
            cg.exploitgym_rows({"results": EXPLOITGYM["results"]})


class GetScores(unittest.TestCase):
    def test_every_row_names_its_column(self):
        files = {cg.data_url(cg.CYBERGYM_KEY): CYBERGYM, cg.data_url(cg.EXPLOITGYM_KEY): EXPLOITGYM}
        rows = cg.get_scores(fetch=files.__getitem__)
        self.assertEqual({r["benchmark"] for r in rows}, set(cg.KEYS))

    def test_an_empty_board_is_an_error(self):
        files = {
            cg.data_url(cg.CYBERGYM_KEY): {"level1": []},
            cg.data_url(cg.EXPLOITGYM_KEY): EXPLOITGYM,
        }
        with self.assertRaises(ValueError):
            cg.get_scores(fetch=files.__getitem__)

    def test_each_column_is_credited_to_its_own_page(self):
        self.assertEqual(cg.source_url(cg.CYBERGYM_KEY), "https://www.cybergym.io/cybergym/")
        self.assertEqual(cg.source_url(cg.EXPLOITGYM_KEY), "https://www.cybergym.io/exploitgym/")


if __name__ == "__main__":
    unittest.main()
