#!/usr/bin/env python3
"""Tests for fetch_huggingface.py table parsing and structured eval metadata.

The fixtures pin the model-card shapes that used to lose or misalign scores:

  * a leading Category column ("| Category | Benchmark | Model | ... |"),
    where the old parser read category names as benchmark labels;
  * continuation rows in those tables that omit the category cell entirely,
    which shifted every value one column left (the score recorded then
    belonged to a *competitor*, not the model);
  * soft line breaks (\\x0b from word-processor exports) inside header cells,
    which made splitlines() shred the table so it was never parsed;
  * the Hub's structured eval metadata (evalResults + model-index), which the
    README-table parser never saw at all.
"""

from __future__ import annotations

import unittest

import fetch_huggingface as fh

REPO = "example-org/Example-30B"


def md_table(*lines: str) -> str:
    return "\n".join(lines) + "\n"


class TestCategoryColumnTables(unittest.TestCase):
    def test_labels_come_from_benchmark_column(self):
        md = md_table(
            "| Category | Benchmark | Example-30B | Rival-27B |",
            "| :---- | :---- | :---: | :---: |",
            "| *General* | MCP Atlas | **75.5** | 62.5 |",
            "|  | DeepSearch QA | 74.6 | 71.1 |",
            "| *Coding* | SWE-Bench Pro | 51.2 | 50.2 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(
            scores,
            {"MCP Atlas": 75.5, "DeepSearch QA": 74.6, "SWE-Bench Pro": 51.2},
        )

    def test_continuation_row_missing_category_cell_realigns(self):
        # Second row has one cell fewer than the header: the category cell is
        # omitted, not empty. 83.1 is the model's score; 65.8 is the rival's.
        md = md_table(
            "| Category | Benchmark | Example-30B | Rival-27B |",
            "| :---- | :---- | :---: | :---: |",
            "| MATH | AIME24 | 88.1 | 77.9 |",
            "| AIME25 | 83.1 | 65.8 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores, {"AIME24": 88.1, "AIME25": 83.1})

    def test_soft_line_break_inside_header_cell(self):
        md = md_table(
            "| Category | Benchmark | Example-30B\x0bHigh Reasoning | Rival-27B |",
            "| :---- | :---- | :---: | :---: |",
            "| *General* | Gaia2 | 43.3 | 40.0 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores, {"Gaia2": 43.3})

    def test_lower_is_better_cells_are_skipped(self):
        md = md_table(
            "| Category | Benchmark | Example-30B | Rival-27B |",
            "| :---- | :---- | :---: | :---: |",
            "| *Safety* | CI Memories | Violation (↓): 26.4 <br> Coverage: 64.8 | Violation (↓): 12.1 |",
            "| *General* | IFBench | 77.0 | 76.0 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores, {"IFBench": 77.0})

    def test_markdown_escapes_are_removed_from_labels(self):
        md = md_table(
            "| Category | Benchmark | Example-30B |",
            "| :---- | :---- | :---: |",
            "| *Agentic* | 𝛕3\\-Banking | 23.5 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores, {"𝛕3-Banking": 23.5})


class TestCrossTableMerge(unittest.TestCase):
    def test_exact_name_match_beats_loose_match(self):
        # The base-model table matches only loosely ("Example-30B Base") and
        # appears first; the post-trained table must still win.
        md = md_table(
            "| Benchmark | Example-30B Base | Other Base |",
            "| :---- | :---: | :---: |",
            "| MMLU-Pro | 73.2 | 69.2 |",
        ) + "\n" + md_table(
            "| Benchmark | Example-30B | Other |",
            "| :---- | :---: | :---: |",
            "| MMLU-Pro | 84.9 | 80.1 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores, {"MMLU-Pro": 84.9})

    def test_equally_matched_tables_keep_best_run(self):
        md = md_table(
            "| Benchmark | Example-30B |",
            "| :---- | :---: |",
            "| AIME24 | 88.1 |",
        ) + "\n" + md_table(
            "| Benchmark | Example-30B |",
            "| :---- | :---: |",
            "| AIME24 | 96.7 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores, {"AIME24": 96.7})

    def test_params_rows_are_not_benchmarks(self):
        md = md_table(
            "| Category | Benchmark | Example-30B |",
            "| :---- | :---- | :---: |",
            "| **Params** | **#Activated / #Total** | **15B / 309B** |",
            "| **General** | BBH | 88.5 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores, {"BBH": 88.5})


class TestEvalResults(unittest.TestCase):
    def test_extracts_widget_entries(self):
        payload = {
            "evalResults": [
                {
                    "filename": ".eval_results/x.yaml",
                    "verified": False,
                    "data": {
                        "dataset": {"id": "Idavidrein/gpqa", "task_id": "diamond"},
                        "value": 83.5,
                    },
                    "pullRequest": 6,
                },
                {
                    "data": {
                        "dataset": {"id": "cais/hle", "task_id": "hle"},
                        "value": 22.0,
                    }
                },
                {
                    "data": {
                        "dataset": {
                            "id": "MathArena/aime_2026",
                            "task_id": "MathArena/aime_2026",
                        },
                        "value": 94.7,
                    }
                },
            ]
        }
        self.assertEqual(
            fh.extract_eval_results(payload),
            {
                # Task suffix kept only when it adds information.
                "Idavidrein/gpqa (diamond)": 83.5,
                "cais/hle": 22.0,
                "MathArena/aime_2026": 94.7,
            },
        )

    def test_ignores_malformed_entries(self):
        payload = {
            "evalResults": [
                {"data": {"dataset": {"id": "a/b"}, "value": "n/a"}},
                {"data": {"dataset": {}, "value": 1.0}},
                {"data": {"dataset": {"id": "c/d"}, "value": True}},
                "not-a-dict",
                {"data": {"dataset": {"id": "e/f"}, "value": "76.0%"}},
            ]
        }
        self.assertEqual(fh.extract_eval_results(payload), {"e/f": 76.0})


class TestModelIndex(unittest.TestCase):
    def test_extracts_metrics(self):
        payload = {
            "model-index": [
                {
                    "name": "example",
                    "results": [
                        {
                            "dataset": {"name": "AI2 Reasoning Challenge (25-Shot)", "type": "ai2_arc"},
                            "metrics": [{"type": "acc_norm", "name": "normalized accuracy", "value": 62.03}],
                        },
                        {
                            "dataset": {"name": "MultiBench", "type": "multibench"},
                            "metrics": [
                                {"type": "acc", "name": "accuracy", "value": 50.0},
                                {"type": "f1", "name": "f1", "value": 61.5},
                            ],
                        },
                    ],
                }
            ]
        }
        self.assertEqual(
            fh.extract_model_index(payload),
            {
                # Single metric keeps the bare dataset name; several metrics
                # per dataset get disambiguating suffixes.
                "AI2 Reasoning Challenge (25-Shot)": 62.03,
                "MultiBench (accuracy)": 50.0,
                "MultiBench (f1)": 61.5,
            },
        )

    def test_handles_missing_index(self):
        self.assertEqual(fh.extract_model_index({"model-index": None}), {})
        self.assertEqual(fh.extract_model_index({}), {})


if __name__ == "__main__":
    unittest.main()
