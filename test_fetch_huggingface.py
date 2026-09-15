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
  * an HTML table whose first row is a caption over the comparison columns
    ("Reference models") rather than the header, which made the header look
    like a data row so no column named the model and the whole table was
    dropped -- while the two-row header beside it, which starts with the row
    that *does* name the model, has to be left as it is;
  * a benchmark captioned inside its own label cell ("HLE Expert-level
    reasoning"), where the caption tripped the MoE-architecture filter;
  * grouped thousands separators in an Elo-scale cell ("1,441"), read as 1;
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


class TestHtmlHeaderBands(unittest.TestCase):
    """A caption row above the header, as IFM's K2-Horizon cards write it."""

    @staticmethod
    def html(*rows: str) -> str:
        return "<table>" + "".join(rows) + "</table>"

    def test_colspan_caption_is_not_read_as_the_header(self):
        html = self.html(
            '<tr><th></th><th colspan="2">Reference models</th></tr>',
            "<tr><th>Benchmark</th><th>Example-30B</th><th>Rival-27B</th></tr>",
            '<tr><td colspan="3">Coding</td></tr>',
            "<tr><td>SWE-bench Verified</td><td>70.6</td><td>47.7</td></tr>",
        )
        scores = fh.extract_scores_from_tables(fh.parse_html_tables(html), REPO)
        self.assertEqual(scores, {"SWE-bench Verified": 70.6})

    def test_a_full_width_first_row_still_leads_its_table(self):
        # Nothing to promote past: this really is the header, and reading the
        # row under it as one would turn a score into a benchmark name.
        html = self.html(
            "<tr><th>Benchmark</th><th>Example-30B</th><th>Rival-27B</th></tr>",
            "<tr><td>SWE-bench Verified</td><td>70.6</td><td>47.7</td></tr>",
            "<tr><td>LiveCodeBench v6</td><td>37.4</td><td>29.8</td></tr>",
        )
        scores = fh.extract_scores_from_tables(fh.parse_html_tables(html), REPO)
        self.assertEqual(scores, {"SWE-bench Verified": 70.6, "LiveCodeBench v6": 37.4})

    def test_two_row_header_keeps_the_row_that_names_the_model(self):
        # MiniCPM5's shape, and the reason the markup cannot decide this: the
        # same colspan sits above a rowspan header whose second row carries the
        # *rest* of the comparison columns. Promoting here would drop the
        # model's own column and take a rival's numbers.
        html = self.html(
            '<tr><th rowspan="2"></th><th rowspan="2">Example-30B</th>'
            '<th colspan="2">Rivals</th></tr>',
            "<tr><th>Rival Large</th><th>Rival Small</th></tr>",
            "<tr><td>MMLU-Pro</td><td>84.9</td><td>80.1</td><td>71.2</td></tr>",
        )
        scores = fh.extract_scores_from_tables(fh.parse_html_tables(html), REPO)
        self.assertEqual(scores, {"MMLU-Pro": 84.9})

    def test_a_link_cell_naming_the_repo_does_not_promote(self):
        # An artifact table names the repo in a URL, not in a header. Its first
        # row fills the table's width, which is what says "header" -- a caption
        # band labels groups, so it names fewer cells than the row beneath it.
        md = md_table(
            "| Artifact | Link | Status |",
            "| :---- | :---- | :---- |",
            "| Model card | [HF](https://huggingface.co/example-org/Example-30B) | Available |",
            "| Blog post | [Blog](https://example.org/blog/) | Available |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores, {})

    def test_transposed_tables_are_left_alone(self):
        # Models in rows: the model's own row must not be read as a header.
        md = md_table(
            "| Model | MMLU-Pro | AIME24 |",
            "| :---- | :---: | :---: |",
            "| Example-30B | 84.9 | 88.1 |",
            "| Rival-27B | 80.1 | 77.9 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores, {"MMLU-Pro": 84.9, "AIME24": 88.1})


class TestCaptionedLabels(unittest.TestCase):
    def test_benchmark_caption_survives_the_architecture_filter(self):
        md = md_table(
            "| Benchmark | Example-30B |",
            "| :---- | :---: |",
            "| HLE Expert-level reasoning | 18.6 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores, {"HLE Expert-level reasoning": 18.6})

    def test_expert_count_rows_are_still_specs(self):
        md = md_table(
            "| Benchmark | Example-30B |",
            "| :---- | :---: |",
            "| # Experts | 128 |",
            "| Activated Experts | 8 |",
            "| Experts per Token | 8 |",
            "| GPQA Diamond | 80.8 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores, {"GPQA Diamond": 80.8})


class TestNumberParsing(unittest.TestCase):
    def test_grouped_thousands_are_one_number(self):
        # An Elo-scale column (GDPval-AA at 1,441) used to stop at the "1".
        self.assertEqual(fh.parse_score("1,441"), 1441.0)
        self.assertEqual(fh.parse_score("1,298.34"), 1298.34)

    def test_a_lone_comma_group_is_left_alone(self):
        # Not a thousands separator: three digits are required after the comma,
        # so a decimal comma reads as it always did rather than as 631.
        self.assertEqual(fh.parse_score("63,1"), 63.0)


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
