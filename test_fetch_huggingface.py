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
    README-table parser never saw at all;
  * a lab's own shorthand for its model ("DS-V4.1-Flash" for
    DeepSeek-V4.1-Flash), which matched nothing, so the whole frontier table
    was dropped and only the *base* column -- the one header spelling the repo
    out -- was read;
  * a parameter count rounded differently in the heading than in the repo name
    ("Solar Open (102B)" for Solar-Open-100B);
  * a model derived from the repo sitting in the repo's own table
    ("DeepSeek-R1-0528-Qwen3-8B"), whose numbers landed on the 685B model;
  * "Pass@1 20.4" read as 1, "1st" read as 1, "63,1" read as 63, and a model
    name read as a score because it had a number in it;
  * a card reporting one benchmark at several settings, in the tables and in
    the structured metadata alike, where whichever came first used to win.
"""

from __future__ import annotations

import unittest
from unittest import mock

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

    def test_a_decimal_comma_is_a_decimal_point(self):
        # Not a thousands separator: three digits are required after the comma,
        # so a card writing "63,1" is writing 63.1, not 63 and not 631.
        self.assertEqual(fh.parse_score("63,1"), 63.1)
        self.assertEqual(fh.parse_score("8,25"), 8.25)


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


class TestModelNaming(unittest.TestCase):
    """Which column is this model? The question the whole ingest turns on."""

    def test_a_family_initialism_names_the_model(self):
        # DeepSeek's own card heads its frontier table "DS-V4.1-Flash".
        md = md_table(
            "| Benchmark (Metric) | Opus-5.0 | DS-V4-Pro | DS-V4.1-Flash |",
            "| :--- | :---: | :---: | :---: |",
            "| Terminal-Bench 4.0 (Pass@1) | 51.8 | 12.4 | 31.2 |",
            "| ProgramBench (Almost@1) | 37.0 | 15.5 | 20.3 |",
        )
        scores = fh.extract_scores_from_tables(
            fh.parse_markdown_tables(md), "deepseek-ai/DeepSeek-V4.1-Flash"
        )
        self.assertEqual(
            scores,
            {"Terminal-Bench 4.0 (Pass@1)": 31.2, "ProgramBench (Almost@1)": 20.3},
        )

    def test_a_first_letter_abbreviation_names_the_model(self):
        md = md_table(
            "| Benchmark | N-3-Ultra <br> 550B-A55B | Rival 1T |",
            "| :--- | :---: | :---: |",
            "| SWE-Bench Verified | 70.7 | 75.3 |",
        )
        scores = fh.extract_scores_from_tables(
            fh.parse_markdown_tables(md), "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
        )
        self.assertEqual(scores, {"SWE-Bench Verified": 70.7})

    def test_a_rounded_parameter_count_still_names_the_model(self):
        md = md_table(
            "| Category | Benchmarks | Solar Open (102B) | gpt-oss-120b (117B, high) |",
            "| :--- | :--- | :---: | :---: |",
            "| *General* | KMMLU | 73.0 | 72.7 |",
        )
        scores = fh.extract_scores_from_tables(
            fh.parse_markdown_tables(md), "upstage/Solar-Open-100B"
        )
        self.assertEqual(scores, {"KMMLU": 73.0})

    def test_a_different_parameter_count_is_a_different_model(self):
        md = md_table(
            "| Benchmark | Qwen3.5-4B | Qwen3.5-122B-A10B |",
            "| :--- | :---: | :---: |",
            "| MMLU-Pro | 60.1 | 84.9 |",
        )
        scores = fh.extract_scores_from_tables(
            fh.parse_markdown_tables(md), "Qwen/Qwen3.5-9B"
        )
        self.assertEqual(scores, {})

    def test_the_same_words_spaced_differently_still_match(self):
        # Olmo's table heads its released model "Olmo3 Instruct 7B" and the
        # checkpoints it was built from "Olmo 3 Instruct 7B SFT"/"DPO".
        md = md_table(
            "| Skill | Benchmark | Olmo 3 Instruct 7B SFT | Olmo 3 Instruct 7B DPO | Olmo3 Instruct 7B |",
            "| :--- | :--- | :---: | :---: | :---: |",
            "| Math | AIME 2024 | 6.7 | 30.1 | 44.3 |",
        )
        scores = fh.extract_scores_from_tables(
            fh.parse_markdown_tables(md), "allenai/Olmo-3-7B-Instruct"
        )
        self.assertEqual(scores, {"AIME 2024": 44.3})

    def test_a_base_column_loses_to_the_post_trained_one(self):
        md = md_table(
            "| Benchmark | Example-30B Base | Example-30B | Rival |",
            "| :--- | :---: | :---: | :---: |",
            "| MMLU-Pro | 73.2 | 84.9 | 80.1 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores, {"MMLU-Pro": 84.9})

    def test_a_model_built_from_this_one_is_not_this_one(self):
        # R1-0528's card carries a table of models distilled *from* it, and the
        # 8B distill's name contains the repo's whole name.
        md = md_table(
            "| | AIME 24 | GPQA Diamond |",
            "| :--- | :---: | :---: |",
            "| DeepSeek-R1-0528-Qwen3-8B | 86.0 | 61.1 |",
            "| Qwen3-8B | 76.3 | 62.0 |",
        )
        scores = fh.extract_scores_from_tables(
            fh.parse_markdown_tables(md), "deepseek-ai/DeepSeek-R1-0528"
        )
        self.assertEqual(scores, {})

    def test_a_release_date_tells_two_releases_apart(self):
        md = md_table(
            "| Model Name | Acc avg |",
            "| :--- | :---: |",
            "| Qwen3-235B-A22B (Thinking) | 70.6 |",
            "| Qwen3-235B-A22B-Thinking-2507 (Full Attention) | 82.9 |",
        )
        scores = fh.extract_scores_from_tables(
            fh.parse_markdown_tables(md), "Qwen/Qwen3-235B-A22B-Thinking-2507"
        )
        self.assertEqual(scores, {"Acc avg": 82.9})

    def test_the_row_the_llm_json_slug_asks_for_wins(self):
        # One repo serves both modes; the row being filled names which.
        md = md_table(
            "| Benchmark | Example-30B (Thinking) | Example-30B (Non-thinking) |",
            "| :--- | :---: | :---: |",
            "| AIME25 | 92.3 | 74.1 |",
        )
        tables = fh.parse_markdown_tables(md)
        self.assertEqual(
            fh.extract_scores_from_tables(tables, REPO, "example-30b-instruct"),
            {"AIME25": 74.1},
        )
        self.assertEqual(
            fh.extract_scores_from_tables(tables, REPO, "example-30b-reasoning"),
            {"AIME25": 92.3},
        )

    def test_two_variants_and_nothing_to_choose_between_them(self):
        # Neither the repo nor the row says which mode it wants, and picking
        # the leftmost is picking whichever the author typed first.
        md = md_table(
            "| Benchmark | Example-30B (Thinking) | Example-30B (Non-thinking) |",
            "| :--- | :---: | :---: |",
            "| AIME25 | 92.3 | 74.1 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores, {})

    def test_one_name_in_two_cells_is_not_a_tie(self):
        # A download table lists a model once per precision; the first is as
        # good as the last, and nothing is skipped over it.
        md = md_table(
            "| Model | Precision | MMLU-Pro |",
            "| :--- | :--- | :---: |",
            "| Example-30B | BF16 | 84.9 |",
            "| Example-30B | FP8 | 84.9 |",
        )
        self.assertEqual(fh.select_row(fh.parse_markdown_tables(md)[0], REPO), 0)

    def test_a_model_column_on_the_right_is_still_the_model_column(self):
        md = md_table(
            "| MMLU-Pro | AIME24 | Model |",
            "| :---: | :---: | :--- |",
            "| 84.9 | 88.1 | Example-30B |",
            "| 80.1 | 77.9 | Rival-27B |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores, {"MMLU-Pro": 84.9, "AIME24": 88.1})

    def test_a_mapped_heading_names_the_column_nothing_else_would(self):
        aliases = {"example-org/Example-30B": ["BF16"]}
        md = md_table(
            "| Benchmark | BF16 | FP8 | NVFP4 |",
            "| :--- | :---: | :---: | :---: |",
            "| MathVista_MINI | 71.90 | 71.05 | 71.30 |",
        )
        tables = fh.parse_markdown_tables(md)
        self.assertEqual(fh.extract_scores_from_tables(tables, REPO), {})
        with mock.patch.object(fh, "load_model_column_aliases", lambda *a, **k: aliases):
            self.assertEqual(
                fh.extract_scores_from_tables(tables, REPO), {"MathVista_MINI": 71.9}
            )


class TestLabelNormalisation(unittest.TestCase):
    def test_inline_markup_leaves_the_label(self):
        md = md_table(
            "| Benchmark | Example-30B |",
            "| :--- | :---: |",
            "| AIME24<sub>(Mean@32)</sub> | 93.3 |",
            "| Terminal-Bench<br>4.0 (Pass@1) | 31.2 |",
            "| MGSM&nbsp;(EM) | 80.2 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(
            scores,
            {
                "AIME24 (Mean@32)": 93.3,
                "Terminal-Bench 4.0 (Pass@1)": 31.2,
                "MGSM (EM)": 80.2,
            },
        )

    def test_a_nested_row_keeps_the_benchmark_it_hangs_off(self):
        md = md_table(
            "| Benchmark | Example-30B |",
            "| :--- | :---: |",
            "| TauBench V3 | |",
            "| &nbsp;&nbsp;Airline | 81.5 |",
            "| &nbsp;&nbsp;Telecom | 92.9 |",
            "| BrowseComp | 44.4 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(
            scores,
            {"TauBench V3 Airline": 81.5, "TauBench V3 Telecom": 92.9, "BrowseComp": 44.4},
        )

    def test_a_row_in_bold_is_not_a_nested_row(self):
        md = md_table(
            "| Benchmark | Example-30B |",
            "| :--- | :---: |",
            "| **Chat** | |",
            "| AlpacaEval 2 LC | 69.1 |",
            "| **Safety** | 64.8 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores, {"AlpacaEval 2 LC": 69.1, "Safety": 64.8})

    def test_a_lower_is_better_label_is_not_a_benchmark(self):
        md = md_table(
            "| Benchmark | Example-30B |",
            "| :--- | :---: |",
            "| HF-ASR (WER\u2193) | 3.11 |",
            "| MMLU-Pro | 84.9 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores, {"MMLU-Pro": 84.9})


class TestValueReading(unittest.TestCase):
    def test_a_metric_name_in_the_cell_is_not_the_value(self):
        # Qwen prints "Pass@1 20.4 Score 42.9" in one cell; the "1" of "Pass@1"
        # used to land as the model's Agent's Last Exam score.
        self.assertEqual(fh.parse_score("Pass@1 20.4 Score 42.9"), 20.4)
        self.assertEqual(fh.parse_score("Without CI 83.7 With CI 90.2"), 83.7)

    def test_a_name_with_a_number_in_it_is_not_a_value(self):
        self.assertIsNone(fh.parse_score("NVIDIA-Nemotron-3-Nano-Omni-30B"))
        self.assertIsNone(fh.parse_score("GLM-4.5-Air"))
        self.assertIsNone(fh.parse_score("8B / 16B"))

    def test_a_rank_is_not_a_score(self):
        self.assertIsNone(fh.parse_score("1st"))
        self.assertIsNone(fh.parse_score("3rd"))

    def test_the_value_after_a_label_is_the_value(self):
        self.assertEqual(fh.parse_score("10/50 steps: 40.1"), 40.1)

    def test_two_qualified_runs_in_one_cell_name_neither(self):
        self.assertIsNone(fh.parse_score("43.2 (no tools) / 57.4 (with tools)"))

    def test_a_footnoted_alternative_keeps_the_leading_value(self):
        self.assertEqual(fh.parse_score("36.8 (39.1\u2020)"), 36.8)

    def test_a_fraction_scaled_table_is_put_back_on_scale(self):
        # Mistral's Ministral cards report 0-1 where every other card reports
        # 0-100, which stored as-is lands as a sub-1% score.
        md = md_table(
            "| Benchmark | Example-30B |",
            "| :--- | :---: |",
            "| MMLU 5-shot | 0.794 |",
            "| AIME24 | 0.898 |",
            "| GPQA Diamond | 0.712 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(
            scores, {"MMLU 5-shot": 79.4, "AIME24": 89.8, "GPQA Diamond": 71.2}
        )

    def test_a_genuinely_near_zero_column_is_left_alone(self):
        # ZeroBench's whole published field is 0.0 to 12.0.
        md = md_table(
            "| Benchmark | Example-30B |",
            "| :--- | :---: |",
            "| ZeroBench | 0.4 |",
            "| MMLU-Pro | 84.9 |",
            "| AIME24 | 88.1 |",
        )
        scores = fh.extract_scores_from_tables(fh.parse_markdown_tables(md), REPO)
        self.assertEqual(scores["ZeroBench"], 0.4)


class TestTablesWithoutTrailingPipes(unittest.TestCase):
    def test_a_row_may_end_without_its_closing_pipe(self):
        # DeepSeek-R1-0528's comparison table is written this way, and
        # requiring the trailing pipe dropped every benchmark on the card.
        md = (
            "| Category | Benchmark (Metric) | DeepSeek R1 | DeepSeek R1 0528\n"
            "|----------|----------|-----------------|---|\n"
            "| General  |\n"
            "|          | MMLU-Pro (EM) | 84.0 | 85.0\n"
            "|          | GPQA-Diamond (Pass@1) | 71.5 | 81.0\n"
        )
        scores = fh.extract_scores_from_tables(
            fh.parse_markdown_tables(md), "deepseek-ai/DeepSeek-R1-0528"
        )
        self.assertEqual(
            scores, {"MMLU-Pro (EM)": 85.0, "GPQA-Diamond (Pass@1)": 81.0}
        )


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

    def test_a_plain_entry_beats_the_qualified_ones(self):
        payload = {
            "evalResults": [
                {"data": {"dataset": {"id": "a/b", "task_id": "b"},
                          "value": 73.5, "notes": "Reasoning: medium"}},
                {"data": {"dataset": {"id": "a/b", "task_id": "b"},
                          "value": 80.1, "notes": "Reasoning: high"}},
                {"data": {"dataset": {"id": "a/b", "task_id": "b"},
                          "value": 80.8, "notes": "GPQA Diamond"}},
            ]
        }
        self.assertEqual(fh.extract_eval_results(payload), {"a/b": 80.8})

    def test_qualified_runs_that_disagree_name_no_value(self):
        # NVIDIA files SWE-bench Verified three times, once per harness, and
        # nothing on the page says which of the three is the benchmark's.
        payload = {
            "evalResults": [
                {"data": {"dataset": {"id": "SWE-bench/SWE-bench_Verified", "task_id": "resolved"},
                          "value": 60.47, "notes": "OpenHands harness"}},
                {"data": {"dataset": {"id": "SWE-bench/SWE-bench_Verified", "task_id": "resolved"},
                          "value": 59.2, "notes": "OpenCode harness"}},
                {"data": {"dataset": {"id": "SWE-bench/SWE-bench_Verified", "task_id": "resolved"},
                          "value": 53.73, "notes": "Codex harness"}},
            ]
        }
        self.assertEqual(fh.extract_eval_results(payload), {})

    def test_one_harness_note_on_its_own_is_still_the_card_s_number(self):
        payload = {
            "evalResults": [
                {"data": {"dataset": {"id": "a/b", "task_id": "b"},
                          "value": 86.1, "notes": "Terminus-2 harness, avg of 5 runs."}},
            ]
        }
        self.assertEqual(fh.extract_eval_results(payload), {"a/b": 86.1})

    def test_a_merged_entry_beats_one_pending_on_a_pull_request(self):
        payload = {
            "evalResults": [
                {"pullRequest": 24, "data": {"dataset": {"id": "a/b", "task_id": "b"},
                                             "value": 84.8, "date": "2026-01-01"}},
                {"data": {"dataset": {"id": "a/b", "task_id": "b"},
                          "value": 86.6, "date": "2026-01-01"}},
            ]
        }
        self.assertEqual(fh.extract_eval_results(payload), {"a/b": 86.6})

    def test_a_board_filed_as_fractions_is_put_back_on_scale(self):
        payload = {
            "evalResults": [
                {"data": {"dataset": {"id": "board/results", "task_id": "overall"}, "value": 0.3026}},
                {"data": {"dataset": {"id": "board/results", "task_id": "swebench"}, "value": 0.5204}},
                {"data": {"dataset": {"id": "board/results", "task_id": "appworld"}, "value": 0.08}},
                {"data": {"dataset": {"id": "other/bench", "task_id": "chart"}, "value": 0.4}},
            ]
        }
        self.assertEqual(
            fh.extract_eval_results(payload),
            {
                "board/results (overall)": 30.26,
                "board/results (swebench)": 52.04,
                "board/results (appworld)": 8.0,
                # One near-zero value on its own is a small model, not a scale.
                "other/bench (chart)": 0.4,
            },
        )



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
