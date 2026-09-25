#!/usr/bin/env python3
"""Tests for the OSWorld leaderboard reader. Run with ./test_osworld.py

fetch_osworld.py reads two boards: the OSWorld-Verified workbook and the
OSWorld 2.0 site's results file. The 2.0 file carries every run the board can
show -- three task releases, three step budgets, a full and an offline set --
and only one cell of that grid per release is a column here. So the
load-bearing tests are the filters that pick that cell, the release routing
(an untracked release is skipped, never folded into a neighbour's column), and
the payload guards that refuse a file which changed what its rows mean.
"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import fetch_osworld as osw
import update


def result(model: str, score: float, **overrides) -> dict:
    row = {
        "model": model,
        "modelFamily": "Example",
        "reasoning": "max",
        "toolSetting": "standard",
        "stepBudget": 500,
        "binaryAccuracy": score,
        "partialScore": score * 2,
        "estimatedCostUsd": None,
        "official": True,
    }
    row.update(overrides)
    return row


def payload(*results: dict, **overrides) -> dict:
    doc = {
        "benchmarkVersion": "OSWorld 2.0",
        "taskVersion": "v2026.06.24",
        "releaseVersions": ["v2026.06.24", "v2026.08.08", "v2.1"],
        "defaultResultReleaseVersion": "v2026.06.24",
        "defaultResultDatasetScope": "full",
        "datasetSize": 108,
        "metrics": {"binaryAccuracy": "Binary Accuracy", "partialScore": "Partial Score"},
        "results": list(results),
    }
    doc.update(overrides)
    return doc


def parse(doc: dict) -> list[dict]:
    with redirect_stderr(io.StringIO()):
        return osw.parse_v2(doc)


class TestV2Filters(unittest.TestCase):
    def test_a_row_naming_no_release_is_the_papers_release(self) -> None:
        rows = parse(payload(result("Alpha", 20.6)))
        self.assertEqual(
            [(r["benchmark"], r["model"], r["score"], r["release"]) for r in rows],
            [("osworld_2_0_2026_06_24", "Alpha", 20.6, "v2026.06.24")],
        )

    def test_each_tracked_release_feeds_its_own_column(self) -> None:
        releases = {"v2026.06.24": "osworld_2_0_2026_06_24", "v9.9": "osworld_9_9"}
        with mock.patch.object(osw, "RELEASES", releases):
            rows = parse(payload(
                result("Alpha", 20.6),
                result("Alpha", 44.33, releaseVersion="v9.9"),
            ))
        self.assertEqual(
            {r["benchmark"]: r["score"] for r in rows},
            {"osworld_2_0_2026_06_24": 20.6, "osworld_9_9": 44.33},
        )

    def test_untracked_releases_are_skipped_and_reported_with_their_field(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            rows = osw.parse_v2(payload(
                result("Alpha", 20.6),
                result("Alpha", 31.43, releaseVersion="v2026.08.08"),
                result("Alpha", 44.33, releaseVersion="v2.1"),
                result("Beta", 45.0, releaseVersion="v2.1"),
            ))
        self.assertEqual([r["release"] for r in rows], ["v2026.06.24", "v2026.08.08"])
        report = err.getvalue()
        self.assertNotIn("v2026.08.08", report)
        self.assertIn("'v2.1', which has no llm.json column: 2 model(s)", report)

    def test_the_two_tracked_releases_feed_their_dated_columns(self) -> None:
        rows = parse(payload(
            result("Alpha", 20.6),
            result("Alpha", 31.43, releaseVersion="v2026.08.08"),
        ))
        self.assertEqual(
            [(r["benchmark"], r["score"]) for r in rows],
            [("osworld_2_0_2026_06_24", 20.6), ("osworld_2_0_2026_08_08", 31.43)],
        )

    def test_only_the_500_step_budget_is_read(self) -> None:
        rows = parse(payload(
            result("Alpha", 4.6, stepBudget=150),
            result("Alpha", 13.0, stepBudget=300),
            result("Alpha", 13.9),
        ))
        self.assertEqual([r["score"] for r in rows], [13.9])

    def test_the_offline_set_is_not_the_full_set(self) -> None:
        rows = parse(payload(
            result("Alpha", 22.1, datasetScope="offline"),
            result("Alpha", 20.6, datasetScope="full"),
        ))
        self.assertEqual([r["score"] for r in rows], [20.6])

    def test_a_row_listed_under_both_scopes_is_read(self) -> None:
        rows = parse(payload(
            result("Alpha", 20.0, datasetScope="offline", availableScopes=["offline", "full"]),
        ))
        self.assertEqual([r["score"] for r in rows], [20.0])

    def test_unofficial_rows_are_not_read(self) -> None:
        rows = parse(payload(
            result("Alpha", 90.0, official=False),
            result("Beta", 10.0),
        ))
        self.assertEqual([r["model"] for r in rows], ["Beta"])

    def test_the_score_is_binary_accuracy_not_the_partial_score(self) -> None:
        (row,) = parse(payload(result("Alpha", 18.2, partialScore=48.91)))
        self.assertEqual(row["score"], 18.2)
        self.assertEqual(row["partial_score"], 48.91)

    def test_every_effort_and_tool_setting_is_its_own_row(self) -> None:
        # The ingest keeps the best of them; the reader must not pre-empt that.
        rows = parse(payload(
            result("Alpha", 20.6, toolSetting="batched tool"),
            result("Alpha", 18.52, toolSetting="standard"),
        ))
        self.assertEqual(sorted(r["score"] for r in rows), [18.52, 20.6])


class TestV2Guards(unittest.TestCase):
    def assert_refused(self, doc: dict) -> None:
        with self.assertRaises(ValueError):
            parse(doc)

    def test_another_benchmarks_file_is_refused(self) -> None:
        self.assert_refused(payload(result("Alpha", 20.0), benchmarkVersion="OSWorld 3.0"))

    def test_a_changed_task_count_is_refused(self) -> None:
        self.assert_refused(payload(result("Alpha", 20.0), datasetSize=82))

    def test_a_renamed_metric_is_refused(self) -> None:
        self.assert_refused(payload(result("Alpha", 20.0), metrics={"successRate": "Success"}))

    def test_a_row_missing_a_field_the_filters_need_is_refused(self) -> None:
        row = result("Alpha", 20.0)
        del row["stepBudget"]
        self.assert_refused(payload(row))

    def test_a_board_with_nothing_readable_is_refused(self) -> None:
        self.assert_refused(payload(result("Alpha", 20.0, stepBudget=150)))

    def test_fractions_are_refused(self) -> None:
        self.assert_refused(payload(*(result(f"M{i}", 0.1 * i) for i in range(1, 7))))

    def test_no_results_list_is_refused(self) -> None:
        doc = payload()
        del doc["results"]
        self.assert_refused(doc)


class TestVerifiedRows(unittest.TestCase):
    def test_verified_rows_name_their_column(self) -> None:
        rows = [
            {"model": "Alpha", "institution": None, "approach_type": "General model",
             "max_steps": 100, "a11y": "No", "coding": "No", "rollout": "No",
             "date": 46000, "score": 70.0},
            {"model": "Alpha", "institution": None, "approach_type": "General model",
             "max_steps": 100, "a11y": "No", "coding": "No", "rollout": "No",
             "date": 46001, "score": 72.0},
        ]
        (row,) = osw.aggregate(rows, foundation_only=True)
        self.assertEqual((row["benchmark"], row["score"], row["runs"]), ("osworld_verified", 71.0, 2))


class TestIngest(unittest.TestCase):
    def test_scores_are_credited_to_their_boards_site(self) -> None:
        model = {"name": "alpha", "scores": {}, "scores_updated": {}, "scores_source": {}}
        doc = {"benchmarks": {k: {} for k in osw.KEYS}, "models": [model]}
        by_key = {k: {"alpha": {"score": 10.0 + i}} for i, k in enumerate(osw.KEYS)}
        matched, updated, _ = update.update_osworld_scores(doc, by_key)
        self.assertEqual((matched, updated), (1, len(osw.KEYS)))
        self.assertEqual(model["scores_source"]["osworld_verified"], "https://os-world.github.io")
        self.assertEqual(model["scores_source"]["osworld_2_0_2026_06_24"], "https://osworld-v2.xlang.ai")

    def test_every_column_the_reader_can_fill_exists_in_llm_json(self) -> None:
        benchmarks = json.loads(
            Path(__file__).resolve().with_name("llm.json").read_text(encoding="utf-8")
        )["benchmarks"]
        for key in osw.KEYS:
            with self.subTest(key=key):
                self.assertIn(key, benchmarks)

    def test_rows_for_an_unknown_column_are_ignored(self) -> None:
        rows = [
            {"benchmark": "osworld_2_0_2026_06_24", "model": "Alpha", "score": 4.6},
            {"benchmark": "osworld_9_9", "model": "Alpha", "score": 99.0},
        ]
        proc = mock.Mock(returncode=0, stdout=json.dumps(rows), stderr="")
        with mock.patch.object(update, "run_fetch", return_value=proc), \
                mock.patch.object(update, "load_osworld_to_slug_mapping", return_value={"Alpha": "alpha"}):
            by_key = update.fetch_osworld_data(update.OSWORLD_SCRIPT, mock.Mock())
        self.assertEqual(by_key, {"osworld_2_0_2026_06_24": {"alpha": rows[0]}})


if __name__ == "__main__":
    unittest.main()
