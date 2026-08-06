#!/usr/bin/env python3
"""Tests for the Datacurve DeepSWE leaderboard reader. Run with ./test_datacurve.py

The board publishes one row per configuration (harness + model + reasoning
effort) inside a versioned JSON artifact whose path the page carries. These tests
pin the parts that decide what reaches llm.json: the configuration label (which
has to match benchlm.ai's spelling, since both DeepSWE sources share one name
mapping), the best-configuration-per-model reduction, the metric, and the
artifact discovery that keeps a version bump from breaking the scraper.
"""

from __future__ import annotations

import unittest
import urllib.error
from unittest import mock

import fetch_datacurve as dc


def row(model: str, effort: str | None, pass_at_1: float | None, **extra) -> dict:
    return {
        "model": model,
        "harness": "mini-swe-agent",
        "reasoning_effort": effort,
        "pass_at_1": pass_at_1,
        "pass_at_4": None if pass_at_1 is None else pass_at_1 + 0.2,
        "n_runs": 4,
        **extra,
    }


PAYLOAD = {
    "n_tasks_in_set": 113,
    "generated_at": "2026-08-06T05:49:48+00:00",
    "rows": [
        row("glm-5-2", "max", 0.4378),
        row("glm-5-2", "high", 0.3628),
        row("kimi-k2-7-code", None, 0.3053),
        row("broken", "max", None),
    ],
}


class TestConfigLabel(unittest.TestCase):
    def test_effort_is_bracketed_like_benchlm_spells_it(self) -> None:
        self.assertEqual(dc.config_label(row("glm-5-2", "max", 0.4)), "glm-5-2[max]")

    def test_model_without_an_effort_keeps_its_bare_name(self) -> None:
        self.assertEqual(dc.config_label(row("kimi-k2-7-code", None, 0.3)), "kimi-k2-7-code")
        self.assertEqual(dc.config_label(row("kimi-k2-7-code", "", 0.3)), "kimi-k2-7-code")

    def test_row_without_a_model_has_no_label(self) -> None:
        self.assertIsNone(dc.config_label({"reasoning_effort": "max"}))


class TestParseRows(unittest.TestCase):
    def scores(self, **kwargs) -> dict[str, dict]:
        return {r["model"]: r for r in dc.parse_rows(PAYLOAD, **kwargs)}

    def test_best_configuration_per_model_by_default(self) -> None:
        rows = self.scores(metric="pass@1", all_configs=False)
        self.assertEqual(set(rows), {"glm-5-2[max]", "kimi-k2-7-code"})
        self.assertEqual(rows["glm-5-2[max]"]["score"], 43.78)

    def test_all_configs_keeps_every_row(self) -> None:
        rows = self.scores(metric="pass@1", all_configs=True)
        self.assertEqual(
            set(rows), {"glm-5-2[max]", "glm-5-2[high]", "kimi-k2-7-code"}
        )

    def test_score_is_the_metric_as_a_percentage(self) -> None:
        rows = self.scores(metric="pass@4", all_configs=False)
        self.assertEqual(rows["glm-5-2[max]"]["score"], 63.78)

    def test_row_without_the_metric_is_dropped(self) -> None:
        self.assertNotIn("broken[max]", self.scores(metric="pass@1", all_configs=True))

    def test_rows_are_ranked_best_first(self) -> None:
        rows = dc.parse_rows(PAYLOAD, metric="pass@1", all_configs=True)
        self.assertEqual([r["rank"] for r in rows], [1, 2, 3])
        self.assertEqual(rows[0]["model"], "glm-5-2[max]")

    def test_base_model_is_kept_for_the_reduction(self) -> None:
        rows = self.scores(metric="pass@1", all_configs=True)
        self.assertEqual(rows["glm-5-2[high]"]["base_model"], "glm-5-2")

    def test_payload_without_rows_raises(self) -> None:
        with self.assertRaises(ValueError):
            dc.parse_rows({}, metric="pass@1", all_configs=False)


class TestArtifactDiscovery(unittest.TestCase):
    def test_path_is_read_off_the_page(self) -> None:
        html = 'x ["artifact","/artifacts/v1.1/leaderboard-live.json"] y'
        with mock.patch.object(dc, "fetch_text", return_value=html):
            self.assertEqual(
                dc.discover_artifact_url(),
                "https://deepswe.datacurve.ai/artifacts/v1.1/leaderboard-live.json",
            )

    def test_highest_version_wins_when_several_appear(self) -> None:
        html = (
            '"/artifacts/v1.1/leaderboard-live.json" '
            '"/artifacts/v1.10/leaderboard-live.json" '
            '"/artifacts/v1.2/leaderboard-live.json"'
        )
        with mock.patch.object(dc, "fetch_text", return_value=html):
            self.assertTrue(dc.discover_artifact_url().endswith("v1.10/leaderboard-live.json"))

    def test_page_without_a_path_falls_back_to_the_pinned_one(self) -> None:
        with mock.patch.object(dc, "fetch_text", return_value="<html>no data</html>"):
            self.assertEqual(dc.discover_artifact_url(), dc.URL)

    def test_unreachable_page_falls_back_instead_of_raising(self) -> None:
        with mock.patch.object(dc, "fetch_text", side_effect=urllib.error.URLError("down")):
            self.assertEqual(dc.discover_artifact_url(), dc.URL)

    def test_pinned_url_points_at_the_site(self) -> None:
        self.assertTrue(dc.URL.startswith(dc.SITE_URL))
        self.assertTrue(dc.URL.endswith("leaderboard-live.json"))


class TestGetScores(unittest.TestCase):
    def test_explicit_artifact_url_skips_discovery(self) -> None:
        import json as _json

        with mock.patch.object(dc, "fetch_text", return_value=_json.dumps(PAYLOAD)) as fetch:
            with mock.patch.object(dc, "discover_artifact_url") as discover:
                rows = dc.get_scores(artifact_url="https://example.com/a.json")
        discover.assert_not_called()
        fetch.assert_called_once_with("https://example.com/a.json")
        self.assertEqual(rows[0]["model"], "glm-5-2[max]")

    def test_non_object_payload_raises(self) -> None:
        with mock.patch.object(dc, "fetch_text", return_value="[]"):
            with self.assertRaises(ValueError):
                dc.get_scores(artifact_url="https://example.com/a.json")


if __name__ == "__main__":
    unittest.main()
