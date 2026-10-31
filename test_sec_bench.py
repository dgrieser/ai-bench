#!/usr/bin/env python3
"""Tests for the SEC-bench Pro reader. Run with ./test_sec_bench.py

The site's data file carries every snapshot of the benchmark, and the newest
one is a different, larger task set that scores the same model up to twelve
points higher. The column is pinned to snapshot 260505, so what these tests pin
is that the reader takes that snapshot's overall board and its headline score
only, and that a missing snapshot or a changed task count stops the read
instead of filing another task set under the same column.
"""

from __future__ import annotations

import copy
import unittest

import fetch_sec_bench as sb


def result(model: str, headline: float, completed: float) -> dict:
    return {
        "agent": "OpenCode",
        "model": model,
        "effort": "high",
        "backend": "AWS Bedrock",
        "timeouts": 3,
        "date": "2026-06-17",
        "score_modes": {
            "headline": {"solved": 4, "total": 183, "score": headline},
            "completed": {"solved": 4, "total": 180, "score": completed},
        },
    }


PAYLOAD = {
    "default_version": "260617",
    "snapshots": {
        "260505": {
            "leaderboards": [
                {"name": "v8", "instances": 103, "results": [result("Kimi K2.5", 1.9, 2.0)]},
                {
                    "name": "overall",
                    "instances": 183,
                    "results": [result("Kimi K2.5", 2.2, 2.3), result("Opus 4.6", 25.7, 60.9)],
                },
            ]
        },
        "260617": {
            "leaderboards": [
                {"name": "overall", "instances": 344, "results": [result("Kimi K2.5", 2.3, 2.3)]}
            ]
        },
    },
}


class ReadsTheVendorSnapshot(unittest.TestCase):
    def test_overall_headline_of_260505(self):
        rows = sb.get_scores(fetch=lambda url: PAYLOAD)
        self.assertEqual(
            {r["model"]: r["score"] for r in rows if r["snapshot"] == "260505"},
            {"Kimi K2.5": 2.2, "Opus 4.6": 25.7},
        )
        self.assertEqual({r["key"] for r in rows if r["snapshot"] == "260505"}, {"sec_bench_pro"})

    def test_260617_is_its_own_column(self):
        rows = [r for r in sb.get_scores(fetch=lambda url: PAYLOAD) if r["snapshot"] == "260617"]
        self.assertEqual([(r["key"], r["model"], r["score"], r["rank"]) for r in rows],
                         [("sec_bench_pro_260617", "Kimi K2.5", 2.3, 1)])
        self.assertEqual(sb.source_url("sec_bench_pro_260617"), "https://sec-bench.github.io/")

    def test_a_snapshot_without_a_column_is_reported_not_read(self):
        payload = copy.deepcopy(PAYLOAD)
        payload["snapshots"]["260901"] = copy.deepcopy(payload["snapshots"]["260617"])
        self.assertEqual(sb.unread_snapshots(payload), ["260901"])
        rows = sb.get_scores(fetch=lambda url: payload)
        self.assertNotIn("260901", {r["snapshot"] for r in rows})

    def test_the_completed_only_rate_never_lands(self):
        rows = sb.get_scores(fetch=lambda url: PAYLOAD)
        self.assertNotIn(60.9, [r["score"] for r in rows])

    def test_a_missing_snapshot_stops_the_read(self):
        payload = copy.deepcopy(PAYLOAD)
        del payload["snapshots"]["260505"]
        with self.assertRaises(ValueError):
            sb.get_scores(fetch=lambda url: payload)

    def test_a_changed_task_count_stops_the_read(self):
        payload = copy.deepcopy(PAYLOAD)
        payload["snapshots"]["260505"]["leaderboards"][1]["instances"] = 344
        with self.assertRaises(ValueError):
            sb.get_scores(fetch=lambda url: payload)

    def test_credited_to_the_snapshot_page(self):
        self.assertEqual(sb.URL, "https://sec-bench.github.io/260505/")


if __name__ == "__main__":
    unittest.main()
