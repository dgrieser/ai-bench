#!/usr/bin/env python3
"""Tests for the Agents' Last Exam reader. Run with ./test_agents_last_exam.py

ALE publishes the same runs under a dozen splits whose pass rates differ by an
order of magnitude -- full/overall sits near 30%, full/last-exam near 3% -- so
reading the wrong one is indistinguishable from every model collapsing. These
tests pin the split the reader selects, the metric it takes out of it, and that
a missing split is an error rather than a fallback.
"""

from __future__ import annotations

import unittest
from unittest import mock

import fetch_agents_last_exam as ale


def row(model: str, split: str, pass_rate: float, avg_score: float, harness="claude_code",
        variant="thinking-max") -> dict:
    return {
        "split": split,
        "harness": harness,
        "model": model,
        "harnessVariant": variant,
        "runs": 150,
        "tasks": 150,
        "splitTasks": 152,
        "passes": round(pass_rate * 150),
        "passRate": pass_rate,
        "avgScore": avg_score,
        "totalCostUsd": 1234.5,
        "costSource": "official",
    }


PAYLOAD = {
    "rows": [
        row("alpha", "full/overall", 0.3158, 0.552),
        row("alpha", "full/overall", 0.2763, 0.519, variant="thinking-low"),
        row("beta", "full/overall", 0.2697, 0.5122, harness="grok_build"),
        # Other tiers of the same runs, on entirely different scales.
        row("alpha", "full/last-exam", 0.029, 0.108),
        row("beta", "full/last-exam", 0.0, 0.104),
        row("alpha", "linux_only", 0.19, 0.422),
        row("alpha", "unlicensed/overall", 0.31, 0.55),
    ]
}


def scores(payload=PAYLOAD, **kwargs) -> list[dict]:
    with mock.patch.object(ale, "fetch_json", return_value=payload):
        return ale.get_scores(**kwargs)


class TestSelectSplit(unittest.TestCase):
    def test_default_split_is_the_sites_default_view(self):
        self.assertEqual(ale.SPLIT, "full/overall")

    def test_reads_only_the_selected_split(self):
        self.assertEqual(len(ale.select_split(PAYLOAD, "full/overall")), 3)

    def test_another_tier_is_reachable_but_not_the_default(self):
        self.assertEqual(len(ale.select_split(PAYLOAD, "full/last-exam")), 2)

    def test_missing_split_names_what_is_there(self):
        with self.assertRaises(ValueError) as ctx:
            ale.select_split(PAYLOAD, "full/near-term")
        self.assertIn("'full/near-term'", str(ctx.exception))
        self.assertIn("full/overall", str(ctx.exception))

    def test_a_response_without_rows_is_an_error(self):
        with self.assertRaises(ValueError):
            ale.select_split({"data": []}, "full/overall")


class TestGetScores(unittest.TestCase):
    def test_pass_rate_is_the_stored_score_as_a_percentage(self):
        self.assertEqual(scores()[0]["score"], 31.58)

    def test_partial_credit_score_is_carried_but_is_not_the_score(self):
        top = scores()[0]
        self.assertEqual(top["avg_score"], 55.2)
        self.assertNotEqual(top["avg_score"], top["score"])

    def test_the_last_exam_tier_never_leaks_into_the_default_view(self):
        self.assertNotIn(2.9, [e["score"] for e in scores()])
        self.assertEqual(len(scores()), 3)

    def test_every_harness_variant_is_kept(self):
        # update.py folds them onto one slug; the reader must not pre-empt that.
        alpha = [e for e in scores() if e["model"] == "alpha"]
        self.assertEqual(
            sorted(e["harness_variant"] for e in alpha), ["thinking-low", "thinking-max"]
        )

    def test_rows_are_ranked_best_first(self):
        got = scores()
        self.assertEqual([e["rank"] for e in got], [1, 2, 3])
        self.assertEqual([e["score"] for e in got], sorted((e["score"] for e in got), reverse=True))

    def test_an_explicit_split_is_read_on_its_own_scale(self):
        got = scores(split="full/last-exam")
        self.assertEqual([e["score"] for e in got], [2.9, 0.0])

    def test_a_row_without_a_pass_rate_is_dropped(self):
        broken = row("gamma", "full/overall", 0.1, 0.2)
        broken.pop("passRate")
        self.assertEqual(len(scores({"rows": [*PAYLOAD["rows"], broken]})), 3)

    def test_a_boolean_is_not_a_pass_rate(self):
        broken = row("gamma", "full/overall", 0.1, 0.2)
        broken["passRate"] = True
        self.assertEqual(len(scores({"rows": [*PAYLOAD["rows"], broken]})), 3)

    def test_a_row_without_a_model_is_dropped(self):
        broken = row("", "full/overall", 0.1, 0.2)
        self.assertEqual(len(scores({"rows": [*PAYLOAD["rows"], broken]})), 3)

    def test_a_missing_avg_score_is_reported_as_none(self):
        broken = row("gamma", "full/overall", 0.1, 0.2)
        broken["avgScore"] = None
        got = scores({"rows": [*PAYLOAD["rows"], broken]})
        self.assertIsNone(next(e for e in got if e["model"] == "gamma")["avg_score"])


if __name__ == "__main__":
    unittest.main()
