#!/usr/bin/env python3
"""Tests for the OpenRouter model-page reader. Run with ./test_openrouter.py

An OpenRouter page carries one GPQA Diamond run per provider endpoint, plus a
run through OpenRouter's own router and, now and then, an endpoint that
scored 0 because it failed rather than because it answered. The page also
carries tau2-bench airline and a mirror of Artificial Analysis' numbers,
neither of which is read.

What these tests pin is what could silently move the number: that the query
is found by its key rather than by position, that only gpqa_diamond endpoint
rows count, that the router row and a 0 are left out and one endpoint counts
once, that the score is the median of what is left, and that a page without
the query stops the read instead of reporting a model without scores.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import fetch_openrouter as fo
import update


def score_row(provider: str, score: float, endpoint: str | None, benchmark="gpqa_diamond"):
    return {
        "provider_name": provider,
        "benchmark_type": benchmark,
        "score": score,
        "run_count": 1,
        "endpoint_id": endpoint,
    }


def page(rows: list[dict] | None, permaslug: str = "x/model-20260101") -> str:
    """A model page whose flight payload holds the given benchmarkScores rows."""
    queries = [
        {
            "dehydratedAt": 1,
            "state": {"data": {"scores": []}},
            "queryKey": ["model-page", "appStats", {"permaslug": permaslug}],
        },
    ]
    if rows is not None:
        queries.append(
            {
                "dehydratedAt": 1,
                "state": {"data": {"scores": rows}},
                "queryKey": ["model-page", "benchmarkScores", {"permaslug": permaslug}],
            }
        )
    payload = json.dumps({"queries": queries})
    # The flight payload is split across pushes at arbitrary points.
    half = len(payload) // 2
    pushes = "".join(
        f"<script>self.__next_f.push([1,{json.dumps(chunk)}])</script>"
        for chunk in (payload[:half], payload[half:])
    )
    return f"<html><body>{pushes}</body></html>"


class TestParse(unittest.TestCase):
    def test_median_of_the_endpoint_runs(self) -> None:
        row = fo.parse_model(
            page(
                [
                    score_row("A", 0.80, "e1"),
                    score_row("B", 0.90, "e2"),
                    score_row("C", 0.85, "e3"),
                ]
            ),
            "x/model",
        )
        self.assertEqual(row["score"], 85.0)
        self.assertEqual(row["runs"], 3)
        self.assertEqual((row["low"], row["high"]), (80.0, 90.0))
        self.assertEqual(row["source"], "https://openrouter.ai/x/model")
        self.assertEqual(row["permaslug"], "x/model-20260101")

    def test_router_row_and_failed_runs_are_left_out(self) -> None:
        row = fo.parse_model(
            page(
                [
                    score_row("auto-routing", 0.99, None),
                    score_row("Broken", 0, "e0"),
                    score_row("A", 0.80, "e1"),
                    score_row("B", 0.90, "e2"),
                ]
            ),
            "x/model",
        )
        self.assertEqual(row["runs"], 2)
        self.assertEqual(row["score"], 85.0)

    def test_other_benchmarks_are_not_read(self) -> None:
        row = fo.parse_model(
            page(
                [
                    score_row("A", 0.70, "e1", benchmark="tau_bench_verified_airline"),
                    score_row("A", 0.80, "e1"),
                ]
            ),
            "x/model",
        )
        self.assertEqual(row["score"], 80.0)
        self.assertEqual(row["runs"], 1)

    def test_one_endpoint_counts_once(self) -> None:
        row = fo.parse_model(
            page([score_row("A", 0.80, "e1"), score_row("A", 0.80, "e1"), score_row("B", 0.9, "e2")]),
            "x/model",
        )
        self.assertEqual(row["runs"], 2)

    def test_a_page_with_no_runs_reports_no_score(self) -> None:
        row = fo.parse_model(page([]), "x/model")
        self.assertIsNone(row["score"])
        self.assertEqual(row["runs"], 0)

    def test_a_page_without_the_query_stops_the_read(self) -> None:
        with self.assertRaisesRegex(ValueError, "benchmarkScores"):
            fo.parse_model(page(None), "x/model")

    def test_a_percentage_scale_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "fraction"):
            fo.parse_model(page([score_row("A", 85.0, "e1")]), "x/model")

    def test_variants_fold_onto_the_plain_id(self) -> None:
        self.assertEqual(fo.base_id("dots-studio/dots-3-note-preview:free"), "dots-studio/dots-3-note-preview")
        self.assertEqual(fo.base_id("openai/gpt-6-sol"), "openai/gpt-6-sol")

    def test_one_failed_page_is_that_models_every_failed_page_is_the_site(self) -> None:
        pages = {
            "https://openrouter.ai/x/good": page([score_row("A", 0.8, "e1")]),
            "https://openrouter.ai/x/bad": "<html></html>",
        }
        with mock.patch.object(fo, "fetch_text", side_effect=pages.__getitem__):
            rows = fo.get_scores(["x/good", "x/bad", "x/good:free"])
        self.assertEqual([r["model"] for r in rows], ["x/good"])
        with mock.patch.object(fo, "fetch_text", return_value="<html></html>"):
            with self.assertRaises(ValueError):
                fo.get_scores(["x/bad"])


class TestIngest(unittest.TestCase):
    def mapping(self, payload: dict) -> Path:
        path = Path(tempfile.mkdtemp()) / "mapping.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_only_mapped_ids_are_fetched(self) -> None:
        path = self.mapping({"x/a": "a", "x/b": "__unmappable__", "x/c": "__closed_weights__"})
        proc = SimpleNamespace(returncode=0, stdout="[]", stderr="")
        with mock.patch.object(update, "run_fetch", return_value=proc) as run:
            update.fetch_openrouter_data(Path("fetch_openrouter.py"), path)
        cmd = run.call_args.args[0]
        self.assertEqual([cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--model"], ["x/a"])

    def test_scores_land_on_gpqa_with_the_model_page(self) -> None:
        doc = {
            "benchmarks": {},
            "models": [{"name": "a", "scores": {"gpqa_diamond": None}}],
        }
        by_slug = {"a": {"score": 89.56, "source": "https://openrouter.ai/x/a"}}
        matched, updated, changes = update.update_openrouter_scores(doc, by_slug)
        self.assertEqual((matched, updated), (1, 1))
        model = doc["models"][0]
        self.assertEqual(model["scores"]["gpqa_diamond"], 89.6)
        self.assertEqual(model["scores_source"]["gpqa_diamond"], "https://openrouter.ai/x/a")
        self.assertEqual(changes, [("a", "gpqa_diamond", None, 89.6)])


if __name__ == "__main__":
    unittest.main()
