#!/usr/bin/env python3
"""Tests for the Vals AI leaderboard reader. Run with ./test_vals.py

Two things carry the weight here.

``test_a_renamed_slug_is_not_read_as_the_old_board``: Vals serves one board per
URL and its slugs have already moved once (``terminal-bench-2`` sits beside
``terminal-bench-2-1``). If ``/benchmarks/gpqa`` ever redirected to a successor,
a reader that trusted the URL would file the new board's numbers under the old
column and every model would appear to move at once, so the parser checks the
slug the payload reports against the slug it asked for.

``test_aime_is_read_at_the_2025_task``: Vals' AIME board pools the 2024 and 2025
exams into its "overall" task while llm.json keeps a column per exam year. The
task override is what stops the pooled number reaching ``aime_2025``.
"""

from __future__ import annotations

import html
import json
import unittest
from pathlib import Path
from unittest import mock

import fetch_vals as fv


def cell(accuracy: float, provider: str = "Example", stderr: float = 1.5) -> dict:
    return {
        "accuracy": accuracy,
        "latency": 100.0,
        "stderr": stderr,
        "cost_per_test": 0.5,
        "temperature": None,
        "top_p": None,
        "max_output_tokens": None,
        "reasoning": None,
        "reasoning_effort": None,
        "verbosity": None,
        "compute_effort": None,
        "provider": provider,
        "harness": None,
    }


def encode(value):
    """Re-encode a plain value the way Astro serializes island props."""
    if isinstance(value, dict):
        return [0, {key: encode(item) for key, item in value.items()}]
    if isinstance(value, list):
        return [1, [encode(item) for item in value]]
    return [0, value]


def page(slug: str, tasks: dict, updated: str = "2026-09-01", **metadata) -> str:
    """A Vals benchmark page with the BenchmarkView island embedded."""
    board = {
        "metadata": {
            "benchmark": slug,
            "slug": slug,
            "version": "1",
            "updated": updated,
            "tasks": {name: name for name in tasks},
            "models": sorted({m for cells in tasks.values() for m in cells}),
            **metadata,
        },
        "tasks": tasks,
    }
    props = {"benchmarkView": encode({"default": board, **board})}
    attr = html.escape(json.dumps(props), quote=True)
    return (
        '<astro-island component-url="/_astro/GlobalToaster.DgiC_xH1.js" props="{}">'
        "</astro-island>"
        f'<astro-island component-url="/_astro/BenchmarkView.Bv3BoWAp.js"'
        f' props="{attr}" client="load"></astro-island>'
    )


def stub_page(body: str):
    return mock.patch.object(fv, "fetch_html", return_value=body)


class TestDeserialize(unittest.TestCase):
    def test_round_trips_nested_values(self) -> None:
        value = {"a": [1, 2, {"b": "c"}], "d": None, "e": 1.5}
        self.assertEqual(fv.deserialize(encode(value)), value)

    def test_unknown_type_yields_its_raw_value(self) -> None:
        # A Date (type 3) appearing someday must not take the ingest down.
        self.assertEqual(fv.deserialize([3, "2026-09-01"]), "2026-09-01")

    def test_a_plain_value_passes_through(self) -> None:
        self.assertEqual(fv.deserialize("already decoded"), "already decoded")


class TestParseBoard(unittest.TestCase):
    def test_reads_the_requested_task(self) -> None:
        body = page(
            "gpqa",
            {
                "overall": {"zai/glm-5.3": cell(80.0)},
                "diamond_zero_shot_cot": {"zai/glm-5.3": cell(70.0)},
            },
        )
        metadata, cells = fv.parse_board(body, "gpqa")
        self.assertEqual(metadata["updated"], "2026-09-01")
        self.assertEqual(cells["zai/glm-5.3"]["accuracy"], 80.0)

    def test_a_renamed_slug_is_not_read_as_the_old_board(self) -> None:
        body = page("gpqa-2", {"overall": {"zai/glm-5.3": cell(80.0)}})
        with self.assertRaises(ValueError) as caught:
            fv.parse_board(body, "gpqa")
        self.assertIn("gpqa", str(caught.exception))

    def test_a_missing_task_is_refused(self) -> None:
        body = page("aime", {"overall": {"zai/glm-5.3": cell(80.0)}})
        with self.assertRaises(ValueError) as caught:
            fv.parse_board(body, "aime")
        self.assertIn("aime_2025", str(caught.exception))

    def test_a_page_without_the_island_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            fv.parse_board("<html><body>maintenance</body></html>", "gpqa")


class TestGetScores(unittest.TestCase):
    def test_rows_carry_the_column_the_board_feeds(self) -> None:
        body = page(
            "mmmu",
            {"overall": {"zai/glm-5.3-flash": cell(86.01), "openai/gpt-5.5": cell(88.27)}},
        )
        with stub_page(body):
            rows = fv.get_scores(["mmmu"])
        self.assertEqual({r["key"] for r in rows}, {"mmmu_pro"})
        by_model = {r["model"]: r for r in rows}
        self.assertEqual(by_model["zai/glm-5.3-flash"]["score"], 86.01)
        self.assertEqual(by_model["zai/glm-5.3-flash"]["label"], "glm-5.3-flash")
        self.assertEqual(by_model["zai/glm-5.3-flash"]["date"], "2026-09-01")
        # Rank is over the board, best first.
        self.assertEqual(by_model["openai/gpt-5.5"]["rank"], 1)
        self.assertEqual(by_model["zai/glm-5.3-flash"]["rank"], 2)

    def test_aime_is_read_at_the_2025_task(self) -> None:
        body = page(
            "aime",
            {
                # The pooled number, which must not reach aime_2025.
                "overall": {"zai/glm-5.3": cell(90.0)},
                "aime_2024": {"zai/glm-5.3": cell(95.0)},
                "aime_2025": {"zai/glm-5.3": cell(85.0)},
            },
        )
        with stub_page(body):
            rows = fv.get_scores(["aime"])
        self.assertEqual([(r["key"], r["task"], r["score"]) for r in rows],
                         [("aime_2025", "aime_2025", 85.0)])

    def test_a_cell_without_a_number_is_dropped(self) -> None:
        body = page(
            "mmmu",
            {
                "overall": {
                    "zai/glm-5.3": cell(86.0),
                    "openai/pending": {"accuracy": None, "provider": "OpenAI"},
                    "openai/flagged": {"accuracy": True, "provider": "OpenAI"},
                }
            },
        )
        with stub_page(body):
            rows = fv.get_scores(["mmmu"])
        self.assertEqual([r["model"] for r in rows], ["zai/glm-5.3"])


class TestBenchmarkTable(unittest.TestCase):
    def test_every_board_names_a_column_llm_json_tracks(self) -> None:
        path = Path(__file__).resolve().with_name("llm.json")
        doc = json.loads(path.read_text(encoding="utf-8"))
        for slug, key in fv.BENCHMARKS.items():
            with self.subTest(slug=slug):
                self.assertIn(key, doc["benchmarks"])

    def test_no_two_boards_feed_one_column(self) -> None:
        keys = list(fv.BENCHMARKS.values())
        self.assertEqual(len(keys), len(set(keys)))

    def test_task_overrides_name_a_board_that_is_ingested(self) -> None:
        self.assertLessEqual(set(fv.TASKS), set(fv.BENCHMARKS))


if __name__ == "__main__":
    unittest.main()
