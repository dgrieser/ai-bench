#!/usr/bin/env python3
"""Tests for many-to-one model mappings. Run with ./test_source_collisions.py

Every mapping file folds model variants onto one llm.json slug -- a base row and
its "[high]" sibling, a label spelled two ways, a dated re-release. The score
that lands in llm.json must not depend on the leaderboard's own row ordering, so
every ingest resolves a collision the same way: best reported run wins. These
tests feed each fetcher the colliding rows in both orders and fail if the answer
moves.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import update

SCRIPT = Path("unused.py")


def write_json(payload: object) -> Path:
    path = Path(tempfile.mkdtemp()) / "mapping.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def stub_run(payload: object):
    return mock.patch.object(
        update.subprocess,
        "run",
        return_value=SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr=""),
    )


class TestKeepBestRow(unittest.TestCase):
    def test_higher_score_wins_either_order(self) -> None:
        for rows in (
            [{"score": 30}, {"score": 45}],
            [{"score": 45}, {"score": 30}],
        ):
            by_slug: dict = {}
            for row in rows:
                update.keep_best_row(by_slug, "m", row, "score")
            self.assertEqual(by_slug["m"]["score"], 45)

    def test_whole_row_is_kept_not_just_the_score(self) -> None:
        by_slug: dict = {}
        update.keep_best_row(by_slug, "m", {"score": 30, "runs": 1}, "score")
        update.keep_best_row(by_slug, "m", {"score": 45, "runs": 9}, "score")
        self.assertEqual(by_slug["m"], {"score": 45, "runs": 9})

    def test_scoreless_row_never_displaces_a_scored_one(self) -> None:
        by_slug: dict = {}
        update.keep_best_row(by_slug, "m", {"score": 30}, "score")
        update.keep_best_row(by_slug, "m", {"score": None}, "score")
        self.assertEqual(by_slug["m"]["score"], 30)

    def test_a_scored_row_displaces_a_scoreless_one(self) -> None:
        by_slug: dict = {}
        update.keep_best_row(by_slug, "m", {"score": None}, "score")
        update.keep_best_row(by_slug, "m", {"score": 30}, "score")
        self.assertEqual(by_slug["m"]["score"], 30)

    def test_booleans_are_not_scores(self) -> None:
        by_slug: dict = {}
        update.keep_best_row(by_slug, "m", {"score": 0.5}, "score")
        update.keep_best_row(by_slug, "m", {"score": True}, "score")
        self.assertEqual(by_slug["m"]["score"], 0.5)


class TestRowFetchers(unittest.TestCase):
    """The five ingests that read one score field off a whole source row."""

    CASES = [
        ("swe_rebench", "resolved_rate", "fetch_swe_rebench_data"),
        ("osworld", "success_rate", "fetch_osworld_data"),
        ("toolathlon", "score", "fetch_toolathlon_data"),
        ("deepswe", "score", "fetch_deepswe_data"),
        ("frontierswe", "score", "fetch_frontierswe_data"),
    ]

    def test_best_row_wins_in_either_payload_order(self) -> None:
        for source, score_key, func_name in self.CASES:
            mapping = write_json({"Model": "m", "Model [high]": "m"})
            rows = [
                {"model": "Model", score_key: 30.0},
                {"model": "Model [high]", score_key: 45.0},
            ]
            for order in (rows, list(reversed(rows))):
                with self.subTest(source=source, first=order[0]["model"]):
                    with stub_run(order):
                        by_slug = getattr(update, func_name)(SCRIPT, mapping)
                    self.assertEqual(by_slug["m"][score_key], 45.0)


class TestLlmstatsMerge(unittest.TestCase):
    def setUp(self) -> None:
        self.models = write_json({"m-base": "m", "m-high": "m"})
        self.benchmarks = write_json({"HLE": "hle", "GPQA": "gpqa_diamond"})

    def fetch(self, payload: object) -> dict:
        with stub_run(payload):
            return update.fetch_llmstats_data(SCRIPT, self.models, self.benchmarks)

    def test_shared_benchmark_takes_the_best_run(self) -> None:
        rows = [
            {"model": "m-base", "scores": {"HLE": 0.30}},
            {"model": "m-high", "scores": {"HLE": 0.45}},
        ]
        for order in (rows, list(reversed(rows))):
            self.assertEqual(self.fetch(order)["m"]["hle"], 45.0)

    def test_disjoint_benchmarks_still_merge(self) -> None:
        by_slug = self.fetch(
            [
                {"model": "m-base", "scores": {"HLE": 0.30}},
                {"model": "m-high", "scores": {"GPQA": 0.90}},
            ]
        )
        self.assertEqual(by_slug["m"], {"hle": 30.0, "gpqa_diamond": 90.0})

    def test_aliased_labels_within_one_row_take_the_best(self) -> None:
        benchmarks = write_json({"HLE": "hle", "HLE (no tools)": "hle"})
        with stub_run([{"model": "m-base", "scores": {"HLE": 0.30, "HLE (no tools)": 0.45}}]):
            by_slug = update.fetch_llmstats_data(SCRIPT, self.models, benchmarks)
        self.assertEqual(by_slug["m"]["hle"], 45.0)


class TestHuggingfaceMerge(unittest.TestCase):
    def test_repeated_rows_take_the_best_run(self) -> None:
        mapping = write_json({"IFEval": "ifeval"})
        rows = [
            {"model": "m", "repo": "org/m-base", "scores": {"IFEval": 30.0}},
            {"model": "m", "repo": "org/m-high", "scores": {"IFEval": 45.0}},
        ]
        for order in (rows, list(reversed(rows))):
            with stub_run(order):
                by_slug = update.fetch_huggingface_data(SCRIPT, mapping)
            # The winning run carries the model card it was read from.
            self.assertEqual(
                by_slug["m"]["ifeval"],
                (45.0, "https://huggingface.co/org/m-high"),
            )


class TestSpheronMerge(unittest.TestCase):
    def test_largest_vram_estimate_wins_per_quant(self) -> None:
        mapping = write_json({"org/Model": "m", "org/Model-0731": "m"})
        rows = [
            {"model": "org/Model", "vram_fp16": 600, "vram_int8": 300, "vram_int4": None},
            {"model": "org/Model-0731", "vram_fp16": 580, "vram_int8": 320, "vram_int4": 90},
        ]
        for order in (rows, list(reversed(rows))):
            with stub_run(order):
                by_slug = update.fetch_spheron_data(SCRIPT, mapping)
            self.assertEqual(by_slug["m"], {"fp16": 600, "int8": 320, "int4": 90})


if __name__ == "__main__":
    unittest.main()
