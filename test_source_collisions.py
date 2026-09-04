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
    """The ingests that read one score field off a whole source row."""

    CASES = [
        ("osworld", "success_rate", "fetch_osworld_data"),
        ("toolathlon", "score", "fetch_toolathlon_data"),
        ("mcp_atlas", "score", "fetch_mcp_atlas_data"),
        ("bfcl", "score", "fetch_bfcl_data"),
        ("frontierswe", "score", "fetch_frontierswe_data"),
        ("tbench", "score", "fetch_tbench_data"),
        ("agents_last_exam", "score", "fetch_agents_last_exam_data"),
    ]

    # The revision-split sources file rows under a column per revision, so the
    # same collision resolves one level down.
    REVISION_CASES = [
        ("deepswe", "fetch_deepswe_data", "deepswe"),
        ("datacurve", "fetch_datacurve_data", "deepswe"),
        ("frontiercode", "fetch_frontiercode_data", "frontiercode"),
        ("swe_marathon", "fetch_swe_marathon_data", "swe_marathon"),
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

    def test_best_row_wins_within_a_revision_in_either_payload_order(self) -> None:
        for source, func_name, base in self.REVISION_CASES:
            mapping = write_json({"Model": "m", "Model [high]": "m"})
            rows = [
                {"model": "Model", "revision": "1.1", "score": 30.0},
                {"model": "Model [high]", "revision": "1.1", "score": 45.0},
            ]
            for order in (rows, list(reversed(rows))):
                with self.subTest(source=source, first=order[0]["model"]):
                    with stub_run(order):
                        by_key = getattr(update, func_name)(SCRIPT, mapping)
                    self.assertEqual(by_key[f"{base}_1_1"]["m"]["score"], 45.0)


class TestAaCodingAgentsMerge(unittest.TestCase):
    """The Coding Agent Index lists one row per (agent, effort) variant and
    carries several benchmarks per row, so the fold is per benchmark key."""

    def test_best_variant_wins_per_key_in_either_order(self) -> None:
        mapping = write_json({"model": "m", "model [high]": "m"})
        rows = [
            {"model": "model", "key": "swe_atlas_qna", "score": 30.0},
            {"model": "model", "key": "terminal_bench_2_1", "score": 80.0},
            {"model": "model [high]", "key": "swe_atlas_qna", "score": 45.0},
            {"model": "model [high]", "key": "terminal_bench_2_1", "score": 70.0},
        ]
        for order in (rows, list(reversed(rows))):
            with self.subTest(first=order[0]["model"]):
                with stub_run(order):
                    by_slug = update.fetch_aa_coding_agents_data(SCRIPT, mapping)
                self.assertEqual(
                    by_slug["m"], {"swe_atlas_qna": 45.0, "terminal_bench_2_1": 80.0}
                )


class TestRevisionRouting(unittest.TestCase):
    """DeepSWE, FrontierCode and SWE-Marathon keep a column per revision.

    A row therefore has to say which revision it measured before it can be
    written anywhere, and rows compete only against their own revision -- the
    best-run rule must never let a retired revision's higher number displace
    the current re-run's, which is the blend the split exists to end.
    """

    def route(self, *rows: dict, base: str = "frontiercode") -> dict:
        by_key: dict = {}
        for row in rows:
            update.keep_best_by_revision(by_key, base, "m", row)
        return by_key

    def test_each_revision_lands_in_its_own_column(self) -> None:
        old = {"model": "Model", "revision": "1.0", "score": 45.0}
        new = {"model": "Model Redux", "revision": "1.1", "score": 30.0}
        for order in ((old, new), (new, old)):
            with self.subTest(first=order[0]["revision"]):
                by_key = self.route(*order)
                self.assertEqual(by_key["frontiercode_1_0"]["m"]["score"], 45.0)
                self.assertEqual(by_key["frontiercode_1_1"]["m"]["score"], 30.0)

    def test_a_retired_revisions_higher_score_never_reaches_the_current_column(self) -> None:
        by_key = self.route(
            {"model": "Model", "revision": "1.0", "score": 99.0},
            {"model": "Model", "revision": "1.1", "score": 1.0},
        )
        self.assertEqual(by_key["frontiercode_1_1"]["m"]["score"], 1.0)

    def test_within_one_revision_the_best_run_still_wins(self) -> None:
        low = {"model": "Model", "revision": "1.1", "score": 30.0}
        high = {"model": "Model [high]", "revision": "1.1", "score": 45.0}
        for order in ((low, high), (high, low)):
            with self.subTest(first=order[0]["model"]):
                self.assertEqual(
                    self.route(*order)["frontiercode_1_1"]["m"]["score"], 45.0
                )

    def test_a_row_naming_no_revision_is_refused(self) -> None:
        by_key: dict = {}
        self.assertFalse(
            update.keep_best_by_revision(by_key, "frontiercode", "m", {"score": 45.0})
        )
        self.assertEqual(by_key, {})

    def test_a_revision_without_a_column_is_refused(self) -> None:
        by_key: dict = {}
        self.assertFalse(
            update.keep_best_by_revision(
                by_key, "frontiercode", "m", {"revision": "9.9", "score": 45.0}
            )
        )
        self.assertEqual(by_key, {})

    def test_every_split_benchmark_routes_the_same_way(self) -> None:
        for base in ("deepswe", "frontiercode", "swe_marathon"):
            with self.subTest(base=base):
                by_key = self.route({"revision": "1.0", "score": 1.0}, base=base)
                self.assertEqual(list(by_key), [f"{base}_1_0"])

    def test_fetch_routes_a_folded_name_by_revision(self) -> None:
        """Two leaderboard names folding onto one slug, one per revision."""
        mapping = write_json({"Model": "m", "Model Redux": "m"})
        rows = [
            {"model": "Model", "revision": "1.0", "score": 45.0},
            {"model": "Model Redux", "revision": "1.1", "score": 30.0},
        ]
        for order in (rows, list(reversed(rows))):
            with self.subTest(first=order[0]["model"]):
                with stub_run(order):
                    by_key = update.fetch_frontiercode_data(SCRIPT, mapping)
                self.assertEqual(by_key["frontiercode_1_0"]["m"]["score"], 45.0)
                self.assertEqual(by_key["frontiercode_1_1"]["m"]["score"], 30.0)

    def test_fetch_drops_rows_naming_no_revision(self) -> None:
        mapping = write_json({"Model": "m"})
        with stub_run([{"model": "Model", "score": 45.0}]):
            self.assertEqual(update.fetch_frontiercode_data(SCRIPT, mapping), {})


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
            {"model": "org/Model", "source": "https://spheron/model", "vram_fp16": 600, "vram_int8": 300, "vram_int4": None},
            {"model": "org/Model-0731", "source": "https://spheron/revision", "vram_fp16": 580, "vram_int8": 320, "vram_int4": 90},
        ]
        for order in (rows, list(reversed(rows))):
            with stub_run(order):
                by_slug = update.fetch_spheron_data(SCRIPT, mapping)
            self.assertEqual(by_slug["m"], {
                "fp16": 600,
                "int8": 320,
                "int4": 90,
                "source": {
                    "fp16": "https://spheron/model",
                    "int8": "https://spheron/revision",
                    "int4": "https://spheron/revision",
                },
            })

    def test_source_is_stored_with_each_vram_value(self) -> None:
        doc = {"models": [{"name": "m", "vram": {"fp16": 10}}]}
        fetched = {
            "m": {
                "fp16": 10,
                "int8": 5,
                "int4": None,
                "source": {
                    "fp16": "https://spheron/model",
                    "int8": "https://spheron/model",
                },
            }
        }

        update.update_spheron_vram(doc, fetched)

        self.assertEqual(doc["models"][0]["vram_source"], {
            "fp16": "https://spheron/model",
            "int8": "https://spheron/model",
        })


if __name__ == "__main__":
    unittest.main()
