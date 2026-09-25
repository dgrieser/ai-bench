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
        ("toolathlon", "score", "fetch_toolathlon_data"),
        ("real_swe", "score", "fetch_real_swe_data"),
        ("programbench", "score", "fetch_programbench_data"),
        ("mcp_atlas", "score", "fetch_mcp_atlas_data"),
        ("bfcl", "score", "fetch_bfcl_data"),
        ("agents_last_exam", "score", "fetch_agents_last_exam_data"),
    ]

    # The revision-split sources file rows under a column per revision, so the
    # same collision resolves one level down. The current revision is named per
    # source: FrontierSWE numbered its re-run 2.0 where the others went to 1.1,
    # and a row naming a revision its benchmark has no column for is refused
    # rather than filed (see TestRevisionRouting). `extra` carries whatever else
    # a source needs on a row before it can be filed at all -- FrontierCode
    # splits on its task subset as well, so its rows name one.
    REVISION_CASES = [
        ("deepswe", "fetch_deepswe_data", "deepswe", "1.1", {}),
        ("datacurve", "fetch_datacurve_data", "deepswe", "1.1", {}),
        (
            "frontiercode", "fetch_frontiercode_data", "frontiercode", "1.1",
            {"subset": "main"},
        ),
        ("frontierswe", "fetch_frontierswe_data", "frontierswe", "2.0", {}),
        ("swe_marathon", "fetch_swe_marathon_data", "swe_marathon", "1.1", {}),
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

    def test_tbench_best_row_wins_within_its_board_in_either_payload_order(self) -> None:
        # fetch_tbench.py names the column on every row, one board per
        # Terminal-Bench revision, so the fold is per column like the above.
        mapping = write_json({"Model": "m", "Model [high]": "m"})
        rows = [
            {"benchmark": "terminal_bench_2_1", "model": "Model", "score": 30.0},
            {"benchmark": "terminal_bench_2_1", "model": "Model [high]", "score": 45.0},
            {"benchmark": "terminal_bench_4_0", "model": "Model", "score": 20.0},
        ]
        for order in (rows, list(reversed(rows))):
            with self.subTest(first=order[0]["model"]):
                with stub_run(order):
                    by_key = update.fetch_tbench_data(SCRIPT, mapping)
                self.assertEqual(by_key["terminal_bench_2_1"]["m"]["score"], 45.0)
                self.assertEqual(by_key["terminal_bench_4_0"]["m"]["score"], 20.0)

    def test_osworld_best_row_wins_within_its_board_in_either_payload_order(self) -> None:
        # fetch_osworld.py names the column on every row, the Verified board and
        # each tracked OSWorld 2.0 release alike, so the fold is per column.
        mapping = write_json({"Model": "m", "Model [high]": "m"})
        rows = [
            {"benchmark": "osworld_2_0_2026_06_24", "model": "Model", "score": 3.0},
            {"benchmark": "osworld_2_0_2026_06_24", "model": "Model [high]", "score": 4.6},
            {"benchmark": "osworld_verified", "model": "Model", "score": 70.0},
        ]
        for order in (rows, list(reversed(rows))):
            with self.subTest(first=order[0]["model"]):
                with stub_run(order):
                    by_key = update.fetch_osworld_data(SCRIPT, mapping)
                self.assertEqual(by_key["osworld_2_0_2026_06_24"]["m"]["score"], 4.6)
                self.assertEqual(by_key["osworld_verified"]["m"]["score"], 70.0)

    def test_best_row_wins_within_a_revision_in_either_payload_order(self) -> None:
        for source, func_name, base, revision, extra in self.REVISION_CASES:
            mapping = write_json({"Model": "m", "Model [high]": "m"})
            rows = [
                {"model": "Model", "revision": revision, "score": 30.0, **extra},
                {"model": "Model [high]", "revision": revision, "score": 45.0, **extra},
            ]
            key = f"{base}_{revision.replace('.', '_')}"
            for order in (rows, list(reversed(rows))):
                with self.subTest(source=source, first=order[0]["model"]):
                    with stub_run(order):
                        by_key = getattr(update, func_name)(SCRIPT, mapping)
                    self.assertEqual(by_key[key]["m"]["score"], 45.0)


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


class TestValsMerge(unittest.TestCase):
    """Vals names a model by the provider it called, so one llm.json slug can
    collect rows from two paths (fireworks/ and openai/ serving one open model),
    each carrying several benchmark keys. The fold has to be per key, and the
    best row has to win whichever order the payload arrives in."""

    def test_best_path_wins_per_key_in_either_order(self) -> None:
        mapping = write_json({"fireworks/model": "m", "together/model": "m"})
        rows = [
            {"model": "fireworks/model", "key": "mmlu_pro", "score": 30.0},
            {"model": "fireworks/model", "key": "gpqa_diamond", "score": 80.0},
            {"model": "together/model", "key": "mmlu_pro", "score": 45.0},
            {"model": "together/model", "key": "gpqa_diamond", "score": 70.0},
        ]
        for order in (rows, list(reversed(rows))):
            with self.subTest(first=order[0]["model"]):
                with stub_run(order):
                    by_slug = update.fetch_vals_data(SCRIPT, mapping)
                self.assertEqual(by_slug["m"], {"mmlu_pro": 45.0, "gpqa_diamond": 80.0})


class TestRevisionRouting(unittest.TestCase):
    """DeepSWE, FrontierCode, FrontierSWE and SWE-Marathon keep a column per revision.

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
        for base in ("deepswe", "frontiercode", "frontierswe", "swe_marathon"):
            with self.subTest(base=base):
                by_key = self.route({"revision": "1.0", "score": 1.0}, base=base)
                self.assertEqual(list(by_key), [f"{base}_1_0"])

    def test_a_benchmark_only_takes_the_revisions_it_published(self) -> None:
        """FrontierSWE numbered its re-run 2.0; the others went to 1.1.

        Each base knows its own labels, so the row a sibling's numbering would
        produce is refused rather than landing in a column nothing renders.
        """
        self.assertEqual(
            list(self.route({"revision": "2.0", "score": 1.0}, base="frontierswe")),
            ["frontierswe_2_0"],
        )
        self.assertEqual(self.route({"revision": "1.1", "score": 1.0}, base="frontierswe"), {})
        self.assertEqual(self.route({"revision": "2.0", "score": 1.0}, base="frontiercode"), {})

    def test_fetch_routes_a_folded_name_by_revision(self) -> None:
        """Two leaderboard names folding onto one slug, one per revision."""
        mapping = write_json({"Model": "m", "Model Redux": "m"})
        rows = [
            {"model": "Model", "revision": "1.0", "subset": "main", "score": 45.0},
            {"model": "Model Redux", "revision": "1.1", "subset": "main", "score": 30.0},
        ]
        for order in (rows, list(reversed(rows))):
            with self.subTest(first=order[0]["model"]):
                with stub_run(order):
                    by_key = update.fetch_frontiercode_data(SCRIPT, mapping)
                self.assertEqual(by_key["frontiercode_1_0"]["m"]["score"], 45.0)
                self.assertEqual(by_key["frontiercode_1_1"]["m"]["score"], 30.0)

    def test_fetch_routes_the_task_subsets_apart(self) -> None:
        """Main and Extended are one run scored twice, ten points apart.

        Letting them compete for one column would be the same blend the
        revision split exists to end, so each lands in its own.
        """
        mapping = write_json({"Model": "m"})
        rows = [
            {"model": "Model", "revision": "1.1", "subset": "main", "score": 30.0},
            {"model": "Model", "revision": "1.1", "subset": "extended", "score": 43.0},
        ]
        for order in (rows, list(reversed(rows))):
            with self.subTest(first=order[0]["subset"]):
                with stub_run(order):
                    by_key = update.fetch_frontiercode_data(SCRIPT, mapping)
                self.assertEqual(by_key["frontiercode_1_1"]["m"]["score"], 30.0)
                self.assertEqual(
                    by_key["frontiercode_extended_1_1"]["m"]["score"], 43.0
                )

    def test_fetch_drops_rows_naming_no_revision(self) -> None:
        mapping = write_json({"Model": "m"})
        with stub_run([{"model": "Model", "subset": "main", "score": 45.0}]):
            self.assertEqual(update.fetch_frontiercode_data(SCRIPT, mapping), {})

    def test_fetch_drops_rows_whose_subset_has_no_column(self) -> None:
        # No subset named at all, a subset nothing tracks, and the one board
        # published without a column of its own: 1.0's Extended.
        mapping = write_json({"Model": "m"})
        for row in (
            {"model": "Model", "revision": "1.1", "score": 45.0},
            {"model": "Model", "revision": "1.1", "subset": "diamond", "score": 45.0},
            {"model": "Model", "revision": "1.0", "subset": "extended", "score": 45.0},
        ):
            with self.subTest(subset=row.get("subset")):
                with stub_run([row]):
                    self.assertEqual(
                        update.fetch_frontiercode_data(SCRIPT, mapping), {}
                    )


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
    def test_a_partial_card_is_a_failed_page(self) -> None:
        """Its missing scores are not ones it stopped reporting (#226)."""
        mapping = write_json({"IFEval": "ifeval"})
        update.RUN_REPORTS = update.RunReports()
        rows = [
            {"model": "m", "repo": "org/m-part", "scores": {"IFEval": 30.0}, "partial": True},
            {"model": "n", "repo": "org/n-full", "scores": {"IFEval": 45.0}},
        ]
        with stub_run(rows):
            update.fetch_huggingface_data(SCRIPT, mapping)
        self.assertEqual(update.RUN_REPORTS.failed_pages, {"https://huggingface.co/org/m-part"})

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

    def test_a_plain_label_beats_a_qualified_one_whatever_the_values(self) -> None:
        # One card, one benchmark, two runs. "Best harness" is a maximum over
        # configurations and "w/ tools" is a different question being asked;
        # neither is the number the column holds, so neither wins on being
        # bigger.
        mapping = write_json({
            "Terminal Bench 2.1": "terminal_bench_2_1",
            "Terminal Bench 2.1 (best harness)": "terminal_bench_2_1",
            "HLE": "hle",
            "HLE w/ tools": "hle",
        })
        rows = [{
            "model": "m",
            "repo": "org/m",
            "scores": {
                "Terminal Bench 2.1 (best harness)": 90.6,
                "Terminal Bench 2.1": 85.2,
                "HLE w/ tools": 63.9,
                "HLE": 36.8,
            },
        }]
        with stub_run(rows):
            by_slug = update.fetch_huggingface_data(SCRIPT, mapping)
        self.assertEqual(by_slug["m"]["terminal_bench_2_1"][0], 85.2)
        self.assertEqual(by_slug["m"]["hle"][0], 36.8)

    def test_equally_qualified_labels_still_take_the_best_run(self) -> None:
        # A card printing one benchmark twice at two budgets, with nothing to
        # separate the labels, keeps the policy it always had.
        mapping = write_json({"LiveCodeBench v6": "livecodebench", "LiveCodeBench-v6": "livecodebench"})
        rows = [{
            "model": "m",
            "repo": "org/m",
            "scores": {"LiveCodeBench v6": 30.8, "LiveCodeBench-v6": 80.6},
        }]
        with stub_run(rows):
            by_slug = update.fetch_huggingface_data(SCRIPT, mapping)
        self.assertEqual(by_slug["m"]["livecodebench"][0], 80.6)

    def test_the_structured_channel_beats_the_card_s_own_table(self) -> None:
        # One column read two ways: the Hub's "Evaluation results" widget files
        # GPQA Diamond under its dataset id, the card's table heads it "GPQA
        # Diamond", and after the mapping both are gpqa_diamond. Taking the
        # larger is how a table number displaces the structured one, which is
        # the channel a benchmark owner can write to.
        mapping = write_json({
            "Idavidrein/gpqa (diamond)": "gpqa_diamond",
            "GPQA Diamond": "gpqa_diamond",
        })
        rows = [{
            "model": "m",
            "repo": "org/m",
            "scores": {"Idavidrein/gpqa (diamond)": 70.8, "GPQA Diamond": 76.8},
            "channels": {"Idavidrein/gpqa (diamond)": "metadata", "GPQA Diamond": "table"},
        }]
        with stub_run(rows):
            by_slug = update.fetch_huggingface_data(SCRIPT, mapping)
        self.assertEqual(by_slug["m"]["gpqa_diamond"][0], 70.8)

    def test_a_plain_table_label_still_beats_a_qualified_structured_one(self) -> None:
        # Channel is the tiebreak under the qualifier rule, not over it.
        mapping = write_json({"cais/hle (with tools)": "hle", "HLE": "hle"})
        rows = [{
            "model": "m",
            "repo": "org/m",
            "scores": {"cais/hle (with tools)": 63.9, "HLE": 36.8},
            "channels": {"cais/hle (with tools)": "metadata", "HLE": "table"},
        }]
        with stub_run(rows):
            by_slug = update.fetch_huggingface_data(SCRIPT, mapping)
        self.assertEqual(by_slug["m"]["hle"][0], 36.8)

    def test_a_crawl_without_channels_keeps_the_old_policy(self) -> None:
        # A payload cached before this field existed says nothing about where
        # its labels came from, and must not be read as saying "table".
        mapping = write_json({"A": "hle", "B": "hle"})
        rows = [{"model": "m", "repo": "org/m", "scores": {"A": 30.1, "B": 24.4}}]
        with stub_run(rows):
            by_slug = update.fetch_huggingface_data(SCRIPT, mapping)
        self.assertEqual(by_slug["m"]["hle"][0], 30.1)

    def test_pass_at_k_is_qualified_for_every_k_but_one(self) -> None:
        # pass@10 is best-of-ten. The first-digit range this used to test for
        # let 10, 11 and 100 through as if they were pass@1.
        for label in ("Foo (Pass@2)", "Foo (Pass@5)", "Foo (Pass@10)",
                      "Foo (Pass@16)", "Foo (Pass@100)"):
            with self.subTest(label=label):
                self.assertEqual(update.hf_label_rank(label), 1)
        for label in ("Foo (Pass@1)", "Foo", "Foo (Avg@32)"):
            with self.subTest(label=label):
                self.assertEqual(update.hf_label_rank(label), 0)

    def test_a_no_tools_label_is_not_a_qualified_one(self) -> None:
        # Every benchmark column here holds the no-tool run, so a label saying
        # so names the column rather than a variant of it.
        mapping = write_json({"HLE": "hle", "HLE (no tools)": "hle"})
        rows = [{"model": "m", "repo": "org/m", "scores": {"HLE": 30.1, "HLE (no tools)": 24.4}}]
        with stub_run(rows):
            by_slug = update.fetch_huggingface_data(SCRIPT, mapping)
        self.assertEqual(by_slug["m"]["hle"][0], 30.1)



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
