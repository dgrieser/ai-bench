#!/usr/bin/env python3
"""Tests for editing decided mappings from the admin page's Models tab.

Run with ./test_mapping_edits.py

Two records, both answering no queued question: `mapping-edit` re-points,
parks, adds or removes any line of a model mapping file, and `rejection-edit`
changes a name's "not these" list. Without a queue to bound them, what holds
them is the rest of what holds a mapping answer -- the route table, the
route's own list, if_previous, the matcher's exact tier -- and that is what
most of this file checks. _mapping_index.py, which publishes what the page
edits, is tested at the end.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _answers
import _mapping_index
import _rejections
import propose
from _answers import MAPPING, MAPPING_EDIT, MODEL_RENAME, REJECTION_EDIT, Answer, AnswerError

TBENCH = "update_tbench_mapping.py"
HF = "update_huggingface_mapping.py"

UNIVERSES = {
    propose.MODELS: ["glm-5-3", "devstral-2", "hy4"],
    propose.BENCHMARKS: ["swe_bench_verified", "hle"],
    propose.AA_SLUGS: ["aa-one", "aa-two"],
}

LLM_DOC = {
    "benchmarks": {"hle": {"name": "HLE"}},
    "models": [
        {"name": n, "scores": {}, "scores_updated": {}, "scores_source": {}}
        for n in ("devstral-2", "glm-5-3", "hy4")
    ],
}


class EditTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.llm = self.tmp / "llm.json"
        self.llm.write_text(json.dumps(LLM_DOC), encoding="utf-8")
        # What the file says now, per test; the real files are never read.
        self.file: dict[str, list[str] | None] = {}
        patcher = mock.patch.object(
            _answers, "current_value", side_effect=lambda _route, subject: self.file.get(subject)
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.rejected: set[str] = set()
        patcher = mock.patch.object(_rejections, "rejected", side_effect=lambda *_a, **_k: set(self.rejected))
        patcher.start()
        self.addCleanup(patcher.stop)

    def validate(self, records):
        # An empty queue, enforced: none of these records may need one.
        return _answers.validate(records, llm_path=self.llm, universes=UNIVERSES,
                                 queue=set(), require_queue=True)

    def accepted(self, record) -> Answer:
        answers, failures = self.validate([record])
        self.assertEqual([str(f) for f in failures], [])
        return answers[0]

    def refused(self, record, needle: str) -> None:
        answers, failures = self.validate([record])
        self.assertEqual(answers, [], f"expected {record} to be refused")
        self.assertIn(needle, failures[0].message)

    def edit(self, **overrides) -> dict:
        record = {
            "kind": MAPPING_EDIT, "route": TBENCH, "route_kind": "*",
            "subject": "Hy4 preview", "answer": "hy4", "if_previous": ["__unmappable__"],
        }
        record.update(overrides)
        return record

    def rejection(self, **overrides) -> dict:
        record = {
            "kind": REJECTION_EDIT, "route": TBENCH, "route_kind": "*",
            "subject": "Some Model", "add": [], "remove": [],
        }
        record.update(overrides)
        return record


class TestMappingEdit(EditTestCase):
    def test_a_parked_line_can_be_brought_back_without_a_queue(self) -> None:
        """The case this exists for: a cleanup parked a name, and nothing asks again."""
        self.file["Hy4 preview"] = ["__unmappable__"]
        answer = self.accepted(self.edit())
        self.assertEqual((answer.kind, answer.value), (MAPPING_EDIT, "hy4"))

    def test_a_line_can_move_to_another_model_or_be_parked(self) -> None:
        self.file["Hy4 preview"] = ["hy4"]
        self.accepted(self.edit(answer="glm-5-3", if_previous=["hy4"]))
        self.accepted(self.edit(answer="__unmappable__", if_previous=["hy4"]))

    def test_a_new_line_may_be_added(self) -> None:
        self.accepted(self.edit(subject="Hy4 (max)", if_previous=None))

    def test_null_removes_the_line(self) -> None:
        self.file["Hy4 preview"] = ["hy4"]
        self.assertIsNone(self.accepted(self.edit(answer=None, if_previous=["hy4"])).value)

    def test_removing_a_line_that_is_not_there_is_refused(self) -> None:
        self.refused(self.edit(answer=None, if_previous=None), "nothing to remove")

    def test_the_answer_must_be_stated(self) -> None:
        record = self.edit()
        del record["answer"]
        self.refused(record, "'answer' is required")

    def test_no_change_is_refused(self) -> None:
        self.file["Hy4 preview"] = ["hy4"]
        self.refused(self.edit(if_previous=["hy4"]), "already maps")

    def test_staleness_is_checked(self) -> None:
        self.file["Hy4 preview"] = ["glm-5-3"]
        self.refused(self.edit(), "stale")
        record = self.edit()
        del record["if_previous"]
        self.refused(record, "'if_previous' is required")

    def test_markers_are_not_answers(self) -> None:
        self.refused(self.edit(answer="__pending__"), "parking marker")
        self.refused(self.edit(answer="__closed_weights__"), "__unmappable__ to decline")

    def test_the_answer_comes_from_the_routes_list(self) -> None:
        self.refused(self.edit(answer="not-a-model"), "not one of the known models")

    def test_a_model_the_batch_creates_is_an_answer(self) -> None:
        answers, failures = self.validate([
            self.edit(answer="hy4-max", subject="Hy4 max", if_previous=None),
            {"kind": "model-create", "name": "hy4-max"},
        ])
        self.assertEqual([str(f) for f in failures], [])

    def test_an_exact_name_cannot_go_to_another_model(self) -> None:
        """test_propose.py would fail on it; refused up front instead."""
        self.refused(self.edit(subject="GLM 5.3", if_previous=None), "normalizes to exactly 'glm-5-3'")
        # Parking it is fine: a sentinel is a decision about the row, not a match.
        self.accepted(self.edit(subject="GLM 5.3", answer="__unmappable__", if_previous=None))

    def test_a_benchmark_label_can_be_edited(self) -> None:
        """The Benchmarks tab's edits: a card label onto a column."""
        self.assertEqual(self.accepted(self.edit(route=HF, subject="Humanity's Last Exam",
                                                 answer="hle", if_previous=None)).value, "hle")
        self.refused(self.edit(route=HF, subject="Humanity's Last Exam", answer="hy4", if_previous=None),
                     "not one of the known benchmarks")

    def test_an_exact_label_cannot_go_to_another_column(self) -> None:
        self.refused(self.edit(route=HF, subject="HLE", answer="swe_bench_verified", if_previous=None),
                     "normalizes to exactly 'hle'")

    def test_a_column_the_batch_creates_is_an_answer(self) -> None:
        _answers_out, failures = self.validate([
            self.edit(route=HF, subject="New Bench (pass@1)", answer="new_bench", if_previous=None),
            {"kind": "benchmark-create", "key": "new_bench", "fields": {
                "name": "New Bench", "short_name": "NB", "category": "Knowledge",
                "description": "A new benchmark.", "settings": ["pass@1"],
                "urls": ["https://example.com"]}},
        ])
        self.assertEqual([str(f) for f in failures], [])

    def test_rejections_stay_with_model_routes(self) -> None:
        self.refused(self.rejection(route=HF, add=["hle"]), "only models and aa-slugs")

    def test_the_route_is_a_table_key(self) -> None:
        self.refused(self.edit(route="../etc/passwd"), "unknown route")
        self.refused(self.edit(path="/tmp/x"), "is not accepted")

    def test_a_source_name_is_one_line(self) -> None:
        self.refused(self.edit(subject=" padded", if_previous=None), "no surrounding spaces")
        self.refused(self.edit(subject="a\nb", if_previous=None), "control characters")
        self.refused(self.edit(subject="x" * 201, if_previous=None), "at most 200")

    def test_the_aa_file_is_keyed_by_a_model(self) -> None:
        aa = _answers.AA_ROUTE
        self.accepted(self.edit(route=aa, subject="hy4", answer="aa-one", if_previous=None))
        self.refused(self.edit(route=aa, subject="nope", answer="aa-one", if_previous=None),
                     "not a model in llm.json")
        self.refused(self.edit(route=aa, subject="hy4", answer="hy4", if_previous=None),
                     "not one of the known aa-slugs")

    def test_one_line_is_answered_once_per_batch(self) -> None:
        queue = {(TBENCH, "mapping", "Hy4 preview")}
        _answers_out, failures = _answers.validate(
            [
                {"kind": MAPPING, "route": TBENCH, "route_kind": "mapping",
                 "subject": "Hy4 preview", "answer": "hy4", "if_previous": None},
                self.edit(if_previous=None),
            ],
            llm_path=self.llm, universes=UNIVERSES, queue=queue,
        )
        self.assertEqual([f.index for f in failures], [1])
        self.assertIn("answered twice", failures[0].message)

    def test_an_edit_onto_a_model_the_batch_renames_is_refused(self) -> None:
        self.file["Hy4 preview"] = ["__unmappable__"]
        _answers_out, failures = self.validate([
            self.edit(),
            {"kind": MODEL_RENAME, "name": "hy4", "new_name": "hy4-final"},
        ])
        self.assertTrue(any("also changed by a model-rename" in f.message for f in failures))


class TestRejectionEdit(EditTestCase):
    def test_a_candidate_can_be_offered_again(self) -> None:
        self.rejected = {"glm-5-3", "hy4"}
        answer = self.accepted(self.rejection(remove=["hy4"]))
        self.assertEqual(answer.value, {"add": [], "remove": ["hy4"]})

    def test_a_candidate_can_be_ruled_out(self) -> None:
        self.assertEqual(self.accepted(self.rejection(add=["hy4"])).value["add"], ["hy4"])

    def test_removing_what_is_not_there_is_stale(self) -> None:
        self.refused(self.rejection(remove=["hy4"]), "stale")

    def test_adding_twice_is_refused(self) -> None:
        self.rejected = {"hy4"}
        self.refused(self.rejection(add=["hy4"]), "already ruled out")

    def test_an_empty_edit_is_refused(self) -> None:
        self.refused(self.rejection(), "nothing to change")

    def test_added_candidates_come_from_the_list(self) -> None:
        self.refused(self.rejection(add=["nope"]), "not among the known models")
        self.refused(self.rejection(add=["__unmappable__"]), "sentinel")

    def test_one_name_cannot_be_added_and_removed(self) -> None:
        self.rejected = {"hy4"}
        self.refused(self.rejection(add=["hy4"], remove=["hy4"]), "both added and removed")


class TestApply(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def test_remove_keeps_a_sorted_file_sorted(self) -> None:
        path = self.tmp / "x-mapping.json"
        path.write_text(json.dumps({"a": "1", "b": "2", "c": "3"}, indent=2) + "\n", encoding="utf-8")
        _answers._remove_line(path, "b")
        self.assertEqual(path.read_text(encoding="utf-8"), json.dumps({"a": "1", "c": "3"}, indent=2) + "\n")

    def test_remove_keeps_an_unsorted_file_in_its_order(self) -> None:
        path = self.tmp / "x-mapping.json"
        path.write_text(json.dumps({"z": "1", "b": "2", "a": "3"}, indent=2) + "\n", encoding="utf-8")
        _answers._remove_line(path, "b")
        self.assertEqual(list(json.loads(path.read_text(encoding="utf-8"))), ["z", "a"])

    def test_remove_refuses_a_missing_key(self) -> None:
        path = self.tmp / "x-mapping.json"
        path.write_text("{}\n", encoding="utf-8")
        with self.assertRaises(AnswerError):
            _answers._remove_line(path, "b")

    def test_an_edit_goes_through_the_routes_writer(self) -> None:
        written = []
        with mock.patch.object(_answers, "writer_for", return_value=lambda k, v: written.append((k, v))):
            _answers._apply_one(
                Answer(0, MAPPING_EDIT, "Hy4 preview", route=TBENCH, route_kind="*", value="hy4"),
                self.tmp / "llm.json",
            )
        self.assertEqual(written, [("Hy4 preview", "hy4")])

    def test_a_removal_takes_the_line_out(self) -> None:
        path = self.tmp / "x-mapping.json"
        path.write_text(json.dumps({"Hy4 preview": "hy4"}, indent=2) + "\n", encoding="utf-8")
        with mock.patch.object(_answers, "mapping_path", return_value=path):
            _answers._apply_one(
                Answer(0, MAPPING_EDIT, "Hy4 preview", route=TBENCH, route_kind="*", value=None),
                self.tmp / "llm.json",
            )
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {})

    def test_the_edited_file_is_in_the_rollback_snapshot(self) -> None:
        paths = _answers.touchable_paths(
            [Answer(0, MAPPING_EDIT, "x", route=TBENCH, route_kind="*", value=None),
             Answer(1, REJECTION_EDIT, "y", route=TBENCH, route_kind="*", value={"add": [], "remove": []})],
            self.tmp / "llm.json",
        )
        self.assertIn(_answers.mapping_path(_answers.resolve_route(TBENCH, "*")), paths)
        self.assertIn(_rejections.REJECTIONS, paths)

    def test_rejection_edit_adds_removes_and_drops_an_empty_name(self) -> None:
        path = self.tmp / "rejected-candidates.json"
        path.write_text(json.dumps({TBENCH: {"Some Model": ["a", "b"]}}), encoding="utf-8")
        with mock.patch.object(_rejections, "freeze_decisions", return_value=False):
            _rejections.edit(TBENCH, "Some Model", add=["c"], remove=["a"], path=path)
            self.assertEqual(_rejections.load(path)[TBENCH]["Some Model"], {"b", "c"})
            _rejections.edit(TBENCH, "Some Model", remove=["b", "c"], path=path)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {})


class TestReceived(unittest.TestCase):
    """The pills on the Benchmarks tab: which model and score a label delivers."""

    HF_PAGE = "https://huggingface.co/org/m"

    def doc(self, source, value=50.0):
        return {
            "benchmarks": {"hle": {}},
            "models": [{"name": "m", "scores": {"hle": value}, "scores_source": {"hle": source}}],
        }

    def files(self, *labels):
        return {_mapping_index.HF_LABEL_ROUTE: {label: "hle" for label in labels} | {"Params": "__unmappable__"}}

    def got(self, doc, files, **kwargs):
        return _mapping_index.received(doc, files, **kwargs).get(_mapping_index.HF_LABEL_ROUTE, {})

    def test_a_lone_label_gets_the_score(self) -> None:
        self.assertEqual(self.got(self.doc(self.HF_PAGE), self.files("HLE")), {"HLE": [["m", 50.0, 0]]})

    def test_a_score_another_source_overrode_is_not_the_labels(self) -> None:
        self.assertEqual(self.got(self.doc("https://artificialanalysis.ai/models/m"), self.files("HLE")), {})
        self.assertEqual(self.got(self.doc("https://huggingface.co/datasets/x/y"), self.files("HLE")), {})

    def test_the_crawl_says_which_label_printed_it(self) -> None:
        got = self.got(self.doc(self.HF_PAGE), self.files("HLE", "HLE (text)"),
                       crawl={"m": {"HLE": 50.0, "HLE (text)": 41.0}})
        self.assertEqual(got, {"HLE": [["m", 50.0, 0]]})

    def test_one_number_under_two_headings_is_both_labels(self) -> None:
        got = self.got(self.doc(self.HF_PAGE), self.files("HLE", "HLE (text)"),
                       crawl={"m": {"HLE": 50.0, "HLE (text)": 50.0}})
        self.assertEqual(got, {"HLE": [["m", 50.0, 0]], "HLE (text)": [["m", 50.0, 0]]})

    def test_without_a_crawl_the_last_index_decides(self) -> None:
        prior = {_mapping_index.HF_LABEL_ROUTE: {"HLE (text)": [["m", 50.0, 0]]}}
        got = self.got(self.doc(self.HF_PAGE), self.files("HLE", "HLE (text)"), prior=prior)
        self.assertEqual(got, {"HLE (text)": [["m", 50.0, 0]]})

    def test_a_number_the_card_no_longer_prints_is_no_labels(self) -> None:
        got = self.got(self.doc(self.HF_PAGE), self.files("HLE", "HLE (text)"), crawl={"m": {"MMLU": 80.0}})
        self.assertEqual(got, {})

    def test_with_nothing_to_go_on_the_score_is_shared(self) -> None:
        got = self.got(self.doc(self.HF_PAGE), self.files("HLE", "HLE (text)"))
        self.assertEqual(got, {"HLE": [["m", 50.0, 2]], "HLE (text)": [["m", 50.0, 2]]})


class TestMappingIndex(unittest.TestCase):
    def test_every_model_route_says_which_pages_it_writes(self) -> None:
        """A route missing here would show on the page as supplying nothing."""
        ids = {r["id"] for r in _mapping_index.covered_routes()}
        self.assertEqual(ids, set(_mapping_index.route_pages()))

    def test_only_model_and_aa_routes_are_covered(self) -> None:
        universes = {r["universe"] for r in _mapping_index.covered_routes()}
        self.assertEqual(universes, {propose.MODELS, propose.AA_SLUGS})

    def test_attribution_matches_on_a_path_boundary(self) -> None:
        table = _mapping_index._prefix_table(_mapping_index.route_pages())
        attribute = lambda url: _mapping_index.attribute(url, table)
        self.assertEqual(attribute("https://openrouter.ai/qwen/qwen3"), "update_openrouter_mapping.py *")
        self.assertEqual(attribute("https://openrouter.ai.evil.example/x"), _mapping_index.OTHER)
        self.assertEqual(attribute(None), _mapping_index.OTHER)
        self.assertEqual(attribute("https://huggingface.co/tencent/Hy4"), _mapping_index.CARD)
        # The Coding Agent Index lives under AA's host and keeps its own credit.
        self.assertEqual(attribute("https://artificialanalysis.ai/agents/coding-agents"),
                         "update_aa_coding_agents_mapping.py *")
        self.assertEqual(attribute("https://artificialanalysis.ai/models/tencent-hy4"),
                         "update_artificialanalysis_mapping.py *")

    def test_contributions_skip_derived_columns_and_empty_scores(self) -> None:
        doc = {
            "benchmarks": {"hle": {}, "coding_index": {"derived": True}, "mmlu_pro": {}},
            "models": [{
                "name": "m",
                "scores": {"hle": 10, "coding_index": 5, "mmlu_pro": None},
                "scores_source": {"hle": "https://openrouter.ai/m", "coding_index": "https://github.com/x"},
                "vram": {"fp16": 10},
            }],
        }
        self.assertEqual(_mapping_index.contributions(doc), {"m": {
            "update_openrouter_mapping.py *": ["hle"],
            "update_spheron_mapping.py *": [_mapping_index.VRAM],
        }})

    def test_parked_names_that_look_like_a_model_are_offered_back(self) -> None:
        """A cleanup parks the preview's names; the release's card offers them."""
        routes = [{"id": "r *", "universe": propose.MODELS}, {"id": "b *", "universe": propose.BENCHMARKS}]
        files = {
            "r *": {"tencent/hy4-preview": "__unmappable__", "Hy4": "hy4", "hy40": "__unmappable__",
                    "GLM 5.3 preview": "__unmappable__", "Old Name": "__closed_weights__"},
            "b *": {"hy4 preview": "__unmappable__"},
        }
        got = _mapping_index.parked(["hy4", "glm-5", "renamed"], routes, files,
                                    former={"renamed": ["old-name"]})
        self.assertEqual(got, {
            "hy4": [["r *", "tencent/hy4-preview"]],
            "renamed": [["r *", "Old Name"]],
        })

    def test_the_published_document_has_every_part_the_page_reads(self) -> None:
        doc = _mapping_index.build()
        self.assertEqual(set(doc), {"generated", "routes", "files", "rejections", "parked",
                                    "contributions", "received"})
        self.assertEqual(set(doc["files"]), {r["id"] for r in doc["routes"]})
        self.assertIn(_mapping_index.HF_LABEL_ROUTE, doc["files"])
        self.assertTrue(any(doc["files"].values()), "every mapping file read empty")


if __name__ == "__main__":
    unittest.main()
