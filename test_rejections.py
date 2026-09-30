#!/usr/bin/env python3
"""Tests for "none of these": ruling candidates out without answering.

Run with ./test_rejections.py. Offline. What is guarded: the record is held to
the same gate as a mapping (a queued question, candidates from the route's own
list); applying it writes the rejection file and never the mapping file; and
the queue then drops those candidates, hides a question with nothing left, and
asks again the moment a new candidate appears.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _answers
import _rejections
import pending_prompts
import propose
from test_answers import UNIVERSES, AnswersTestCase, TBENCH

SKIP = _answers.CANDIDATES_SKIP


def skip(**overrides) -> dict:
    record = {"kind": SKIP, "route": TBENCH, "route_kind": "mapping",
              "subject": "Some Model", "answer": ["glm-5-3"]}
    record.update(overrides)
    return record


class TestValidation(AnswersTestCase):
    def test_a_skip_names_a_queued_question_and_known_candidates(self) -> None:
        answer = self.accepted(skip())
        self.assertEqual((answer.kind, answer.route, answer.value), (SKIP, TBENCH, ["glm-5-3"]))
        self.refused(skip(subject="Never Asked"), "not a question")
        self.refused(skip(answer=["no-such-model"]), "not among the known")
        self.refused(skip(answer=[]), "non-empty list")
        self.refused(skip(answer="glm-5-3"), "non-empty list")
        self.refused(skip(answer=["__unmappable__"]), "sentinel")
        self.refused(skip(route="../../etc/passwd"), "unknown route")

    def test_an_aa_question_is_checked_against_aa_slugs(self) -> None:
        record = skip(route=_answers.AA_ROUTE, route_kind="aa-mapping",
                      subject="devstral-2", answer=["aa-one"])
        self.accepted(record)
        self.refused({**record, "answer": ["glm-5-3"]}, "not among the known")
        self.assertTrue(_answers.needs_aa_slugs([record]))

    def test_an_empty_universe_refuses_rather_than_trusting_the_record(self) -> None:
        self.refused(skip(), "list is empty", universes={**UNIVERSES, propose.MODELS: []})


class TestApply(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.file = self.tmp / "rejected-candidates.json"
        patcher = mock.patch.object(_rejections, "REJECTIONS", self.file)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_rejections_accumulate_per_route_and_name(self) -> None:
        _rejections.add(TBENCH, "Some Model", ["b", "a"], path=self.file)
        _rejections.add(TBENCH, "Some Model", ["a", "c"], path=self.file)
        _rejections.add("update_vals_mapping.py", "x/y", ["z"], path=self.file)
        self.assertEqual(_rejections.rejected(TBENCH, "Some Model", path=self.file), {"a", "b", "c"})
        doc = json.loads(self.file.read_text(encoding="utf-8"))
        self.assertEqual(doc[TBENCH]["Some Model"], ["a", "b", "c"])
        self.assertEqual(_rejections.rejected(TBENCH, "Other", path=self.file), set())

    def test_applying_writes_the_rejection_and_no_mapping(self) -> None:
        answer = _answers.Answer(index=0, kind=SKIP, subject="Some Model", route=TBENCH,
                                 route_kind="mapping", value=["glm-5-3"])
        route = _answers.resolve_route(TBENCH, "mapping")
        mapping_file = _answers.mapping_path(route)
        before = mapping_file.read_bytes()
        log = _answers._apply_one(answer, self.tmp / "llm.json")
        self.assertEqual(mapping_file.read_bytes(), before)
        self.assertIn("none of glm-5-3", log[0])
        self.assertEqual(_rejections.rejected(TBENCH, "Some Model", path=self.file), {"glm-5-3"})

    def test_the_aa_route_keeps_its_own_file(self) -> None:
        with mock.patch("_artificialanalysis_mapping.add_ignored_aa_suggestions") as aa:
            written = _rejections.add(_answers.AA_ROUTE, "devstral-2", ["aa-one"], path=self.file)
        aa.assert_called_once_with("devstral-2", ["aa-one"])
        self.assertNotEqual(written, self.file)
        self.assertFalse(self.file.exists())

    def test_the_file_is_not_mistaken_for_a_source_mapping(self) -> None:
        self.assertNotIn("mapping", _rejections.REJECTIONS.name)


class TestQueue(unittest.TestCase):
    """What pending_prompts.render_json publishes once candidates are ruled out."""

    def render(self, ruled_out: set[str], models: list[str]) -> list[dict]:
        entry = {"command": TBENCH, "kind": "mapping", "subject": "minimax-m3-1-flash-preview",
                 "question": "?", "candidates": [], "default": None, "note": None}
        universes = {propose.MODELS: models, propose.BENCHMARKS: [], propose.AA_SLUGS: []}
        with mock.patch.object(propose, "build_universes", return_value=universes), \
                mock.patch.object(_rejections, "rejected", return_value=ruled_out), \
                mock.patch.object(pending_prompts, "_answers_current_value", return_value=None):
            doc = json.loads(pending_prompts.render_json([entry], Path("llm.json"), skip_aa=True))
        return doc["questions"]

    def test_nothing_ruled_out_changes_nothing(self) -> None:
        (question,) = self.render(set(), ["minimax-m3", "minimax-m2-1"])
        self.assertEqual({c["option"] for c in question["candidates"]}, {"minimax-m3", "minimax-m2-1"})
        self.assertNotIn("ruled_out", question)

    def test_ruled_out_candidates_are_dropped(self) -> None:
        (question,) = self.render({"minimax-m3"}, ["minimax-m3", "minimax-m2-1"])
        self.assertEqual([c["option"] for c in question["candidates"]], ["minimax-m2-1"])
        self.assertEqual(question["ruled_out"], 1)

    def test_a_question_with_nothing_left_leaves_the_queue(self) -> None:
        self.assertEqual(self.render({"minimax-m3", "minimax-m2-1"}, ["minimax-m3", "minimax-m2-1"]), [])

    def test_a_new_candidate_brings_it_back_with_only_the_new_one(self) -> None:
        (question,) = self.render({"minimax-m3", "minimax-m2-1"},
                                  ["minimax-m3", "minimax-m2-1", "minimax-m3-1-flash"])
        self.assertEqual([c["option"] for c in question["candidates"]], ["minimax-m3-1-flash"])


if __name__ == "__main__":
    unittest.main()
