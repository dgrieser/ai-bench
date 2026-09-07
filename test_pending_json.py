#!/usr/bin/env python3
"""Tests for the queue as data (`pending_prompts.py --format json`).

Run with ./test_pending_json.py

This file is committed on every refresh, so its whole contract is that it
changes only when the questions change. A timestamp, a run id, or a candidate
list that reshuffles under a growing universe would each rewrite it on runs
where nothing was asked or answered -- burying the real diffs and defeating the
workflow's "no data changed" brake.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pending_prompts
import propose

ENTRIES = [
    {
        "command": "./update_tbench_mapping.py -w",
        "kind": "mapping",
        "subject": "GLM-5.3",
        "question": "Which llm.json model is Terminal-Bench 'GLM-5.3'?",
        "candidates": ["glm-5-3"],
        "default": None,
        "note": None,
    },
    {
        "command": "./check_new.py",
        "kind": "new-model",
        "subject": "fresh-slug",
        "question": "Add newly released model 'fresh-slug'?",
        "candidates": [],
        "default": None,
        "note": "released 2026-09-01",
    },
    {
        "command": "./update_llmstats_mapping.py -w",
        "kind": "llmstats-benchmark",
        "subject": "swe_bench_verified",
        "question": "Which benchmark is llm-stats 'swe_bench_verified'?",
        "candidates": [],
        "default": None,
        "note": None,
    },
]

UNIVERSES = {
    propose.MODELS: ["glm-5-3", "glm-5-2", "devstral-2"],
    propose.BENCHMARKS: ["swe_bench_verified", "hle"],
    propose.AA_SLUGS: [],
}


class PendingJsonTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.llm = self.tmp / "llm.json"
        self.llm.write_text("{}", encoding="utf-8")
        universes = mock.patch.object(propose, "build_universes", return_value=UNIVERSES)
        universes.start()
        self.addCleanup(universes.stop)
        # What a mapping file says today is real state; these tests are about
        # the shape of the render, not about the files.
        current = mock.patch.object(pending_prompts, "_answers_current_value", return_value=None)
        current.start()
        self.addCleanup(current.stop)

    def render(self, entries) -> str:
        return pending_prompts.render_json(entries, self.llm, skip_aa=True)

    def questions(self, entries) -> list[dict]:
        return json.loads(self.render(entries))["questions"]


class TestDeterminism(PendingJsonTestCase):
    def test_the_input_order_does_not_show(self) -> None:
        self.assertEqual(self.render(ENTRIES), self.render(list(reversed(ENTRIES))))

    def test_rendering_twice_is_byte_identical(self) -> None:
        self.assertEqual(self.render(ENTRIES), self.render(ENTRIES))

    def test_nothing_dated_leaks_in(self) -> None:
        """A timestamp would rewrite the file on every run."""
        text = self.render(ENTRIES)
        self.assertNotRegex(text, r"\d{4}-\d{2}-\d{2}T")
        for word in ("generated", "timestamp", "run_id", "updated_at"):
            self.assertNotIn(word, text.lower())

    def test_the_shape_matches_the_repos_other_json(self) -> None:
        text = self.render(ENTRIES)
        self.assertTrue(text.endswith("\n"))
        self.assertIn('\n  "questions"', text)
        for question in json.loads(text)["questions"]:
            keys = list(question)
            self.assertEqual(keys, sorted(keys), "each question's keys must be sorted")

    def test_an_empty_queue_still_renders(self) -> None:
        self.assertEqual(json.loads(self.render([]))["questions"], [])


class TestContent(PendingJsonTestCase):
    def test_each_question_carries_what_an_answer_needs(self) -> None:
        by_subject = {q["subject"]: q for q in self.questions(ENTRIES)}
        tbench = by_subject["GLM-5.3"]
        self.assertEqual(tbench["route"], "update_tbench_mapping.py")
        self.assertEqual(tbench["route_kind"], "mapping")
        self.assertIn("if_previous", tbench)

    def test_the_two_llmstats_files_stay_distinguishable(self) -> None:
        """One script owns two files; the kind is what tells them apart."""
        question = next(q for q in self.questions(ENTRIES) if q["subject"] == "swe_bench_verified")
        self.assertEqual(question["route"], "update_llmstats_mapping.py")
        self.assertEqual(question["route_kind"], "llmstats-benchmark")

    def test_candidates_are_graded_and_carry_their_reason(self) -> None:
        question = next(q for q in self.questions(ENTRIES) if q["subject"] == "GLM-5.3")
        best = question["candidates"][0]
        self.assertEqual(best["option"], "glm-5-3")
        self.assertEqual(best["confidence"], "exact")
        self.assertTrue(best["reason"])

    def test_candidates_are_capped_so_the_tail_cannot_reshuffle(self) -> None:
        many = dict(ENTRIES[0], subject="glm")
        with mock.patch.object(
            propose,
            "build_universes",
            return_value=dict(UNIVERSES, models=[f"glm-5-{i}" for i in range(40)]),
        ):
            question = self.questions([many])[0]
        self.assertLessEqual(len(question["candidates"]), pending_prompts.MAX_CANDIDATES)

    def test_no_universe_is_embedded(self) -> None:
        """Hundreds of model names and thousands of AA slugs, each on its own
        schedule, would rewrite this file on runs where nothing was asked -- and
        empty it whenever Artificial Analysis was unreachable."""
        doc = json.loads(self.render(ENTRIES))
        self.assertEqual(set(doc), {"questions"})
        self.assertNotIn("devstral-2", self.render(ENTRIES))

    def test_a_question_with_no_route_still_appears(self) -> None:
        """A source nobody has routed yet is still a question a person can see."""
        orphan = dict(ENTRIES[0], command="./update_unrouted_mapping.py -w")
        question = self.questions([orphan])[0]
        self.assertEqual(question["route"], "update_unrouted_mapping.py")
        self.assertEqual(question["subject"], "GLM-5.3")


class TestOutput(PendingJsonTestCase):
    def test_out_writes_through_the_cli_and_creates_its_directory(self) -> None:
        report = self.tmp / "queue.jsonl"
        report.write_text(
            "".join(json.dumps(e) + "\n" for e in ENTRIES), encoding="utf-8"
        )
        target = self.tmp / "_pending" / "pending.json"
        argv = [
            str(report),
            "--format=json",
            "--skip-aa",
            f"--out={target}",
            f"--llm-json={self.llm}",
        ]
        with mock.patch("sys.argv", ["pending_prompts.py", *argv]):
            self.assertEqual(pending_prompts.main(), 0)
        self.assertTrue(target.exists())
        doc = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual([q["subject"] for q in doc["questions"]],
                         ["fresh-slug", "swe_bench_verified", "GLM-5.3"])


if __name__ == "__main__":
    unittest.main()
