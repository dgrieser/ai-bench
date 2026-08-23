#!/usr/bin/env python3
"""Tests for the new-model decisions. Run with ./test_new_models.py

The load-bearing tests here are test_ignored_model_is_removed_and_dismissed and
test_decisions_are_applied_under_collect_mode: together they are the reason the
proposal PR can offer "ignore" at all. If the first breaks, flipping a line to
__ignored__ silently keeps the model; if the second breaks, the decision is only
ever applied by a local interactive run -- and the unattended run that follows
the merge is the only run that ever sees it.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import _new_models
import _prompts
from _new_models import ADDED, IGNORED


def model(name: str, scores: dict | None = None) -> dict:
    return {"name": name, "scores": scores or {"swe_bench_verified": None}}


class DecisionsCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.llm = self.tmp / "llm.json"
        self.decisions = self.tmp / "check_new-decisions.json"
        self.dismissed = self.tmp / "check_new-dismissed.json"
        self.write_llm(["keep-me", "drop-me"])

    def write_llm(self, names: list[str]) -> None:
        self.llm.write_text(
            json.dumps({"models": [model(n) for n in names]}, indent=2) + "\n", encoding="utf-8"
        )

    def write_decisions(self, decisions: dict[str, str]) -> None:
        self.decisions.write_text(json.dumps(decisions, indent=2) + "\n", encoding="utf-8")

    def apply(self) -> list[str]:
        return _new_models.apply_decisions(self.llm, self.decisions, self.dismissed)

    def names(self) -> list[str]:
        return [m["name"] for m in json.loads(self.llm.read_text(encoding="utf-8"))["models"]]


class TestApplyDecisions(DecisionsCase):
    def test_nothing_recorded_is_a_no_op(self) -> None:
        self.assertEqual(_new_models.apply_decisions(self.llm, self.decisions, self.dismissed), [])
        self.assertEqual(self.names(), ["keep-me", "drop-me"])

    def test_ignored_model_is_removed_and_dismissed(self) -> None:
        self.write_decisions({"drop-me": IGNORED})
        log = self.apply()

        self.assertEqual(self.names(), ["keep-me"])
        self.assertEqual(_new_models.load_dismissed(self.dismissed), {"drop-me"})
        # The question is answered: it must not come back on the next run.
        self.assertEqual(_new_models.load_decisions(self.decisions), {})
        self.assertTrue(any("removed" in line for line in log))

    def test_added_model_is_kept_and_the_entry_consumed(self) -> None:
        self.write_decisions({"keep-me": ADDED})
        log = self.apply()

        self.assertEqual(self.names(), ["keep-me", "drop-me"])
        self.assertEqual(_new_models.load_dismissed(self.dismissed), set())
        self.assertEqual(_new_models.load_decisions(self.decisions), {})
        self.assertTrue(any("kept" in line for line in log))

    def test_both_answers_in_one_file(self) -> None:
        self.write_decisions({"keep-me": ADDED, "drop-me": IGNORED})
        self.apply()
        self.assertEqual(self.names(), ["keep-me"])
        self.assertEqual(_new_models.load_dismissed(self.dismissed), {"drop-me"})

    def test_ignoring_a_model_that_is_already_gone_still_dismisses_it(self) -> None:
        # Hand-removed from llm.json before the run got to the decision.
        self.write_llm(["keep-me"])
        self.write_decisions({"drop-me": IGNORED})
        self.apply()
        self.assertEqual(_new_models.load_dismissed(self.dismissed), {"drop-me"})
        self.assertEqual(_new_models.load_decisions(self.decisions), {})

    def test_a_scored_model_is_still_removed_and_the_loss_reported(self) -> None:
        # A refresh beat the decision to it. The decision is the newer answer and
        # wins, but the log has to say what went with it.
        self.llm.write_text(
            json.dumps({"models": [model("drop-me", {"swe_bench_verified": 42.0})]}, indent=2),
            encoding="utf-8",
        )
        self.write_decisions({"drop-me": IGNORED})
        log = self.apply()
        self.assertEqual(self.names(), [])
        self.assertTrue(any("dropped 1 score(s)" in line for line in log))

    def test_an_unknown_value_is_left_for_a_person(self) -> None:
        self.write_decisions({"drop-me": "maybe?"})
        log = self.apply()

        self.assertEqual(self.names(), ["keep-me", "drop-me"], "must not act on a typo")
        self.assertEqual(_new_models.load_decisions(self.decisions), {"drop-me": "maybe?"})
        self.assertEqual(_new_models.load_dismissed(self.dismissed), set())
        self.assertTrue(any("unknown decision" in line for line in log))

    def test_decisions_are_applied_under_collect_mode(self) -> None:
        """The unattended run is the only one that ever sees a merged decision.

        Collect mode freezes the answers *it* would invent; these were written by
        a person in a PR. Freezing them would re-add an ignored model forever.
        """
        saved = os.environ.get(_prompts.ENV_VAR)
        os.environ[_prompts.ENV_VAR] = str(self.tmp / "queue.jsonl")
        try:
            self.assertTrue(_prompts.freeze_decisions())
            self.write_decisions({"drop-me": IGNORED})
            self.apply()
        finally:
            os.environ.pop(_prompts.ENV_VAR, None)
            if saved is not None:
                os.environ[_prompts.ENV_VAR] = saved

        self.assertEqual(self.names(), ["keep-me"])
        self.assertEqual(_new_models.load_dismissed(self.dismissed), {"drop-me"})

    def test_llm_json_keeps_its_formatting(self) -> None:
        self.write_decisions({"drop-me": IGNORED})
        self.apply()
        text = self.llm.read_text(encoding="utf-8")
        self.assertTrue(text.endswith("}\n"), "two-space indent, trailing newline")
        self.assertIn('\n  "models": [', text)


class TestDismiss(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "check_new-dismissed.json"

    def test_dismiss_is_frozen_under_collect_mode(self) -> None:
        # The interactive [n] answer: collect mode queues the candidate instead
        # of offering it, so nothing may be recorded on its behalf.
        saved = os.environ.get(_prompts.ENV_VAR)
        os.environ[_prompts.ENV_VAR] = str(self.tmp / "queue.jsonl")
        try:
            self.assertFalse(_new_models.dismiss("some-model", self.path))
            self.assertEqual(_new_models.load_dismissed(self.path), set())
            self.assertTrue(_new_models.dismiss("some-model", self.path, force=True))
        finally:
            os.environ.pop(_prompts.ENV_VAR, None)
            if saved is not None:
                os.environ[_prompts.ENV_VAR] = saved
        self.assertEqual(_new_models.load_dismissed(self.path), {"some-model"})

    def test_dismiss_is_sorted_and_idempotent(self) -> None:
        _new_models.dismiss("b-model", self.path)
        _new_models.dismiss("a-model", self.path)
        self.assertFalse(_new_models.dismiss("a-model", self.path))
        self.assertEqual(
            json.loads(self.path.read_text(encoding="utf-8")), ["a-model", "b-model"]
        )


class TestRecording(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "check_new-decisions.json"

    def test_record_writes_an_added_line_and_finds_it(self) -> None:
        _new_models.record_proposed("new-model-9b", self.path)
        self.assertEqual(_new_models.load_decisions(self.path), {"new-model-9b": ADDED})
        self.assertEqual(_new_models.find_line("new-model-9b", self.path), 2)
        self.assertIsNone(_new_models.find_line("absent", self.path))

    def test_record_never_overwrites_a_reviewer_answer(self) -> None:
        self.path.write_text(json.dumps({"new-model-9b": IGNORED}, indent=2), encoding="utf-8")
        _new_models.record_proposed("new-model-9b", self.path)
        self.assertEqual(_new_models.load_decisions(self.path), {"new-model-9b": IGNORED})

    def test_find_line_is_not_fooled_by_a_substring_key(self) -> None:
        self.path.write_text(
            '{\n  "glm-5-1-air": "%s",\n  "glm-5-1": "%s"\n}\n' % (ADDED, ADDED), encoding="utf-8"
        )
        self.assertEqual(_new_models.find_line("glm-5-1", self.path), 3)

    def test_the_committed_file_is_a_json_object(self) -> None:
        # It ships in the tree so the format stays documented and the PR diff
        # stays a one-line edit rather than a new file.
        raw = json.loads(_new_models.DECISIONS_FILE.read_text(encoding="utf-8"))
        self.assertIsInstance(raw, dict)
        self.assertTrue(all(v in (ADDED, IGNORED) for v in raw.values()), raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
