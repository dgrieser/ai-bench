#!/usr/bin/env python3
"""Tests for the weight statuses: open, upcoming, closed. Run with ./test_weights.py

_openness.py turns every source's own label into one of three statuses, pools
them into an index, and lets a person overrule all of it in
weights-overrides.json. What has to hold:

  * an *upcoming* model (weights announced, not out) is never skipped as
    closed -- not by a source's label, not by the pooled index;
  * pooling is order-free and closed only when every source agrees;
  * a person's override beats every source, matches across spellings, and
    takes back a name an earlier run skipped as __closed_weights__;
  * the admin page's weights-override record is validated and applied like
    every other answer, and adding a closed release as open records it as
    upcoming.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _answers
import _openness as o
import check_new
import fetch_epoch
import fetch_openrouter
import pending_prompts


class OverridesTestCase(unittest.TestCase):
    """Points the module at a scratch overrides file and a fixed index."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.file = self.tmp / "weights-overrides.json"
        for name, value in (
            ("OVERRIDES_FILE", self.file),
            ("_OVERRIDES", None),
            ("_INDEX", {}),
            ("_OFFLINE", {}),
        ):
            patcher = mock.patch.object(o, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def index(self, **statuses: str) -> None:
        o._INDEX = {o.normalize(k.replace("_", " ")): v for k, v in statuses.items()}
        o._OFFLINE = o._INDEX


class TestParsers(unittest.TestCase):
    def test_licences(self) -> None:
        self.assertEqual(o.license_status("proprietary"), o.CLOSED)
        self.assertEqual(o.license_status("Proprietary"), o.CLOSED)
        self.assertEqual(o.license_status("apache_2_0"), o.OPEN)
        self.assertEqual(o.license_status("Modified MIT"), o.OPEN)
        self.assertEqual(o.license_status("MIT (coming soon)"), o.UPCOMING)
        self.assertEqual(o.license_status("TBA"), o.UPCOMING)
        # llm-stats' "unknown" is no licence at all, not an open one.
        self.assertIsNone(o.license_status("unknown"))
        self.assertIsNone(o.license_status(None))
        # The boolean stays what it always was, with upcoming as "not known".
        self.assertIs(o.license_open("proprietary"), False)
        self.assertIsNone(o.license_open("weights coming soon"))

    def test_source_types(self) -> None:
        self.assertEqual(o.source_type_status("Open Weight"), o.OPEN)
        self.assertEqual(o.source_type_status("Open-Weights"), o.OPEN)
        self.assertEqual(o.source_type_status("Open-Source"), o.OPEN)
        self.assertEqual(o.source_type_status("Proprietary"), o.CLOSED)
        self.assertEqual(o.source_type_status("Open weights (coming soon)"), o.UPCOMING)
        self.assertIsNone(o.source_type_status("Pending"))
        self.assertIsNone(o.source_type_status(None))

    def test_epoch_accessibility(self) -> None:
        self.assertEqual(fetch_epoch.weights_status("Open weights (non-commercial)"), o.OPEN)
        self.assertEqual(fetch_epoch.weights_status("API access"), o.CLOSED)
        # "Unreleased" promises nothing: closed, not upcoming.
        self.assertEqual(fetch_epoch.weights_status("Unreleased"), o.CLOSED)
        self.assertEqual(fetch_epoch.weights_status("Open weights (to be released)"), o.UPCOMING)
        self.assertIsNone(fetch_epoch.weights_status(""))

    def test_openrouter_catalogue(self) -> None:
        self.assertEqual(fetch_openrouter.catalogue_weights({"hugging_face_id": "org/m"}), o.OPEN)
        closed = {"hugging_face_id": None,
                  "benchmarks": {"artificial_analysis": {"openness_index": 5.6}}}
        self.assertEqual(fetch_openrouter.catalogue_weights(closed), o.CLOSED)
        # No link and no index: nothing proves it closed.
        self.assertIsNone(fetch_openrouter.catalogue_weights({"hugging_face_id": None}))


class TestPooling(unittest.TestCase):
    def test_most_open_wins_in_any_order(self) -> None:
        self.assertEqual(o.merge(o.CLOSED, o.UPCOMING), o.UPCOMING)
        self.assertEqual(o.merge(o.UPCOMING, o.CLOSED), o.UPCOMING)
        self.assertEqual(o.merge(o.UPCOMING, o.OPEN), o.OPEN)
        self.assertEqual(o.merge(None, o.CLOSED), o.CLOSED)
        self.assertIsNone(o.merge(None, None))

    def test_pool_keys_by_normalized_name(self) -> None:
        statuses: dict[str, str] = {}
        o._pool(statuses, "Some-Model-2.5", o.CLOSED)
        o._pool(statuses, "some model 2.5 (high)", o.UPCOMING)
        self.assertEqual(statuses, {"some model 2 5": o.UPCOMING})

    def test_row_status_prefers_the_status(self) -> None:
        self.assertEqual(o.row_status({"weights": o.UPCOMING, "open_weights": None}), o.UPCOMING)
        self.assertEqual(o.row_status({"open_weights": False}), o.CLOSED)
        self.assertIsNone(o.row_status({}))

    def test_open_index_leaves_upcoming_out(self) -> None:
        with mock.patch.object(o, "_INDEX", {"a": o.OPEN, "b": o.UPCOMING, "c": o.CLOSED}):
            self.assertEqual(o.open_index(), {"a": True, "c": False})


class TestOverrides(OverridesTestCase):
    def test_set_matches_across_spellings_and_replaces(self) -> None:
        self.assertTrue(o.set_override("Some Model 2.5", o.UPCOMING))
        self.assertEqual(o.override_for("some-model-2.5-high"), o.UPCOMING)
        self.assertTrue(o.set_override("some-model-2.5", o.OPEN))
        # One line per model, however many spellings set it.
        self.assertEqual(json.loads(self.file.read_text()), {"some-model-2.5": o.OPEN})
        self.assertFalse(o.set_override("Some Model 2.5", o.OPEN))
        self.assertTrue(o.set_override("Some Model 2.5", None))
        self.assertIsNone(o.override_for("Some Model 2.5"))
        with self.assertRaises(ValueError):
            o.set_override("x", "maybe")

    def test_describe_reports_where_a_verdict_came_from(self) -> None:
        self.index(Some_Model=o.CLOSED)
        self.assertEqual(o.describe("Some Model"), (o.CLOSED, "sources"))
        self.assertEqual(o.describe("Some Model", weights=o.OPEN), (o.OPEN, "source"))
        o.set_override("Some Model", o.UPCOMING)
        self.assertEqual(o.describe("Some Model", weights=o.CLOSED), (o.UPCOMING, "override"))
        self.assertEqual(o.describe("Nobody"), (None, None))


class TestClosedSkip(OverridesTestCase):
    def test_upcoming_is_never_skipped(self) -> None:
        self.index(Model_X=o.UPCOMING)
        self.assertFalse(o.is_closed_weights("Model X"))
        self.assertFalse(o.is_closed_weights("Model Y", weights=o.UPCOMING))

    def test_a_source_label_beats_the_index(self) -> None:
        self.index(Model_X=o.OPEN)
        self.assertTrue(o.is_closed_weights("Model X", open_weights=False))
        self.assertTrue(o.is_closed_weights("Model X", weights=o.CLOSED))

    def test_an_override_beats_every_source(self) -> None:
        self.index(Model_X=o.CLOSED)
        o.set_override("Model X", o.UPCOMING)
        self.assertFalse(o.is_closed_weights("Model X", weights=o.CLOSED))
        o.set_override("Model Z", o.CLOSED)
        # Said about this very name, so it holds even where the guard would not.
        self.assertTrue(o.is_closed_weights("Model Z", guard_names=["model-z"]))
        self.assertFalse(o.is_closed_weights("Model Q", open_weights=False, guard_names=["model-q"]))

    def test_an_override_reopens_a_closed_skip(self) -> None:
        import _tbench_mapping

        path = self.tmp / "mapping.json"
        path.write_text(json.dumps({"Model X": o.CLOSED_WEIGHTS, "Model Y": o.CLOSED_WEIGHTS}))
        self.assertEqual(_tbench_mapping.load_reviewed_tbench_names(path), {"Model X", "Model Y"})
        o.set_override("Model X", o.UPCOMING)
        self.assertEqual(_tbench_mapping.load_reviewed_tbench_names(path), {"Model Y"})
        self.assertEqual(
            _tbench_mapping.load_reviewed_tbench_names(path, include_closed=False), set()
        )


class TestNewReleases(OverridesTestCase):
    def test_other_sources_lift_a_closed_release(self) -> None:
        self.index(Fresh_Model=o.UPCOMING)
        release = {"slug": "fresh-model", "name": "Fresh Model", "_closed": True}
        self.assertEqual(check_new.weights_of(release), o.UPCOMING)
        self.assertEqual(check_new.weights_of({"slug": "other", "_closed": True}), o.CLOSED)
        self.assertEqual(check_new.weights_of({"slug": "other"}), o.OPEN)

    def test_an_override_decides(self) -> None:
        o.set_override("fresh-model", o.CLOSED)
        self.assertEqual(check_new.weights_of({"slug": "fresh-model"}), o.CLOSED)


class TestQueue(OverridesTestCase):
    def test_model_questions_carry_their_weights(self) -> None:
        self.index(Upcoming_Model=o.UPCOMING)
        entries = [
            {"command": "./update_tbench_mapping.py -w", "kind": "mapping",
             "subject": "Upcoming Model", "question": "?", "candidates": []},
            {"command": "./update_tbench_mapping.py -w", "kind": "mapping",
             "subject": "Unknown Model", "question": "?", "candidates": []},
        ]
        with mock.patch.object(pending_prompts.propose, "build_universes",
                               return_value={"models": ["glm-5-3"]}):
            doc = json.loads(pending_prompts.render_json(entries, Path("llm.json"), skip_aa=True))
        by_subject = {q["subject"]: q for q in doc["questions"]}
        self.assertEqual(by_subject["Upcoming Model"]["weights"], o.UPCOMING)
        self.assertEqual(by_subject["Upcoming Model"]["weights_from"], "sources")
        self.assertNotIn("weights", by_subject["Unknown Model"])


class TestOverrideAnswers(OverridesTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.llm = self.tmp / "llm.json"
        self.llm.write_text(json.dumps({"benchmarks": {}, "models": []}))

    def check(self, record, queue=frozenset()):
        return _answers.validate([record], llm_path=self.llm, universes={"models": []},
                                 queue=set(queue))

    def test_validated_and_applied(self) -> None:
        record = {"kind": "weights-override", "name": "Model X",
                  "weights": "upcoming", "if_previous": None}
        answers, failures = self.check(record)
        self.assertEqual([str(f) for f in failures], [])
        log = _answers.apply(answers, llm_path=self.llm)
        self.assertIn("upcoming", log[0])
        self.assertEqual(o.override_for("model-x"), o.UPCOMING)
        # The file is in the batch's rollback snapshot.
        self.assertIn(o.OVERRIDES_FILE, _answers.touchable_paths(answers, self.llm))

    def test_refusals(self) -> None:
        o.set_override("Model X", o.OPEN)
        cases = [
            ({"kind": "weights-override", "name": "Model X", "weights": "maybe",
              "if_previous": "open"}, "must be one of"),
            ({"kind": "weights-override", "name": "Model X", "weights": "closed",
              "if_previous": None}, "stale"),
            ({"kind": "weights-override", "name": "Model X", "weights": "open",
              "if_previous": "open"}, "already has"),
            ({"kind": "weights-override", "name": "---", "weights": "open",
              "if_previous": None}, "no letters"),
            ({"kind": "weights-override", "name": "Model X", "if_previous": "open"},
             "'weights' is required"),
        ]
        for record, needle in cases:
            with self.subTest(needle=needle):
                answers, failures = self.check(record)
                self.assertEqual(answers, [])
                self.assertIn(needle, failures[0].message)

    def test_clearing_hands_the_name_back(self) -> None:
        o.set_override("Model X", o.CLOSED)
        answers, failures = self.check({"kind": "weights-override", "name": "model x",
                                        "weights": None, "if_previous": "closed"})
        self.assertEqual(failures, [])
        _answers.apply(answers, llm_path=self.llm)
        self.assertIsNone(o.override_for("Model X"))

    def test_a_closed_release_added_as_open_is_upcoming(self) -> None:
        queue = {(_answers.NEW_MODEL_ROUTE, _answers.NEW_CLOSED_MODEL, "closed-slug")}
        answers, failures = self.check(
            {"kind": "model-add", "name": "closed-slug", "open_weights": True}, queue)
        self.assertEqual(failures, [])
        with mock.patch.object(_answers, "_add_model"), \
                mock.patch.object(_answers._new_models, "record_proposed"):
            log = _answers.apply(answers, llm_path=self.llm)
        self.assertEqual(o.override_for("closed-slug"), o.UPCOMING)
        self.assertTrue(any("upcoming" in line for line in log))


if __name__ == "__main__":
    unittest.main()
