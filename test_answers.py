#!/usr/bin/env python3
"""Tests for answering the pipeline's questions non-interactively.

Run with ./test_answers.py

_answers.py is a trust boundary: its records arrive from a web form, a workflow
input, a file -- somewhere that is not this repository. Most of what follows is
therefore about what it *refuses*, and the refusals matter more than the
successes. Each one below stands for a way the queue, llm.json or the runner
itself could otherwise be corrupted by a record that looks perfectly ordinary.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import _answers
import edit
import _new_models
import _prompts
import _reference
import propose
import _rename
from _answers import (
    AA_IGNORE,
    MAPPING,
    MODEL_ADD,
    MODEL_CREATE,
    MODEL_EDIT,
    MODEL_RENAME,
    NEW_MODEL,
    REFERENCE_ADD,
    REFERENCE_REMOVE,
    Answer,
    AnswerError,
)

TBENCH = "update_tbench_mapping.py"
LLMSTATS = "update_llmstats_mapping.py"

UNIVERSES = {
    propose.MODELS: ["glm-5-3", "devstral-2"],
    propose.BENCHMARKS: ["swe_bench_verified", "hle"],
    propose.AA_SLUGS: ["aa-one", "aa-two"],
}

LLM_DOC = {
    "benchmarks": {
        "swe_bench_verified": {"name": "SWE-bench Verified"},
        "hle": {"name": "HLE"},
        "coding_index": {"name": "Coding", "derived": True},
    },
    "models": [
        {"name": "devstral-2", "scores": {}, "scores_updated": {}, "scores_source": {}},
        {"name": "glm-5-3", "scores": {}, "scores_updated": {}, "scores_source": {}},
    ],
}


class AnswersTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.llm = self.tmp / "llm.json"
        self.llm.write_text(json.dumps(LLM_DOC), encoding="utf-8")
        # Every mapping question below is one the queue asked, so the
        # queue-membership gate is not what any single test is measuring.
        self.queue = {
            (TBENCH, "mapping", "Some Model"),
            (LLMSTATS, "llmstats-benchmark", "hle"),
            (LLMSTATS, "llmstats-model", "Some Model"),
            (_answers.AA_ROUTE, "aa-mapping", "devstral-2"),
            (_answers.NEW_MODEL_ROUTE, "new-model", "fresh-slug"),
        }
        # current_value reads the real mapping files; the tests decide what the
        # file says instead, so if_previous is exercised without touching them.
        patcher = mock.patch.object(_answers, "current_value", return_value=None)
        self.current_value = patcher.start()
        self.addCleanup(patcher.stop)

    def check(self, record, **kwargs):
        """(answers, failures) for one record. `universes` is overridable, for
        the questions whose answer set is what is being measured."""
        kwargs.setdefault("universes", UNIVERSES)
        return _answers.validate([record], llm_path=self.llm, queue=self.queue, **kwargs)

    def refused(self, record, needle: str, **kwargs) -> None:
        answers, failures = self.check(record, **kwargs)
        self.assertEqual(answers, [], f"expected {record} to be refused")
        self.assertEqual(len(failures), 1)
        self.assertIn(needle, failures[0].message)

    def accepted(self, record, **kwargs) -> Answer:
        answers, failures = self.check(record, **kwargs)
        self.assertEqual([str(f) for f in failures], [])
        self.assertEqual(len(answers), 1)
        return answers[0]

    def mapping(self, **overrides) -> dict:
        record = {
            "kind": MAPPING,
            "route": TBENCH,
            "route_kind": "*",
            "subject": "Some Model",
            "answer": "glm-5-3",
            "if_previous": None,
        }
        record.update(overrides)
        return record


class TestRouting(AnswersTestCase):
    def test_a_route_is_a_table_key_not_a_path(self) -> None:
        self.refused(self.mapping(route="../../etc/passwd"), "unknown route")
        self.refused(self.mapping(route="/etc/passwd"), "unknown route")
        self.refused(self.mapping(route=None), "must be a string")

    def test_a_record_may_not_name_a_module(self) -> None:
        """propose.py imports route.module by name; that name must never be ours.

        The table it reads is hard-coded, so importing from it is safe. A record
        that could supply one would be a one-line import of anything on the
        runner -- with a deploy key and two API tokens in the environment.
        """
        for field in ("module", "writer", "mapping_const", "path", "file"):
            self.refused(self.mapping(**{field: "os"}), f"{field!r} is not accepted")

    def test_route_kind_picks_between_two_files_on_one_script(self) -> None:
        """update_llmstats_mapping.py owns two files with different universes."""
        model = _answers.resolve_route(LLMSTATS, "llmstats-model")
        benchmark = _answers.resolve_route(LLMSTATS, "llmstats-benchmark")
        self.assertNotEqual(model.mapping_const, benchmark.mapping_const)
        self.assertEqual(model.universe, propose.MODELS)
        self.assertEqual(benchmark.universe, propose.BENCHMARKS)

    def test_an_unknown_kind_on_a_two_file_route_is_refused(self) -> None:
        with self.assertRaises(AnswerError):
            _answers.resolve_route(LLMSTATS, "llmstats-nonsense")

    def test_a_single_file_route_takes_any_kind(self) -> None:
        self.assertEqual(
            _answers.resolve_route(TBENCH, "anything").module, "_tbench_mapping"
        )

    def test_every_route_in_the_table_resolves_to_one_of_our_files(self) -> None:
        for route_name, by_kind in propose.ROUTES.items():
            for kind in by_kind:
                path = _answers.mapping_path(_answers.resolve_route(route_name, kind))
                self.assertEqual(path.parent, _answers.HERE)


class TestMappingAnswers(AnswersTestCase):
    def test_a_good_answer_is_accepted(self) -> None:
        answer = self.accepted(self.mapping())
        self.assertEqual((answer.kind, answer.subject, answer.value), (MAPPING, "Some Model", "glm-5-3"))

    def test_unmappable_needs_no_universe(self) -> None:
        self.assertEqual(self.accepted(self.mapping(answer="__unmappable__")).value, "__unmappable__")

    def test_pending_is_never_an_answer(self) -> None:
        """__pending__ is a parking marker; writing it un-answers a decision.

        propose.py writes it for a name it could not resolve, and the reviewed-set
        loaders skip it so the name comes back. Written over a real decision it
        would put that question back in the loop for good.
        """
        self.refused(self.mapping(answer="__pending__"), "parking marker")

    def test_closed_weights_is_the_machines_verdict(self) -> None:
        self.refused(self.mapping(answer="__closed_weights__"), "not by hand")

    def test_an_answer_outside_the_universe_is_refused(self) -> None:
        self.refused(self.mapping(answer="no-such-model"), "not one of the known models")

    def test_a_benchmark_route_uses_the_benchmark_universe(self) -> None:
        record = self.mapping(route=LLMSTATS, route_kind="llmstats-benchmark", subject="hle", answer="hle")
        self.assertEqual(self.accepted(record).value, "hle")
        self.refused(
            self.mapping(route=LLMSTATS, route_kind="llmstats-benchmark", subject="hle", answer="devstral-2"),
            "not one of the known benchmarks",
        )

    def test_an_empty_universe_refuses_rather_than_admits(self) -> None:
        """A cold universe means "cannot check", never "anything goes".

        propose.py falls back to the queue's own candidate list when the source
        is unreachable, which is graceful degradation for a suggestion. For an
        answer it would be a way round the check.
        """
        cold = dict(UNIVERSES, models=[])
        answers, failures = _answers.validate(
            [self.mapping()], llm_path=self.llm, universes=cold, queue=self.queue
        )
        self.assertEqual(answers, [])
        self.assertIn("unreachable", failures[0].message)

    def test_the_aa_route_is_checked_in_both_directions(self) -> None:
        """That file's keys are llm.json names and its values are AA slugs.

        Every other mapping runs the other way. Checking only the value would let
        a plausible, entirely wrong line through.
        """
        good = self.mapping(
            route=_answers.AA_ROUTE, subject="devstral-2", answer="aa-one", route_kind="aa-mapping"
        )
        self.assertEqual(self.accepted(good).value, "aa-one")
        # Queued, so the outer gate stands aside and the direction check is what
        # this measures.
        self.queue.add((_answers.AA_ROUTE, "aa-mapping", "not-a-model"))
        self.refused(
            self.mapping(
                route=_answers.AA_ROUTE, subject="not-a-model", answer="aa-one", route_kind="aa-mapping"
            ),
            "not a model in llm.json",
        )


class TestStaleness(AnswersTestCase):
    def test_if_previous_is_required(self) -> None:
        record = self.mapping()
        del record["if_previous"]
        self.refused(record, "'if_previous' is required")

    def test_a_stale_view_is_refused(self) -> None:
        """The page may be working from a queue up to three hours old."""
        self.current_value.return_value = ["glm-5-3"]
        self.refused(self.mapping(if_previous=None), "stale")

    def test_a_deliberate_overwrite_is_allowed(self) -> None:
        """Knowing what the file says is what makes changing it an answer."""
        self.current_value.return_value = ["devstral-2"]
        self.assertEqual(self.accepted(self.mapping(if_previous="devstral-2")).value, "glm-5-3")

    def test_a_list_value_is_compared_as_a_list(self) -> None:
        """propose.recorded_value returns None for the AA file's list values.

        Reusing it here would make the staleness check blind to exactly the
        entries most worth checking.
        """
        self.current_value.return_value = ["aa-one", "aa-two"]
        self.accepted(self.mapping(if_previous=["aa-one", "aa-two"]))
        self.refused(self.mapping(if_previous=["aa-one"]), "stale")


class TestQueueGate(AnswersTestCase):
    def test_only_a_queued_question_can_be_answered(self) -> None:
        """Otherwise a valid-looking batch can map names nobody asked about.

        Every value would pass the universe check and every line would read like
        a real mapping, while attaching scores to the wrong models.
        """
        self.refused(self.mapping(subject="Never Asked"), "not a question")

    def test_the_gate_can_be_stood_down_for_hand_answers(self) -> None:
        self.accepted(self.mapping(subject="Never Asked"), require_queue=False)

    def test_adding_a_model_needs_the_question_too(self) -> None:
        """add.py builds an entry from whatever name it is handed."""
        self.refused({"kind": MODEL_ADD, "name": "junk-slug"}, "not a question")
        self.assertEqual(self.accepted({"kind": MODEL_ADD, "name": "fresh-slug"}).subject, "fresh-slug")

    def test_ignoring_a_model_needs_the_question_too(self) -> None:
        self.refused({"kind": NEW_MODEL, "subject": "junk", "answer": "__ignored__"}, "not a question")

    def test_rejecting_aa_suggestions_needs_the_question_too(self) -> None:
        self.refused(
            {"kind": AA_IGNORE, "subject": "glm-5-3", "answer": ["aa-one"]}, "not a question"
        )

    def test_a_jsonl_queue_and_a_json_queue_read_alike(self) -> None:
        """Both formats start with "{", so the reader parses rather than sniffs."""
        entry = {"command": f"./{TBENCH} -w", "kind": "mapping", "subject": "Some Model"}
        jsonl = self.tmp / "q.jsonl"
        jsonl.write_text(json.dumps(entry) + "\n", encoding="utf-8")
        published = self.tmp / "pending.json"
        published.write_text(
            json.dumps({"questions": [dict(entry, route=TBENCH, route_kind="mapping")]}),
            encoding="utf-8",
        )
        expected = {(TBENCH, "mapping", "Some Model")}
        self.assertEqual(_answers.load_queue(jsonl), expected)
        self.assertEqual(_answers.load_queue(published), expected)


class TestNewModels(AnswersTestCase):
    def test_added_is_not_the_other_half_of_the_answer(self) -> None:
        """__added__ for a slug absent from llm.json re-asks the question forever.

        apply_decisions() logs "kept, but absent ... left alone", clears the
        decision line and never dismisses the slug, so check_new.py's known set
        misses it and offers the model again on the very next run. "Yes, add it"
        has to run add.py.
        """
        self.refused(
            {"kind": NEW_MODEL, "subject": "fresh-slug", "answer": _new_models.ADDED},
            "model-add",
        )

    def test_ignored_is_accepted(self) -> None:
        answer = self.accepted({"kind": NEW_MODEL, "subject": "fresh-slug", "answer": "__ignored__"})
        self.assertEqual(answer.value, _new_models.IGNORED)

    def test_ignoring_a_model_that_is_already_in_llm_json_is_refused(self) -> None:
        """That is a deletion of a scored entry wearing a new model's clothes."""
        queue = self.queue | {(_answers.NEW_MODEL_ROUTE, "new-model", "devstral-2")}
        answers, failures = _answers.validate(
            [{"kind": NEW_MODEL, "subject": "devstral-2", "answer": "__ignored__"}],
            llm_path=self.llm,
            universes=UNIVERSES,
            queue=queue,
        )
        self.assertEqual(answers, [])
        self.assertIn("already in llm.json", failures[0].message)


class TestModelEdits(AnswersTestCase):
    def test_only_edit_pys_own_metadata_fields_are_offered(self) -> None:
        """edit.py has a flag per field; anything else exits 2 from argparse."""
        self.assertEqual(
            set(_answers.METADATA_FIELDS),
            {"params", "context", "url", "creator", "creator_url", "date_added"},
        )
        # The name is the model's identity, not a field: renaming it has to move
        # every mapping file with it, which is what a model-rename record is for.
        self.refused({"kind": MODEL_EDIT, "name": "devstral-2", "fields": {"name": "x"}}, "not editable")
        self.refused({"kind": MODEL_EDIT, "name": "devstral-2", "fields": {"vram": "8"}}, "not editable")

    def test_a_metadata_value_is_checked_against_its_own_field(self) -> None:
        """A URL that is not one and a date that is not one are silent lies."""
        for field, value in (
            ("url", "huggingface.co/x/y"),
            ("creator_url", "mistral.ai"),
        ):
            self.refused({"kind": MODEL_EDIT, "name": "devstral-2", "fields": {field: value}}, "URL")
        self.refused(
            {"kind": MODEL_EDIT, "name": "devstral-2", "fields": {"date_added": "03.02.2026"}},
            "YYYY-MM-DD",
        )
        answer = self.accepted({
            "kind": MODEL_EDIT,
            "name": "devstral-2",
            "fields": {
                "url": "https://huggingface.co/x/y",
                "creator": "Mistral",
                "creator_url": "https://mistral.ai/",
                "date_added": "2026-02-03",
                "params": "123B",
                "context": "256k",
            },
        })
        self.assertEqual(len(answer.fields), 6)

    def test_a_creator_field_reaches_edit_py_as_its_own_flag(self) -> None:
        """The record keeps the underscore; argparse only knows the dash."""
        answer = Answer(
            index=0, kind=MODEL_EDIT, subject="devstral-2",
            fields={"creator_url": "https://mistral.ai/", "date_added": None},
        )
        with mock.patch.object(_answers, "_run") as run:
            _answers._apply_edit(answer, self.llm)
        argv = run.call_args[0][0]
        self.assertIn("--creator-url=https://mistral.ai/", argv)
        self.assertIn("--date-added=null", argv)

    def test_a_derived_column_takes_no_score(self) -> None:
        """derive_indexes.py recomputes it, so the write is a silent no-op."""
        self.refused(
            {"kind": MODEL_EDIT, "name": "devstral-2", "scores": {"coding_index": 50}},
            "not a benchmark",
        )

    def test_a_score_must_be_a_finite_number(self) -> None:
        for value in (float("inf"), float("nan")):
            self.refused(
                {"kind": MODEL_EDIT, "name": "devstral-2", "scores": {"hle": value}}, "finite"
            )
        self.refused({"kind": MODEL_EDIT, "name": "devstral-2", "scores": {"hle": True}}, "number")

    def test_an_edit_needs_a_model_that_exists(self) -> None:
        self.refused({"kind": MODEL_EDIT, "name": "ghost", "fields": {"params": "7B"}}, "not a model")

    def test_an_empty_edit_is_refused(self) -> None:
        self.refused({"kind": MODEL_EDIT, "name": "devstral-2"}, "nothing to change")

    def test_a_score_carries_the_date_and_page_it_was_read_from(self) -> None:
        answer = self.accepted({
            "kind": MODEL_EDIT,
            "name": "devstral-2",
            "scores": {"hle": 42.5},
            "score_date": "2026-08-06",
            "score_url": "https://example.com/board",
        })
        self.assertEqual(answer.score_date, "2026-08-06")
        self.assertEqual(answer.score_url, "https://example.com/board")

    def test_provenance_defaults_to_today_and_nobody(self) -> None:
        """Absent is not the same as null: edit.py stamps the defaults itself.

        Sending None for either would say the same thing today, but leaving the
        flags off keeps one script deciding what "now" and "unattributed" mean.
        """
        answer = self.accepted({"kind": MODEL_EDIT, "name": "devstral-2", "scores": {"hle": 1}})
        self.assertIsNone(answer.score_date)
        self.assertIsNone(answer.score_url)
        with mock.patch.object(_answers, "_run") as run:
            _answers._apply_edit(answer, self.llm)
        flags = [a for a in run.call_args[0][0] if a.startswith("--score-")]
        self.assertEqual(flags, [])

    def test_a_blank_page_credits_nobody(self) -> None:
        """Which is what a hand edit means, and the weakest precedence rung."""
        answer = self.accepted(
            {"kind": MODEL_EDIT, "name": "devstral-2", "scores": {"hle": 1}, "score_url": "  "}
        )
        self.assertIsNone(answer.score_url)

    def test_provenance_needs_a_score_to_stamp(self) -> None:
        """A params edit has nothing to date or credit."""
        self.refused(
            {"kind": MODEL_EDIT, "name": "devstral-2", "fields": {"params": "70B"},
             "score_date": "2026-08-06"},
            "send at least one score",
        )
        self.refused(
            {"kind": MODEL_EDIT, "name": "devstral-2", "fields": {"params": "70B"},
             "score_url": "https://example.com/board"},
            "send at least one score",
        )

    def test_a_date_that_is_not_one_is_refused(self) -> None:
        for value in ("06.08.2026", "2026-13-01", "yesterday", 20260806):
            self.refused(
                {"kind": MODEL_EDIT, "name": "devstral-2", "scores": {"hle": 1},
                 "score_date": value},
                "date",
            )

    def test_a_date_in_the_future_is_refused(self) -> None:
        """Nothing can have been read from a page tomorrow, and a stray year
        would sit at the top of every recently-updated view until noticed."""
        ahead = (date.today() + timedelta(days=1)).isoformat()
        self.refused(
            {"kind": MODEL_EDIT, "name": "devstral-2", "scores": {"hle": 1}, "score_date": ahead},
            "future",
        )
        self.assertEqual(
            self.accepted({"kind": MODEL_EDIT, "name": "devstral-2", "scores": {"hle": 1},
                           "score_date": date.today().isoformat()}).score_date,
            date.today().isoformat(),
        )

    def test_a_page_must_be_a_url(self) -> None:
        """It decides the score's precedence rung, so free text is not one."""
        for value in ("example.com/board", "javascript:alert(1)", "file:///etc/passwd", 7):
            self.refused(
                {"kind": MODEL_EDIT, "name": "devstral-2", "scores": {"hle": 1},
                 "score_url": value},
                "URL",
            )

    def test_argv_is_built_so_edit_pys_own_argv_scan_cannot_be_fooled(self) -> None:
        """infer_json_file() hand-parses argv before argparse ever sees it.

        It assumes "--flag value" for every long option, so a value that looks
        like a path would silently redirect the write. --flag=VALUE takes its
        one-token branch, and a bare -- pins the path.
        """
        answer = Answer(
            index=0,
            kind=MODEL_EDIT,
            subject="devstral-2",
            fields={"params": "70B"},
            scores={"swe_bench_verified": 61.2, "hle": None},
        )
        with mock.patch.object(_answers, "_run") as run:
            _answers._apply_edit(answer, self.llm)
        argv = run.call_args[0][0]
        self.assertEqual(argv[-2:], ["--", str(self.llm)])
        flags = [a for a in argv if a.startswith("--") and a != "--"]
        self.assertTrue(all("=" in flag for flag in flags), flags)
        self.assertIn("--model=devstral-2", flags)
        self.assertIn("--params=70B", flags)
        self.assertIn("--swe-bench-verified=61.2", flags)
        self.assertIn("--hle=null", flags)  # the literal edit.py reads as "clear"

    def test_the_provenance_flags_edit_py_actually_has(self) -> None:
        """--flag=VALUE for these too, and only the ones the record set."""
        answer = Answer(
            index=0,
            kind=MODEL_EDIT,
            subject="devstral-2",
            scores={"hle": 42.5},
            score_date="2026-08-06",
            score_url="https://example.com/board",
        )
        with mock.patch.object(_answers, "_run") as run:
            _answers._apply_edit(answer, self.llm)
        argv = run.call_args[0][0]
        self.assertIn("--score-date=2026-08-06", argv)
        self.assertIn("--score-url=https://example.com/board", argv)
        self.assertEqual(argv[-2:], ["--", str(self.llm)])
        edit_parser_flags = set(edit.SCORE_PROVENANCE_FLAGS)
        self.assertEqual(edit_parser_flags, {"--score-date", "--score-url"})


class TestModelCreate(AnswersTestCase):
    """A model added by hand, which no queue ever offered.

    model-add is the queue's answer and stays gated on the question. This is the
    other direction, so the guard is the name and the metadata instead.
    """

    def create(self, **overrides) -> dict:
        record = {"kind": MODEL_CREATE, "name": "fresh-model-1"}
        record.update(overrides)
        return record

    def test_a_hand_added_model_needs_no_queued_question(self) -> None:
        answer = self.accepted(self.create())
        self.assertEqual(answer.kind, MODEL_CREATE)
        self.assertEqual(answer.subject, "fresh-model-1")

    def test_the_name_has_to_be_a_slug(self) -> None:
        """It is the model's identity in every mapping file and in AA lookups."""
        for name in ("Fresh Model", "GLM-5.3", "fresh_model", "fresh--model", "-fresh"):
            self.refused(self.create(name=name), "slug")

    def test_an_existing_model_is_not_created_again(self) -> None:
        self.refused(self.create(name="devstral-2"), "already a model")

    def test_metadata_travels_with_it_and_is_checked(self) -> None:
        answer = self.accepted(self.create(fields={
            "url": "https://huggingface.co/x/y",
            "creator": "Mistral",
            "creator_url": "https://mistral.ai/",
            "params": "123B",
            "context": "256k",
        }))
        self.assertEqual(answer.fields["creator"], "Mistral")
        self.refused(self.create(fields={"url": "not-a-url"}), "URL")
        self.refused(self.create(fields={"vram": "8"}), "not editable")

    def test_add_py_is_called_with_every_field(self) -> None:
        answer = Answer(
            index=0,
            kind=MODEL_CREATE,
            subject="fresh-model-1",
            fields={"creator_url": "https://mistral.ai/", "params": "123B"},
        )
        with mock.patch.object(_answers, "_run") as run:
            _answers._apply_one(answer, self.llm)
        argv = run.call_args[0][0]
        self.assertIn("--name=fresh-model-1", argv)
        self.assertIn("--creator-url=https://mistral.ai/", argv)
        self.assertIn("--params=123B", argv)
        self.assertEqual(argv[-2:], ["--", str(self.llm)])
        flags = [a for a in argv if a.startswith("--") and a != "--"]
        self.assertTrue(all("=" in flag for flag in flags), flags)

    def test_creating_records_no_decision_the_way_adding_does(self) -> None:
        """__added__ answers a question check_new.py asked; nobody asked this one."""
        answer = Answer(index=0, kind=MODEL_CREATE, subject="fresh-model-1")
        with mock.patch.object(_answers, "_run"), mock.patch.object(
            _new_models, "record_proposed"
        ) as recorded:
            _answers._apply_one(answer, self.llm)
        recorded.assert_not_called()


class TestModelRename(AnswersTestCase):
    def rename(self, **overrides) -> dict:
        record = {"kind": MODEL_RENAME, "name": "devstral-2", "new_name": "devstral-2-0512"}
        record.update(overrides)
        return record

    def test_a_rename_names_both_ends(self) -> None:
        answer = self.accepted(self.rename())
        self.assertEqual((answer.subject, answer.value), ("devstral-2", "devstral-2-0512"))

    def test_the_model_has_to_exist_and_the_new_name_must_not(self) -> None:
        self.refused(self.rename(name="ghost"), "not a model")
        self.refused(self.rename(new_name="glm-5-3"), "already a model")
        self.refused(self.rename(new_name="devstral-2"), "already its name")

    def test_the_new_name_has_to_be_a_slug(self) -> None:
        self.refused(self.rename(new_name="Devstral 2"), "slug")

    def test_the_rollback_snapshot_covers_every_mapping_file(self) -> None:
        """A rename walks all of them; a half-renamed model is undetectable."""
        answer = Answer(index=0, kind=MODEL_RENAME, subject="devstral-2", value="devstral-2-0512")
        paths = set(_answers.touchable_paths([answer], self.llm))
        self.assertTrue(set(_rename.touched_paths(self.llm)) <= paths)
        self.assertIn(_answers.HERE / "model-name-mapping-tbench-to-artificialanalysis.json", paths)

    def test_it_is_applied_by_the_renamer_not_by_a_json_poke(self) -> None:
        answer = Answer(index=0, kind=MODEL_RENAME, subject="devstral-2", value="devstral-2-0512")
        with mock.patch.object(_rename, "rename", return_value=["did it"]) as renamed:
            log = _answers._apply_one(answer, self.llm)
        renamed.assert_called_once_with("devstral-2", "devstral-2-0512", self.llm)
        self.assertEqual(log, ["did it"])


class TestReferenceModels(AnswersTestCase):
    """The closed models the index carries, edited from the admin page.

    Both kinds move reference-models.json and llm.json together, so the tests
    that matter are about the pair: an added slug that has no entry behind it
    is inert, and an entry left behind after its slug goes is a closed model
    back in the open field's ranking.
    """

    def setUp(self) -> None:
        super().setUp()
        self.list_path = self.tmp / "reference-models.json"
        self.list_path.write_text(json.dumps(["aa-one", "aa-two"]), encoding="utf-8")
        patcher = mock.patch.object(_reference, "REFERENCE_MODELS", self.list_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def carried(self) -> list[str]:
        return json.loads(self.list_path.read_text(encoding="utf-8"))

    def test_an_add_has_to_be_a_slug_aa_publishes(self) -> None:
        answer = self.accepted({"kind": REFERENCE_ADD, "name": "aa-three"},
                               universes={**UNIVERSES, propose.AA_SLUGS: ["aa-three"]})
        self.assertEqual((answer.kind, answer.subject), (REFERENCE_ADD, "aa-three"))
        self.refused({"kind": REFERENCE_ADD, "name": "not-on-aa"},
                     "not an Artificial Analysis slug")
        self.refused({"kind": REFERENCE_ADD, "name": "Not A Slug"}, "slug")

    def test_an_unreachable_aa_does_not_refuse_an_add(self) -> None:
        """build_universes leaves the list empty when AA is down, and a dead
        source must not refuse a batch it has no opinion about."""
        self.accepted({"kind": REFERENCE_ADD, "name": "whatever-1"},
                      universes={**UNIVERSES, propose.AA_SLUGS: []})

    def test_a_slug_is_added_once(self) -> None:
        self.refused({"kind": REFERENCE_ADD, "name": "aa-one"}, "already a reference model")

    def test_a_remove_names_one_that_is_carried(self) -> None:
        answer = self.accepted({"kind": REFERENCE_REMOVE, "name": "aa-two"})
        self.assertEqual((answer.kind, answer.subject), (REFERENCE_REMOVE, "aa-two"))
        self.refused({"kind": REFERENCE_REMOVE, "name": "aa-three"}, "not a reference model")

    def test_the_last_one_cannot_be_removed(self) -> None:
        records = [{"kind": REFERENCE_REMOVE, "name": slug} for slug in ("aa-one", "aa-two")]
        answers, failures = _answers.validate(
            records, llm_path=self.llm, universes=UNIVERSES, queue=self.queue
        )
        self.assertEqual(len(answers), 2)
        self.assertEqual(len(failures), 1)
        self.assertIn("would leave no reference models", failures[0].message)
        self.assertEqual(failures[0].index, 1)

    def test_the_last_one_can_be_swapped_in_one_batch(self) -> None:
        """The floor is about the batch's result, not about each record: the
        one shape it must not block is replacing the final model."""
        records = [
            {"kind": REFERENCE_REMOVE, "name": "aa-one"},
            {"kind": REFERENCE_REMOVE, "name": "aa-two"},
            {"kind": REFERENCE_ADD, "name": "aa-three"},
        ]
        answers, failures = _answers.validate(
            records, llm_path=self.llm,
            universes={**UNIVERSES, propose.AA_SLUGS: ["aa-three"]}, queue=self.queue
        )
        self.assertEqual([str(f) for f in failures], [])
        self.assertEqual(len(answers), 3)

    def test_a_reference_record_and_an_edit_cannot_share_a_model(self) -> None:
        """Both write llm.json, and a batch is applied in order: an edit of a
        model the same batch removes would be looking for a row that is gone."""
        self.list_path.write_text(json.dumps(["devstral-2", "aa-two"]), encoding="utf-8")
        records = [
            {"kind": REFERENCE_REMOVE, "name": "devstral-2"},
            {"kind": MODEL_EDIT, "name": "devstral-2", "fields": {"params": "1B"}},
        ]
        _, failures = _answers.validate(
            records, llm_path=self.llm, universes=UNIVERSES, queue=self.queue
        )
        self.assertTrue(any("touched twice" in f.message for f in failures), failures)

    def test_the_rollback_snapshot_covers_the_list(self) -> None:
        answer = Answer(index=0, kind=REFERENCE_REMOVE, subject="aa-one")
        self.assertIn(self.list_path, _answers.touchable_paths([answer], self.llm))

    def test_adding_carries_the_slug_and_creates_the_entry(self) -> None:
        answer = Answer(index=0, kind=REFERENCE_ADD, subject="aa-three")
        with mock.patch.object(_answers, "_add_model") as added, \
                mock.patch.object(_answers.derive_indexes, "refresh_and_report"):
            _answers._apply_one(answer, self.llm)
        self.assertIn("aa-three", self.carried())
        name, fields, path = added.call_args.args
        self.assertEqual((name, path), ("aa-three", self.llm))
        # A closed model has no weights repository, so the row is checked
        # against the page its numbers are published on.
        self.assertEqual(fields["url"], _reference.aa_model_page("aa-three"))

    def test_adding_a_model_already_in_llm_json_does_not_run_add_py(self) -> None:
        answer = Answer(index=0, kind=REFERENCE_ADD, subject="devstral-2")
        with mock.patch.object(_answers, "_add_model") as added, \
                mock.patch.object(_answers.derive_indexes, "refresh_and_report"):
            _answers._apply_one(answer, self.llm)
        added.assert_not_called()
        self.assertIn("devstral-2", self.carried())

    def test_an_added_model_is_flagged_without_waiting_for_a_refresh(self) -> None:
        """A record-only run never reaches update.py or derive_indexes.py, so
        the flag has to be stamped here or the row sits in the table as an
        open-weight one until the next scheduled refresh."""
        answer = Answer(index=0, kind=REFERENCE_ADD, subject="devstral-2")
        with mock.patch.object(_answers.derive_indexes, "refresh_and_report"):
            _answers._apply_one(answer, self.llm)
        doc = json.loads(self.llm.read_text(encoding="utf-8"))
        flagged = {m["name"] for m in doc["models"] if m.get("reference") is True}
        self.assertEqual(flagged, {"devstral-2"})

    def test_removing_drops_the_slug_and_the_entry(self) -> None:
        self.list_path.write_text(json.dumps(["devstral-2", "aa-two"]), encoding="utf-8")
        answer = Answer(index=0, kind=REFERENCE_REMOVE, subject="devstral-2")
        with mock.patch.object(_answers.derive_indexes, "refresh_and_report"):
            _answers._apply_one(answer, self.llm)
        self.assertEqual(self.carried(), ["aa-two"])
        doc = json.loads(self.llm.read_text(encoding="utf-8"))
        self.assertEqual([m["name"] for m in doc["models"]], ["glm-5-3"])

    def test_the_writer_refuses_to_empty_the_list_too(self) -> None:
        """validate() is the gate that reports it; this is the backstop under
        anything that reached the writer another way."""
        self.list_path.write_text(json.dumps(["aa-one"]), encoding="utf-8")
        with self.assertRaises(ValueError):
            _reference.remove_reference_slug("aa-one")
        self.assertEqual(self.carried(), ["aa-one"])


class TestBatches(AnswersTestCase):
    def test_a_batch_is_capped(self) -> None:
        records = [self.mapping(subject=f"m{i}") for i in range(_answers.MAX_RECORDS + 1)]
        answers, failures = _answers.validate(
            records, llm_path=self.llm, universes=UNIVERSES, queue=self.queue
        )
        self.assertEqual(answers, [])
        self.assertIn("at most", failures[0].message)

    def test_the_cap_the_admin_page_shows_is_the_one_that_is_enforced(self) -> None:
        """Three files carry the number, and only this one decides it.

        The page greys out Send past the cap and says how many to un-answer, so
        a page holding a bigger number than the endpoint would offer a batch
        that cannot be dispatched -- and a smaller one would refuse a sitting
        that was fine.
        """
        for path, pattern in (
            (_answers.HERE / "_admin" / "api.php", r"const MAX_RECORDS = (\d+);"),
            (_answers.HERE / "_admin" / "index.html", r"const MAX_RECORDS = (\d+);"),
        ):
            with self.subTest(path.name):
                found = re.search(pattern, path.read_text(encoding="utf-8"))
                self.assertIsNotNone(found, f"{path.name}: no MAX_RECORDS to check")
                self.assertEqual(int(found.group(1)), _answers.MAX_RECORDS)

    def test_one_model_may_only_be_touched_once(self) -> None:
        """Records apply in order and roll back together, so an edit sent with a
        rename of the same model would look for a name that is no longer there."""
        answers, failures = _answers.validate(
            [
                {"kind": MODEL_RENAME, "name": "devstral-2", "new_name": "devstral-2-0512"},
                {"kind": MODEL_EDIT, "name": "devstral-2", "fields": {"params": "70B"}},
            ],
            llm_path=self.llm,
            universes=UNIVERSES,
            queue=self.queue,
        )
        self.assertEqual(len(answers), 2)
        self.assertEqual(len(failures), 1)
        self.assertIn("touched twice", failures[0].message)

    def test_a_rename_target_may_not_be_created_in_the_same_batch(self) -> None:
        _answers_out, failures = _answers.validate(
            [
                {"kind": MODEL_RENAME, "name": "devstral-2", "new_name": "devstral-3"},
                {"kind": MODEL_CREATE, "name": "devstral-3"},
            ],
            llm_path=self.llm,
            universes=UNIVERSES,
            queue=self.queue,
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("touched twice", failures[0].message)

    def test_two_different_models_in_one_batch_are_fine(self) -> None:
        answers, failures = _answers.validate(
            [
                {"kind": MODEL_RENAME, "name": "devstral-2", "new_name": "devstral-2-0512"},
                {"kind": MODEL_EDIT, "name": "glm-5-3", "fields": {"params": "70B"}},
                {"kind": MODEL_CREATE, "name": "fresh-model-1"},
            ],
            llm_path=self.llm,
            universes=UNIVERSES,
            queue=self.queue,
        )
        self.assertEqual([str(f) for f in failures], [])
        self.assertEqual(len(answers), 3)

    def test_answering_the_same_question_twice_is_refused(self) -> None:
        records = [self.mapping(), self.mapping(answer="devstral-2")]
        _answers_out, failures = _answers.validate(
            records, llm_path=self.llm, universes=UNIVERSES, queue=self.queue
        )
        self.assertTrue(any("twice" in f.message for f in failures))

    def test_one_bad_record_reports_only_that_record(self) -> None:
        records = [self.mapping(), self.mapping(subject="Never Asked")]
        answers, failures = _answers.validate(
            records, llm_path=self.llm, universes=UNIVERSES, queue=self.queue
        )
        self.assertEqual(len(answers), 1)
        self.assertEqual([f.index for f in failures], [1])

    def test_a_failure_part_way_through_rolls_the_whole_batch_back(self) -> None:
        """Half an answered queue is the one outcome nobody could reconstruct."""
        existing = self.tmp / "existing.json"
        existing.write_text('{"before": 1}', encoding="utf-8")
        fresh = self.tmp / "fresh.json"

        def apply_one(answer, llm_path):
            existing.write_text('{"after": 2}', encoding="utf-8")
            fresh.write_text("{}", encoding="utf-8")
            if answer.index == 1:
                raise AnswerError("boom")
            return ["wrote"]

        answers = [Answer(0, MAPPING, "a"), Answer(1, MAPPING, "b")]
        with mock.patch.object(_answers, "touchable_paths", return_value=[existing, fresh]), \
             mock.patch.object(_answers, "_apply_one", side_effect=apply_one):
            with self.assertRaises(AnswerError):
                _answers.apply(answers, llm_path=self.llm)

        self.assertEqual(existing.read_text(encoding="utf-8"), '{"before": 1}')
        self.assertFalse(fresh.exists(), "a file the batch created must not survive a rollback")


class TestCollectMode(AnswersTestCase):
    def test_applying_under_collect_mode_is_refused(self) -> None:
        """Every mapping writer is a no-op while collect mode is on.

        That is what stops CI answering its own questions. Applying there would
        report success for writes that never happened.
        """
        with mock.patch.dict(os.environ, {_prompts.ENV_VAR: "/tmp/report.jsonl"}):
            with self.assertRaises(AnswerError) as caught:
                _answers.apply([Answer(0, MAPPING, "a")], llm_path=self.llm)
        self.assertIn(_prompts.ENV_VAR, str(caught.exception))

    def test_a_child_process_does_not_inherit_collect_mode(self) -> None:
        """add.py returns 0 without doing anything while the variable is set."""
        with mock.patch.dict(os.environ, {_prompts.ENV_VAR: "/tmp/report.jsonl"}):
            self.assertNotIn(_prompts.ENV_VAR, _prompts.child_env())

    def test_a_blank_value_is_not_collect_mode(self) -> None:
        with mock.patch.dict(os.environ, {_prompts.ENV_VAR: ""}):
            self.assertIsNone(_prompts.collecting())


class TestHelperFailures(AnswersTestCase):
    def test_a_failing_helper_is_fatal_rather_than_a_warning(self) -> None:
        """propose.py warns and carries on when add.py fails, which is right for
        a best-effort suggestion. Reporting success for a write that did not
        happen is the one thing the sender cannot detect."""
        completed = mock.Mock(returncode=1, stderr="add.py: boom\n", stdout="")
        with mock.patch("subprocess.run", return_value=completed):
            with self.assertRaises(AnswerError) as caught:
                _answers._run(["true"], "add.py failed")
        self.assertIn("boom", str(caught.exception))


class TestWorkflowWiring(unittest.TestCase):
    """The workflow is the other half of the trust boundary.

    _answers.py can only refuse a record it is handed. If the input reaches a
    shell before it reaches Python, none of that matters: ${{ }} is textual
    substitution before bash parses, on a runner holding a write deploy key and
    two API tokens.
    """

    WORKFLOW = _answers.HERE / ".github" / "workflows" / "update-benchmarks.yml"

    def steps(self) -> list[dict]:
        import yaml

        doc = yaml.safe_load(self.WORKFLOW.read_text(encoding="utf-8"))
        return doc["jobs"]["update"]["steps"]

    def test_no_expression_is_interpolated_into_a_script(self) -> None:
        import re

        offenders = [
            (step.get("name"), expr.strip())
            for step in self.steps()
            for expr in re.findall(r"\$\{\{(.*?)\}\}", step.get("run") or "")
        ]
        self.assertEqual(
            offenders,
            [],
            "pass the value through env: and read it as \"$VAR\" instead",
        )

    def test_the_answers_input_is_only_ever_an_env_value(self) -> None:
        uses = [
            step.get("name")
            for step in self.steps()
            if "inputs.answers" in str(step.get("env") or {})
        ]
        self.assertEqual(uses, ["Apply the answers"])

    def test_every_step_that_acts_on_the_input_is_gated_on_it(self) -> None:
        """The cron and merge paths must be untouched by this feature.

        The test step is deliberately not in this list: it guards the applier
        and has to run on every path, answers or not.
        """
        gated = {"Guard the answers input", "Apply the answers", "Commit and push the answers"}
        names = {s.get("name") for s in self.steps()}
        self.assertTrue(gated <= names, f"missing steps: {gated - names}")
        for step in self.steps():
            if step.get("name") in gated:
                self.assertIn("inputs.answers", str(step.get("if") or ""), step.get("name"))

    def test_the_apply_step_turns_collect_mode_off(self) -> None:
        """Every mapping writer is a no-op while it is on."""
        apply_step = next(s for s in self.steps() if s.get("name") == "Apply the answers")
        self.assertEqual(apply_step["env"][_prompts.ENV_VAR], "")

    def test_the_answer_applier_is_tested_before_anything_is_pushed(self) -> None:
        run = next(
            s["run"] for s in self.steps() if s.get("name", "").startswith("Check collect mode")
        )
        self.assertIn("./test_answers.py", run)

    def test_the_step_api_php_reads_is_the_step_that_applies_answers(self) -> None:
        """The endpoint reports that step's outcome, and the page trusts it.

        The run's own conclusion cannot stand in for it: `Fail if a step of
        update-all failed` reds a run whose answers were applied, committed and
        pushed minutes earlier. If the name here drifts, api.php reports no step
        at all -- which the page reads as "cannot tell" and keeps the answered
        cards locked, so the safe half; but the message it shows is then wrong
        about why.
        """
        api = (_answers.HERE / "_admin" / "api.php").read_text(encoding="utf-8")
        found = re.search(r"const ANSWER_STEP = '([^']+)';", api)
        self.assertIsNotNone(found, "api.php: no ANSWER_STEP to check")
        self.assertIn(
            found.group(1),
            [step.get("name") for step in self.steps()],
            "api.php names a workflow step that does not exist",
        )

    def test_the_answer_step_runs_before_anything_that_can_red_the_run(self) -> None:
        """Which is why the step's outcome and the run's conclusion differ.

        Nothing about this ordering is wrong -- an answer should be recorded
        even when the refresh after it falls over -- but the admin page reads
        the step rather than the run precisely because of it, and a reordering
        that made the run's conclusion authoritative again would leave that
        indirection looking pointless.
        """
        names = [step.get("name") for step in self.steps()]
        self.assertLess(
            names.index("Apply the answers"),
            names.index("Fail if a step of update-all failed"),
        )

    def test_the_admin_page_stays_off_the_published_site(self) -> None:
        """Pages runs Jekyll, which does not copy _-prefixed paths into the site.

        That is already why _matching.py and its siblings 404 there, and it is
        the whole of what keeps _admin/ and _pending/ unpublished. Adding a
        .nojekyll would serve both directories to anyone who asked.
        """
        self.assertFalse(
            (_answers.HERE / ".nojekyll").exists(),
            "a .nojekyll would publish _admin/ and _pending/ on the live site",
        )
        for directory in ("_admin", "_pending"):
            self.assertTrue(directory.startswith("_"), directory)

    def test_the_proposal_pr_is_opt_in(self) -> None:
        """No automatic path may open one.

        The admin page answers the same queue from _pending/pending.json, so a
        proposal PR is a fallback rather than the delivery mechanism -- and one
        nobody intends to merge is not free: the propose step skips while a PR
        is open, so a stale one blocks the next.

        `inputs` is null on a schedule and on a merge, which makes this falsy
        there; only a manual dispatch that ticks the box can open one.
        """
        step = next(s for s in self.steps() if s.get("name") == "Propose the pending mappings")
        self.assertIn("inputs.propose", str(step.get("if")))

        import yaml

        doc = yaml.safe_load(self.WORKFLOW.read_text(encoding="utf-8"))
        propose = doc[True]["workflow_dispatch"]["inputs"]["propose"]
        self.assertIs(propose["default"], False)

    def test_a_push_trigger_is_never_added(self) -> None:
        """The commits ride a deploy key, which does trigger workflow events."""
        import yaml

        doc = yaml.safe_load(self.WORKFLOW.read_text(encoding="utf-8"))
        self.assertNotIn("push", doc[True])

    def test_a_refresh_asked_for_by_hand_is_a_dispatch_with_no_answers(self) -> None:
        """The Runs tab's button, and what the workflow has to accept for it.

        It sends `{"refresh": true}`; api.php turns that into a dispatch whose
        `answers` is empty and whose `skip_refresh` is false. Both halves matter:
        an `answers` the workflow did not default to empty would make every
        automatic run carry a payload, and a `skip_refresh` left true would
        produce a run that applies nothing and then skips the refresh too --
        a green run that did nothing at all.
        """
        import yaml

        doc = yaml.safe_load(self.WORKFLOW.read_text(encoding="utf-8"))
        inputs = doc[True]["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["answers"]["default"], "")
        self.assertIs(inputs["skip_refresh"]["default"], False)

        api = (_answers.HERE / "_admin" / "api.php").read_text(encoding="utf-8")
        # Every input api.php sends is one the workflow declares, refresh or not.
        for name in ("answers", "skip_refresh"):
            self.assertIn(name, inputs, f"api.php sends an input {name} the workflow has no use for")
        # A refresh never carries answers, and never skips the refresh.
        self.assertIn("!$refresh && !empty($request['skip_refresh'])", api)

    def test_both_pages_cycle_the_same_three_themes_under_the_same_key(self) -> None:
        """The index and the admin page are one setting on one device.

        They are separate deployments -- the admin page is served from a host of
        its own -- so nothing but this check keeps the two from drifting into a
        two-state toggle on one and a three-state cycle on the other, which is
        how "system" stopped being reachable the last time.
        """
        import re

        for name in ("llm.html", "_admin/index.html"):
            page = (_answers.HERE / name).read_text(encoding="utf-8")
            with self.subTest(name):
                # Every copy, not the first: llm.html states the list twice,
                # once in the head script that paints before anything else runs
                # and once in the handler, and the two have to agree.
                lists = re.findall(r"THEMES = \[([^\]]+)\]", page)
                self.assertTrue(lists, f"{name}: no THEMES to check")
                for themes in lists:
                    self.assertEqual(
                        re.findall(r'"([a-z]+)"', themes),
                        ["system", "light", "dark"],
                        f"{name}: the cycle has to start at the default and reach it again",
                    )
                self.assertIn('localStorage.getItem("theme")', page)
                self.assertIn('localStorage.setItem("theme"', page)

    def test_the_admin_page_is_not_linked_from_the_published_one(self) -> None:
        """It is behind HTTP auth on another host, and unlisted is the point.

        The admin page links out to the index and to the repository; nothing
        points the other way. A link would put the endpoint's address on a
        public page for no gain -- whoever can use it already knows it.
        """
        import re

        page = (_answers.HERE / "llm.html").read_text(encoding="utf-8")
        # Link targets only. Prose may name the admin page -- the theme cycle
        # comment says where its twin lives -- and that is not a link to it.
        targets = re.findall(r'(?:href|src|action)="([^"]*)"', page)
        self.assertEqual([t for t in targets if "admin" in t.lower()], [], "linked from the index")


class TestShapes(AnswersTestCase):
    def test_a_record_must_be_an_object_of_a_known_kind(self) -> None:
        self.refused("just a string", "expected an object")
        self.refused({"kind": "drop-tables"}, "unknown kind")
        self.refused({"subject": "x"}, "unknown kind")

    def test_an_empty_batch_is_refused(self) -> None:
        answers, failures = _answers.validate([], llm_path=self.llm, universes=UNIVERSES)
        self.assertEqual(answers, [])
        self.assertIn("no records", failures[0].message)

    def test_a_subject_must_be_a_non_empty_string(self) -> None:
        for bad in ("", "   ", None, 7, ["x"]):
            self.refused(self.mapping(subject=bad), "non-empty string")


if __name__ == "__main__":
    unittest.main()
