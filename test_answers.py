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
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _answers
import _new_models
import _prompts
import propose
from _answers import MAPPING, MODEL_ADD, MODEL_EDIT, NEW_MODEL, AA_IGNORE, Answer, AnswerError

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
        """(answers, failures) for one record."""
        return _answers.validate(
            [record], llm_path=self.llm, universes=UNIVERSES, queue=self.queue, **kwargs
        )

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
        self.assertEqual(set(_answers.METADATA_FIELDS), {"params", "context"})
        self.refused({"kind": MODEL_EDIT, "name": "devstral-2", "fields": {"url": "x"}}, "not editable")
        self.refused({"kind": MODEL_EDIT, "name": "devstral-2", "fields": {"name": "x"}}, "not editable")

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


class TestBatches(AnswersTestCase):
    def test_a_batch_is_capped(self) -> None:
        records = [self.mapping(subject=f"m{i}") for i in range(_answers.MAX_RECORDS + 1)]
        answers, failures = _answers.validate(
            records, llm_path=self.llm, universes=UNIVERSES, queue=self.queue
        )
        self.assertEqual(answers, [])
        self.assertIn("at most", failures[0].message)

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

    def test_a_push_trigger_is_never_added(self) -> None:
        """The commits ride a deploy key, which does trigger workflow events."""
        import yaml

        doc = yaml.safe_load(self.WORKFLOW.read_text(encoding="utf-8"))
        self.assertNotIn("push", doc[True])


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
