#!/usr/bin/env python3
"""Tests for the matcher and the proposal builder. Run with ./test_propose.py

The load-bearing test here is test_never_proposes_a_wrong_mapping: it replays the
mapping files' own ground truth through the matcher and fails if a proposal would
disagree with a mapping a human already made. That is the property that makes it
safe to put a proposal in a PR diff, and it is why only normalized equality is
proposable -- a substring tier passes every other test in this file and still
maps DeepSeek-V3 onto deepseek-v3-2-0925.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import _matching
import _prompts
import propose
from _openness import CLOSED_WEIGHTS, PENDING, UNMAPPABLE

HERE = Path(__file__).resolve().parent
DOC = json.loads((HERE / "llm.json").read_text(encoding="utf-8"))
MODEL_NAMES = [m["name"] for m in DOC["models"] if isinstance(m, dict) and m.get("name")]
BENCHMARK_KEYS = sorted(DOC["benchmarks"])


class TestMatcher(unittest.TestCase):
    def test_exact_only_on_normalized_equality(self) -> None:
        match, _ = _matching.propose("Kimi K2.6", ["kimi-k2-6", "kimi-k2-5"])
        self.assertIsNotNone(match)
        self.assertEqual(match.option, "kimi-k2-6")
        self.assertEqual(match.confidence, _matching.EXACT)

    def test_refuses_the_version_prefix_trap(self) -> None:
        # The whole reason the substring tier is not proposable.
        for name, options in (
            ("DeepSeek-V3", ["deepseek-v3-2-0925"]),
            ("hermes-3-70b", ["hermes-4-llama-3-1-70b"]),
            ("mimo-v2-pro", ["mimo-v2-5-pro"]),
        ):
            with self.subTest(name):
                match, alternatives = _matching.propose(name, options)
                self.assertIsNone(match, f"{name} must not be proposed")
                self.assertTrue(alternatives, "but it should still be suggestable")

    def test_prefers_the_exact_option_over_a_longer_one(self) -> None:
        match, _ = _matching.propose("deepseek-v3", ["deepseek-v3-2-0925", "deepseek-v3"])
        self.assertIsNotNone(match)
        self.assertEqual(match.option, "deepseek-v3")

    def test_no_candidates_means_no_proposal(self) -> None:
        match, alternatives = _matching.propose("Totally Made Up 9000", MODEL_NAMES)
        self.assertIsNone(match)
        self.assertEqual(alternatives, [])

    def test_ambiguity_blocks_a_proposal(self) -> None:
        # Two options normalizing to the same string: no way to pick, so don't.
        match, alternatives = _matching.propose("Foo Bar", ["foo-bar", "foo.bar"])
        self.assertIsNone(match)
        self.assertEqual(len(alternatives), 2)

    def test_never_proposes_a_wrong_mapping(self) -> None:
        """Replay every human-made mapping; a disagreeing proposal is a bug."""
        wrong, proposed, total = [], 0, 0
        for path in sorted(HERE.glob("*mapping*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not all(isinstance(v, str) for v in raw.values()):
                continue  # the AA "ignored" file maps to lists
            options = BENCHMARK_KEYS if "benchmark" in path.name else MODEL_NAMES
            for key, truth in raw.items():
                if truth in (UNMAPPABLE, CLOSED_WEIGHTS, PENDING):
                    continue
                total += 1
                subject = key.rsplit("/", 1)[-1] if "/" in key else key
                match, _ = _matching.propose(subject, options)
                if match is None:
                    continue
                proposed += 1
                if match.option != truth:
                    wrong.append((path.name, key, truth, match.option))

        self.assertGreater(total, 100, "ground truth set looks too small to be meaningful")
        self.assertGreater(proposed, 100, "matcher proposes almost nothing; check it still works")
        self.assertEqual(wrong, [], f"{len(wrong)} proposal(s) contradict a human mapping")


class TestRoutes(unittest.TestCase):
    def test_every_mapping_script_is_routed(self) -> None:
        scripts = {p.name for p in HERE.glob("update_*_mapping.py")}
        self.assertEqual(scripts - set(propose.ROUTES), set(), "unrouted source script")
        self.assertEqual(set(propose.ROUTES) - scripts, set(), "route for a script that is gone")

    def test_every_route_resolves(self) -> None:
        writers = propose.writers_by_path()
        self.assertGreaterEqual(len(writers), len(propose.ROUTES))
        for path, writer in writers.items():
            with self.subTest(path.name):
                self.assertTrue(callable(writer))

    def test_spheron_matches_on_the_model_segment(self) -> None:
        route = propose.ROUTES["update_spheron_mapping.py"]["*"]
        entry = {"subject": "moonshotai/Kimi-K2.6"}
        self.assertEqual(propose.match_subject(entry, route), "Kimi-K2.6")

    def test_other_sources_match_on_the_whole_subject(self) -> None:
        route = propose.ROUTES["update_rebench_mapping.py"]["*"]
        entry = {"subject": "org/Some-Model"}
        self.assertEqual(propose.match_subject(entry, route), "org/Some-Model")

    def test_kind_picks_the_right_llmstats_file(self) -> None:
        model = propose.route_for(
            {"command": "./update_llmstats_mapping.py -w", "kind": "llmstats-model"}
        )
        bench = propose.route_for(
            {"command": "./update_llmstats_mapping.py -w", "kind": "llmstats-benchmark"}
        )
        self.assertEqual(model.universe, propose.MODELS)
        self.assertEqual(bench.universe, propose.BENCHMARKS)
        self.assertNotEqual(model.mapping_const, bench.mapping_const)

    def test_unknown_script_is_unroutable_not_a_crash(self) -> None:
        self.assertIsNone(propose.route_for({"command": "./who_knows.py -w", "kind": "mapping"}))


class TestUniverses(unittest.TestCase):
    def test_derived_benchmarks_are_not_proposable_targets(self) -> None:
        # A source score mapped onto a derived column would be overwritten by the
        # next derivation, so it must never reach a proposal -- the interactive
        # prompts already refuse it, and the unattended path has to agree.
        universe = propose.build_universes(HERE / "llm.json", skip_aa=True)[propose.BENCHMARKS]
        derived = [
            key
            for key, benchmark in DOC["benchmarks"].items()
            if isinstance(benchmark, dict) and benchmark.get("derived") is True
        ]
        self.assertTrue(derived, "llm.json declares no derived benchmark; test is vacuous")
        for key in derived:
            with self.subTest(key):
                self.assertNotIn(key, universe)
        self.assertEqual(len(universe), len(DOC["benchmarks"]) - len(derived))


class TestPlanning(unittest.TestCase):
    def setUp(self) -> None:
        self.universes = {
            propose.MODELS: MODEL_NAMES,
            propose.BENCHMARKS: BENCHMARK_KEYS,
            propose.AA_SLUGS: [],
        }

    def entry(self, subject, script="./update_rebench_mapping.py -w", kind="mapping", **kw):
        return {"command": script, "kind": kind, "subject": subject, "candidates": [], **kw}

    def test_exact_becomes_a_real_value_weak_becomes_pending(self) -> None:
        proposals, unroutable = propose.plan_mappings(
            [self.entry("Kimi K2.6"), self.entry("DeepSeek-V3"), self.entry("Nope Nope 1")],
            self.universes,
        )
        self.assertEqual(unroutable, [])
        by_key = {p.key: p for p in proposals}
        self.assertEqual(by_key["Kimi K2.6"].value, "kimi-k2-6")
        self.assertFalse(by_key["Kimi K2.6"].pending)
        self.assertEqual(by_key["DeepSeek-V3"].value, PENDING)
        self.assertTrue(by_key["DeepSeek-V3"].alternatives, "should carry suggestions")
        self.assertEqual(by_key["Nope Nope 1"].value, PENDING)
        self.assertEqual(by_key["Nope Nope 1"].alternatives, [])

    def test_new_model_entries_are_not_mapping_proposals(self) -> None:
        proposals, unroutable = propose.plan_mappings(
            [self.entry("some-model", script="./check_new.py", kind="new-model")], self.universes
        )
        self.assertEqual(proposals, [])
        self.assertEqual(unroutable, [])

    def test_unroutable_is_collected_not_dropped(self) -> None:
        proposals, unroutable = propose.plan_mappings(
            [self.entry("x", script="./mystery.py -w")], self.universes
        )
        self.assertEqual(proposals, [])
        self.assertEqual(len(unroutable), 1)

    def test_plan_reads_the_existing_decision_off_the_file(self) -> None:
        # Real ground truth: a label a person already declined. Re-proposing it
        # would revert that answer and put the name back in the queue for good.
        proposals, _ = propose.plan_mappings(
            [
                self.entry("Agents' Last Exam", script="./update_huggingface_mapping.py -w"),
                self.entry("No Such Label 9000", script="./update_huggingface_mapping.py -w"),
            ],
            self.universes,
        )
        by_key = {p.key: p for p in proposals}
        self.assertEqual(by_key["Agents' Last Exam"].previous, UNMAPPABLE)
        self.assertTrue(by_key["Agents' Last Exam"].answered)
        self.assertIsNone(by_key["No Such Label 9000"].previous)
        self.assertFalse(by_key["No Such Label 9000"].answered)

    def test_aa_falls_back_to_recorded_candidates(self) -> None:
        # With no AA slug list (offline), the prompt's own candidates are the universe.
        entry = self.entry(
            "some-model",
            script="./update_artificialanalysis_mapping.py -w",
            kind="aa-mapping",
            candidates=["some-model", "other-model"],
        )
        proposals, _ = propose.plan_mappings([entry], self.universes)
        self.assertEqual(proposals[0].value, "some-model")


class TestSuggestions(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "m.json"

    def proposal(self, line_value=PENDING, alts=("glm-5-air", "glm-5-2", "glm-4-5-air")):
        self.path.write_text(
            '{\n  "Other": "x",\n  "GLM 5.1": %s,\n  "Last": "y"\n}\n' % json.dumps(line_value),
            encoding="utf-8",
        )
        return propose.Proposal(
            path=self.path,
            key="GLM 5.1",
            value=line_value,
            confidence="none",
            reason="no confident match",
            alternatives=list(alts),
            line=propose.find_line(self.path, "GLM 5.1"),
        )

    def test_find_line_locates_the_key(self) -> None:
        self.proposal()
        self.assertEqual(propose.find_line(self.path, "GLM 5.1"), 3)
        self.assertIsNone(propose.find_line(self.path, "Absent"))

    def test_find_line_is_not_fooled_by_a_substring_key(self) -> None:
        self.path.write_text('{\n  "GLM 5.1 Air": "a",\n  "GLM 5.1": "b"\n}\n', encoding="utf-8")
        self.assertEqual(propose.find_line(self.path, "GLM 5.1"), 3)

    def test_suggestion_replaces_the_line_verbatim(self) -> None:
        proposal = self.proposal()
        real = self.path.read_text(encoding="utf-8").splitlines()[proposal.line - 1]
        body = propose.suggestion_body(proposal, real)
        block = body.split("```suggestion\n")[1].split("\n```")[0]

        self.assertEqual(block, '  "GLM 5.1": "glm-5-air",')
        # Same indentation and same trailing comma, or GitHub produces invalid JSON.
        self.assertEqual(block[: len(block) - len(block.lstrip())], real[: len(real) - len(real.lstrip())])
        self.assertEqual(block.endswith(","), real.endswith(","))
        patched = self.path.read_text(encoding="utf-8").replace(real, block)
        self.assertEqual(json.loads(patched)["GLM 5.1"], "glm-5-air")

    def test_suggestion_mentions_the_remaining_candidates(self) -> None:
        proposal = self.proposal()
        real = self.path.read_text(encoding="utf-8").splitlines()[proposal.line - 1]
        body = propose.suggestion_body(proposal, real)
        self.assertIn("`glm-5-2`", body)
        self.assertIn("`glm-4-5-air`", body)
        self.assertIn("__unmappable__", body)

    def test_cap_is_respected_and_overflow_reported(self) -> None:
        proposals = []
        for index in range(15):
            path = self.tmp / f"m{index}.json"
            path.write_text('{\n  "K": "%s"\n}\n' % PENDING, encoding="utf-8")
            proposals.append(
                propose.Proposal(
                    path=path,
                    key="K",
                    value=PENDING,
                    confidence="none",
                    reason="",
                    alternatives=["a", "b"],
                    line=2,
                )
            )
        comments, dropped = propose.plan_suggestions(proposals, 10)
        self.assertEqual(len(comments), 10)
        self.assertEqual(dropped, 5)
        self.assertTrue(all(c["side"] == "RIGHT" and c["line"] == 2 for c in comments))

    def test_unmappable_suggestion_is_always_present(self) -> None:
        proposal = self.proposal()
        real = self.path.read_text(encoding="utf-8").splitlines()[proposal.line - 1]
        body = propose.suggestion_body(proposal, real)
        blocks = [part.split("\n```")[0] for part in body.split("```suggestion\n")[1:]]

        self.assertEqual(len(blocks), 2, "best guess plus __unmappable__")
        self.assertEqual(blocks[1], '  "GLM 5.1": "__unmappable__",')
        patched = self.path.read_text(encoding="utf-8").replace(real, blocks[1])
        self.assertEqual(json.loads(patched)["GLM 5.1"], "__unmappable__")

    def test_pending_without_candidates_still_gets_unmappable_comment(self) -> None:
        proposal = self.proposal(alts=())
        comments, dropped = propose.plan_suggestions([proposal], 10)
        self.assertEqual(len(comments), 1)
        self.assertEqual(dropped, 0)

        body = comments[0]["body"]
        blocks = [part.split("\n```")[0] for part in body.split("```suggestion\n")[1:]]
        self.assertEqual(blocks, ['  "GLM 5.1": "__unmappable__",'])
        self.assertNotIn("Best guess", body)

    def test_confident_proposals_get_no_comment(self) -> None:
        proposal = self.proposal(line_value="glm-5-air", alts=("glm-5-2",))
        comments, _ = propose.plan_suggestions([proposal], 10)
        self.assertEqual(comments, [])

    def test_line_unchanged_from_main_gets_no_comment(self) -> None:
        # Already __pending__ on main means no hunk, and GitHub 422s a review
        # that points outside the diff -- taking every other suggestion with it.
        proposal = self.proposal()
        proposal.previous = PENDING
        self.assertFalse(proposal.in_diff)
        comments, dropped = propose.plan_suggestions([proposal], 10)
        self.assertEqual(comments, [])
        self.assertEqual(dropped, 0)

    def test_newly_parked_line_still_gets_its_comment(self) -> None:
        proposal = self.proposal()
        proposal.previous = None
        self.assertTrue(proposal.in_diff)
        comments, _ = propose.plan_suggestions([proposal], 10)
        self.assertEqual(len(comments), 1)


class TestAnsweredKeys(unittest.TestCase):
    """A queued name the mapping file already decided must be left alone.

    Parking it again as __pending__ reverts the decision, and since __pending__
    is not a decision the name is queued once more on the next run: the question
    comes back forever and no answer can ever stick.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "m.json"
        self.path.write_text(
            json.dumps({"Decided": UNMAPPABLE, "Parked": PENDING}, indent=2) + "\n",
            encoding="utf-8",
        )

    def proposal(self, key: str) -> propose.Proposal:
        return propose.Proposal(
            path=self.path,
            key=key,
            value=PENDING,
            confidence="none",
            reason="no confident match",
            previous=propose.recorded_value(self.path, key),
        )

    def test_recorded_value_reads_the_file(self) -> None:
        self.assertEqual(propose.recorded_value(self.path, "Decided"), UNMAPPABLE)
        self.assertEqual(propose.recorded_value(self.path, "Parked"), PENDING)
        self.assertIsNone(propose.recorded_value(self.path, "Absent"))
        self.assertIsNone(propose.recorded_value(self.tmp / "gone.json", "Decided"))

    def test_answered_only_for_a_real_decision(self) -> None:
        self.assertTrue(self.proposal("Decided").answered)
        self.assertFalse(self.proposal("Parked").answered)
        self.assertFalse(self.proposal("Absent").answered)

    def test_apply_writes_the_open_questions_but_not_the_decided_one(self) -> None:
        written: list[tuple[str, str]] = []
        saved = propose.writers_by_path
        propose.writers_by_path = lambda: {self.path: lambda k, v: written.append((k, v))}
        try:
            propose.apply_mappings([self.proposal("Decided"), self.proposal("Parked")])
        finally:
            propose.writers_by_path = saved

        self.assertEqual(written, [("Parked", PENDING)])
        self.assertEqual(
            json.loads(self.path.read_text(encoding="utf-8"))["Decided"], UNMAPPABLE
        )

    def test_answered_keys_get_no_comment(self) -> None:
        answered = self.proposal("Decided")
        answered.line = 2
        comments, dropped = propose.plan_suggestions([answered], 10)
        self.assertEqual(comments, [])
        self.assertEqual(dropped, 0)


class TestBody(unittest.TestCase):
    def plan(self, **over):
        plan = {
            "counts": {"total": 3, "confident": 1, "pending": 1, "models": 1, "unroutable": 0},
            "dropped": {"suggestions": 0, "models": 0},
            "mappings": [
                {
                    "file": "a.json",
                    "key": "Kimi K2.6",
                    "value": "kimi-k2-6",
                    "confidence": "exact",
                    "reason": "exact match",
                    "alternatives": [],
                    "line": 4,
                },
                {
                    "file": "a.json",
                    "key": "GLM 5.1",
                    "value": PENDING,
                    "confidence": "none",
                    "reason": "no confident match",
                    "alternatives": ["glm-5-air"],
                    "line": 5,
                },
            ],
            "models": [{"name": "new-model-9b", "note": "released 2026-08-01"}],
            "unroutable": [],
            "comments": [{"path": "a.json", "line": 5, "side": "RIGHT", "body": "x"}],
        }
        plan.update(over)
        return plan

    def test_body_covers_each_section(self) -> None:
        body = propose.render_body(self.plan())
        self.assertIn("Confident mappings", body)
        self.assertIn("`kimi-k2-6`", body)
        self.assertIn("Parked as `__pending__`", body)
        self.assertIn("`glm-5-air`", body)
        self.assertIn("new-model-9b", body)
        self.assertIn("released 2026-08-01", body)

    def test_body_explains_that_pending_is_not_a_decision(self) -> None:
        body = propose.render_body(self.plan())
        self.assertIn("records nothing", body)
        self.assertIn("queued again", body)

    def test_body_reports_caps_and_unroutables(self) -> None:
        body = propose.render_body(
            self.plan(
                dropped={"suggestions": 4, "models": 2},
                unroutable=[{"command": "./mystery.py", "subject": "huh"}],
            )
        )
        self.assertIn("Left for a later run", body)
        self.assertIn("4 parked line(s) got no review comment", body)
        self.assertIn("2 new model(s) not added yet", body)
        self.assertIn("mystery.py", body)

    def test_body_reports_lines_no_suggestion_can_reach(self) -> None:
        body = propose.render_body(
            self.plan(dropped={"suggestions": 0, "models": 0, "unchanged": 2})
        )
        self.assertIn("Left for a later run", body)
        self.assertIn("2 parked line(s) were already `__pending__` on `main`", body)

    def test_body_reports_a_key_it_refused_to_touch(self) -> None:
        body = propose.render_body(
            self.plan(
                answered=[{"file": "a.json", "key": "Agents' Last Exam", "value": UNMAPPABLE}]
            )
        )
        self.assertIn("already answered, left alone", body)
        self.assertIn("Agents' Last Exam", body)
        self.assertIn(UNMAPPABLE, body)

    def test_body_counts_only_the_suggestions_it_will_attach(self) -> None:
        # The count came from the plan's comment list, so it has to move with it:
        # promising a suggestion that was never posted is what sent people
        # looking for review comments that are not there.
        self.assertIn(
            "1 of these carry a clickable suggestion", propose.render_body(self.plan())
        )
        body = propose.render_body(self.plan(comments=[]))
        self.assertNotIn("carry a clickable suggestion", body)
        self.assertIn("None of these carry a review comment", body)

    def test_body_omits_empty_sections(self) -> None:
        body = propose.render_body(self.plan(mappings=[], models=[], comments=[]))
        self.assertNotIn("Confident mappings", body)
        self.assertNotIn("New models", body)


class TestCollectModeGuard(unittest.TestCase):
    def test_propose_refuses_while_collecting(self) -> None:
        # The writers are frozen under collect mode, so applying would no-op.
        saved = os.environ.get(_prompts.ENV_VAR)
        os.environ[_prompts.ENV_VAR] = "/tmp/whatever.jsonl"
        try:
            self.assertTrue(_prompts.freeze_decisions())
        finally:
            os.environ.pop(_prompts.ENV_VAR, None)
            if saved is not None:
                os.environ[_prompts.ENV_VAR] = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
