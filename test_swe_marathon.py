#!/usr/bin/env python3
"""Tests for the SWE-Marathon leaderboard reader. Run with ./test_swe_marathon.py

The site publishes two boards that are not comparable -- a "v1.0 Archive" and a
"v1.1 Current" whose 20 tasks were all updated -- and it stores them
differently. The archive is a plain object-literal array; the current board is
never stored as a leaderboard at all, only as the per-task trial log the page
aggregates in the browser. A scan for the leaderboard literal therefore finds
the *archive*, which is how every current-board score went missing while
archive numbers filled the column. These tests pin both halves: that each board
is found, that each is labelled with the revision the site gives it, and that
the aggregation reproduces the site's own pass@1.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

import fetch_swe_marathon as sm


def trial(reward: float, partial: float = 0.5, cost: float = 2.0, tokens: int = 1_000_000) -> dict:
    return {"reward": reward, "partial": partial, "costUsd": cost, "tokensRaw": tokens}


def config(model: str, agent: str, rewards: list[float], effort: str | None = None) -> dict:
    entry = {"agent": agent, "model": model, "trials": [trial(r) for r in rewards]}
    if effort is not None:
        entry["reasoningEffort"] = effort
    return entry


TRIAL_LOG = {
    "task-a": {"task": "task-a", "configs": [
        config("Kimi K3", "Claude Code", [1.0, 0.0], effort="max"),
        config("Kimi K3", "Claude Code", [0.0, 0.0], effort="low"),
        config("GLM 5.3", "Claude Code", [1.0, 1.0]),
    ]},
    "task-b": {"task": "task-b", "configs": [
        config("Kimi K3", "Claude Code", [1.0, 1.0], effort="max"),
        config("Kimi K3", "Claude Code", [0.0, 1.0], effort="low"),
        config("GLM 5.3", "Claude Code", [0.0, 0.0]),
    ]},
}

ARCHIVE_ARRAY = (
    "var ze=[{rank:`Ref`,id:`oracle`,name:`Oracle (held-out solution)`,"
    "scaffold:`Harbor reference`,ref:!0,pass1:84,partialAvg:84,nLoggedTrials:100},"
    "{rank:1,id:`grok45`,name:`Grok 4.5`,scaffold:`Grok Build`,pass1:29,"
    "partialAvg:null,nLoggedTrials:100},"
    "{rank:2,id:`glm52`,name:`GLM 5.2`,scaffold:`Claude Code`,pass1:13,"
    "partialAvg:null,nLoggedTrials:100}];"
)


def bundle(trial_log: dict | None = TRIAL_LOG, version: str = "v1.1") -> str:
    parts = [ARCHIVE_ARRAY]
    if trial_log is not None:
        parts.append("var Qe={version:`%s`,tasks:JSON.parse(`%s`)};"
                     % (version, json.dumps(trial_log)))
    parts.append(
        "var Et={[at]:{version:at,label:`v1.0 Archive`,status:`archive`,leaderboard:ze},"
        "[it]:{version:it,label:`v1.1 Current`,status:`current`,leaderboard:St}};"
    )
    return "".join(parts)


def scores(bundle_text: str | None = None, **kwargs) -> list[dict]:
    text = bundle() if bundle_text is None else bundle_text
    page = '<script src="/assets/index-abc.js"></script>'
    with mock.patch.object(sm, "fetch_text", side_effect=[page, text]):
        return sm.get_scores(**kwargs)


class TestBoardRevisions(unittest.TestCase):
    def test_the_site_names_which_board_is_archived(self) -> None:
        self.assertEqual(sm.board_revisions(bundle()), {"archive": "1.0", "current": "1.1"})

    def test_a_bundle_naming_no_boards_yields_nothing(self) -> None:
        self.assertEqual(sm.board_revisions("var x=1;"), {})


class TestTrialLogExtraction(unittest.TestCase):
    def test_the_log_carries_its_own_revision(self) -> None:
        logs = sm.extract_trial_logs(bundle())
        self.assertEqual([label for label, _ in logs], ["1.1"])
        self.assertEqual(sorted(logs[0][1]), ["task-a", "task-b"])

    def test_a_backtick_in_the_payload_does_not_end_it_early(self) -> None:
        """Task prose on this site really does contain backticks.

        A bundler embedding JSON in a template literal has to escape them, and
        a scan for the next bare backtick would stop mid-payload.
        """
        log = {"t": {"task": "t", "note": "run `bash timer.sh`", "configs": []}}
        embedded = json.dumps(log).replace("\\", "\\\\").replace("`", "\\`")
        text = "var Qe={version:`v1.1`,tasks:JSON.parse(`%s`)};" % embedded
        (_label, tasks), = sm.extract_trial_logs(text)
        self.assertEqual(tasks["t"]["note"], "run `bash timer.sh`")

    def test_a_log_that_is_not_json_is_an_error_not_a_silent_skip(self) -> None:
        with self.assertRaises(sm.ParseError):
            sm.extract_trial_logs("var Qe={version:`v1.1`,tasks:JSON.parse(`{oops`)};")


class TestLeaderboardFromTrials(unittest.TestCase):
    def rows(self) -> dict[tuple, dict]:
        return {
            (r["model"], r["scaffold"], r["effort"]): r
            for r in sm.leaderboard_from_trials(TRIAL_LOG)
        }

    def test_pass_at_1_is_the_share_of_fully_rewarded_trials_across_tasks(self) -> None:
        # Kimi K3 at max: 1 of 2 on task-a, 2 of 2 on task-b -> 3 of 4.
        self.assertEqual(self.rows()[("Kimi K3", "Claude Code", "max")]["score"], 75.0)
        # GLM 5.3: 2 of 2 then 0 of 2 -> 2 of 4.
        self.assertEqual(self.rows()[("GLM 5.3", "Claude Code", "max")]["score"], 50.0)

    def test_a_partial_reward_does_not_count(self) -> None:
        log = {"t": {"configs": [config("M", "A", [0.99, 0.0])]}}
        self.assertEqual(sm.leaderboard_from_trials(log)[0]["score"], 0.0)

    def test_each_reasoning_effort_is_its_own_row(self) -> None:
        rows = self.rows()
        self.assertEqual(rows[("Kimi K3", "Claude Code", "low")]["score"], 25.0)
        self.assertNotEqual(
            rows[("Kimi K3", "Claude Code", "low")]["score"],
            rows[("Kimi K3", "Claude Code", "max")]["score"],
        )

    def test_a_config_without_a_stated_effort_is_the_models_only_setting(self) -> None:
        # The site calls that one "max".
        self.assertIn(("GLM 5.3", "Claude Code", "max"), self.rows())

    def test_trials_are_counted(self) -> None:
        self.assertEqual(self.rows()[("Kimi K3", "Claude Code", "max")]["trials"], 4)


class TestGetScores(unittest.TestCase):
    def test_both_boards_are_reported_and_labelled(self) -> None:
        rows = scores()
        self.assertEqual({r["revision"] for r in rows}, {"1.0", "1.1"})

    def test_the_current_board_comes_from_the_trial_log_not_the_literal(self) -> None:
        """The regression this replaces: the literal is the *archive*.

        Reading it as the current board is what put v1.0 numbers in a column
        that also received v1.1 numbers from other sources.
        """
        # One row per configuration, so index by the effort too; the ingest's
        # best-run rule is what collapses these to one number per model.
        current = {
            (r["model"], r["effort"]): r["score"]
            for r in scores() if r["revision"] == "1.1"
        }
        self.assertEqual(current, {
            ("Kimi K3", "max"): 75.0,
            ("Kimi K3", "low"): 25.0,
            ("GLM 5.3", "max"): 50.0,
        })
        self.assertNotIn("Grok 4.5", {model for model, _ in current})

    def test_the_archived_board_is_the_stored_literal(self) -> None:
        archive = {r["model"]: r["score"] for r in scores() if r["revision"] == "1.0"}
        self.assertEqual(archive, {"Grok 4.5": 29.0, "GLM 5.2": 13.0})

    def test_reference_rows_are_dropped_by_default(self) -> None:
        self.assertNotIn("Oracle (held-out solution)", {r["model"] for r in scores()})
        with_ref = {r["model"] for r in scores(include_reference=True)}
        self.assertIn("Oracle (held-out solution)", with_ref)

    def test_rows_are_ranked_within_their_own_revision(self) -> None:
        by_revision: dict[str, list[dict]] = {}
        for row in scores():
            by_revision.setdefault(row["revision"], []).append(row)
        for revision, group in by_revision.items():
            self.assertEqual(
                [r["rank"] for r in group], list(range(1, len(group) + 1)), revision
            )

    def test_pinning_a_revision_reports_only_that_board(self) -> None:
        self.assertEqual({r["revision"] for r in scores(revision="1.0")}, {"1.0"})

    def test_pinning_an_unpublished_revision_raises(self) -> None:
        with self.assertRaises(ValueError):
            scores(revision="9.9")

    def test_a_bundle_naming_no_boards_raises(self) -> None:
        with self.assertRaises(sm.ParseError):
            scores(ARCHIVE_ARRAY)

    def test_a_revision_shipping_its_own_log_is_not_also_read_as_a_literal(self) -> None:
        """Should the archive ever ship a trial log, it must not be counted twice."""
        rows = scores(bundle(trial_log=TRIAL_LOG, version="v1.0"))
        archive = [r for r in rows if r["revision"] == "1.0"]
        self.assertEqual({r["model"] for r in archive}, {"Kimi K3", "GLM 5.3"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
