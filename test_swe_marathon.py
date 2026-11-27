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


# The 1.2 shape (2026-11): the "vX Archive" label literals are gone. Versions
# are minified constants, the badge is a ternary on the current one, and 1.2
# ships its leaderboard ready-made beside a trial log of the tasks it shows.
# 1.1 stays a trial log and 1.0 the archive literal.
V12_TRIAL_LOG = {
    # Only the shown tasks: aggregating this would give GPT-7 100%, not the
    # published 82.5, which is how a test can tell which source was read.
    "task-a": {"task": "task-a", "configs": [
        config("GPT-7", "Codex", [1.0, 1.0], effort="max"),
    ]},
}
V12_OBJECT = (
    "var ct={version:`v1.2`,collection:`ff200b68`,scoredTasks:20,sampleTasks:19,"
    "nTrials:2022,leaderboard:["
    "{id:`gpt-7-codex-max`,name:`GPT-7`,displayLabel:`GPT-7 (max)`,"
    "reasoningEffort:`max`,modelVariant:`openai/gpt-7@max`,scaffold:`Codex`,"
    "pass1:82.5,tokAvg:19.232,nTasks:20,nLoggedTrials:153,minTrialsPerTask:2,"
    "pass1Label:`82.5`},"
    "{id:`kimi-k3-cc-max`,name:`Kimi K3`,displayLabel:`Kimi K3 (max)`,"
    "reasoningEffort:`max`,scaffold:`Claude Code`,pass1:58.21428571428571,"
    "tokAvg:29.665,nTasks:20,nLoggedTrials:157,pass1Label:`58.2`}],"
    "tasks:JSON.parse(`%s`)};" % json.dumps(V12_TRIAL_LOG)
)
V12_NAMES = (
    "mt=new Set([`Claude Opus 4.8`]),ht=e=>mt.has(e.model),gt=`v1.2`,_t=`v1.1`,"
    "vt=`v1.0`,yt=gt,bt=[gt,_t,vt],xt=e=>bt.includes(e)&&e!==`v1.2`,"
    "St=e=>`${e} ${e===`v1.2`?`Current`:`Archive`}`;"
)
V12_STATUS = (
    "Xt={[vt]:{version:vt,status:`archive`,resultsPending:!1,leaderboard:ze},"
    "[_t]:{version:_t,status:`archive`,resultsPending:!1,leaderboard:Bt},"
    "[gt]:{version:gt,status:`current`,resultsPending:!1,leaderboard:Jt}};"
)


def bundle_v12(status: bool = True, published_first: bool = False) -> str:
    parts = [
        ARCHIVE_ARRAY,
        "var ot={version:`v1.1`,tasks:JSON.parse(`%s`)};" % json.dumps(TRIAL_LOG),
        V12_OBJECT,
        V12_NAMES,
    ]
    if published_first:
        # The archive-literal scan must not mistake 1.2's board for 1.0's.
        parts.insert(0, parts.pop(2))
    if status:
        parts.append(V12_STATUS)
    return "".join(parts)


class TestBoardRevisions(unittest.TestCase):
    def test_the_site_names_which_board_is_archived(self) -> None:
        self.assertEqual(sm.board_revisions(bundle()), {"1.0": "archive", "1.1": "current"})

    def test_the_12_shape_names_every_board_through_its_status(self) -> None:
        self.assertEqual(
            sm.board_revisions(bundle_v12()),
            {"1.0": "archive", "1.1": "archive", "1.2": "current"},
        )

    def test_the_badge_ternary_alone_is_enough(self) -> None:
        """Without the status entries the badge still says which is current,
        and every other version the bundle names is an archive."""
        self.assertEqual(
            sm.board_revisions(bundle_v12(status=False)),
            {"1.0": "archive", "1.1": "archive", "1.2": "current"},
        )

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


class TestVersion12Shape(unittest.TestCase):
    """The bundle as it stands since 1.2: three boards, three storage shapes."""

    def by_revision(self, text: str | None = None) -> dict[str, dict[tuple, float]]:
        out: dict[str, dict[tuple, float]] = {}
        for r in scores(bundle_v12() if text is None else text):
            out.setdefault(r["revision"], {})[(r["model"], r["effort"])] = r["score"]
        return out

    def test_all_three_boards_are_read(self) -> None:
        self.assertEqual(set(self.by_revision()), {"1.0", "1.1", "1.2"})

    def test_the_published_12_leaderboard_is_read_as_published(self) -> None:
        """Not re-aggregated from its trial log, which lacks the held-out task."""
        self.assertEqual(self.by_revision()["1.2"], {
            ("GPT-7", "max"): 82.5,
            ("Kimi K3", "max"): 58.21,
        })

    def test_11_is_still_aggregated_from_its_trial_log(self) -> None:
        self.assertEqual(self.by_revision()["1.1"][("Kimi K3", "max")], 75.0)

    def test_10_is_still_the_archive_literal(self) -> None:
        self.assertEqual(self.by_revision()["1.0"], {
            ("Grok 4.5", None): 29.0, ("GLM 5.2", None): 13.0,
        })

    def test_the_archive_literal_is_not_confused_with_the_published_board(self) -> None:
        self.assertEqual(
            self.by_revision(bundle_v12(published_first=True))["1.0"],
            {("Grok 4.5", None): 29.0, ("GLM 5.2", None): 13.0},
        )

    def test_12_rows_carry_their_effort_and_label(self) -> None:
        row = next(r for r in scores(bundle_v12()) if r["revision"] == "1.2" and r["rank"] == 1)
        self.assertEqual(
            (row["model"], row["scaffold"], row["effort"], row["display_label"], row["trials"]),
            ("GPT-7", "Codex", "max", "GPT-7 (max)", 153),
        )

    def test_pinning_12(self) -> None:
        self.assertEqual({r["revision"] for r in scores(bundle_v12(), revision="1.2")}, {"1.2"})

    def test_published_leaderboards_reports_its_offset(self) -> None:
        text = bundle_v12()
        ((label, start, rows),) = sm.published_leaderboards(text)
        self.assertEqual(label, "1.2")
        self.assertEqual(text[start], "[")
        self.assertEqual(len(rows), 2)

    def test_two_archives_without_data_of_their_own_skip_the_literal(self) -> None:
        """A literal that could belong to either archive is not guessed at."""
        text = ARCHIVE_ARRAY + V12_OBJECT + V12_NAMES + V12_STATUS
        self.assertEqual({r["revision"] for r in scores(text)}, {"1.2"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
