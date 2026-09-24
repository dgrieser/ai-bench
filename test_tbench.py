#!/usr/bin/env python3
"""Tests for the Terminal-Bench leaderboard reader. Run with ./test_tbench.py

tbench.ai server-renders only the *current* board; 2.1 and 2.0 are a
client-side switch, and the switch reads each board from a Harbor Hub function
by (package, leaderboard name). The reader asks for every board by that pair,
so the load-bearing tests here are the version guard -- an answer naming
another board is refused rather than filed under the column that was asked
for -- and the 2.0 filter, which keeps that board's unverified submissions and
multi-model ensembles out of a first-party column.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

import fetch_tbench as tb
import update


def row(model: str, accuracy: float, agent: str = "Claude Code", effort: str = "max") -> dict:
    return {
        "id": f"row-{model}-{effort}",
        "rank": 1,
        "metadata": {
            "date": "2026-09-01",
            "agent_org": {"url": "https://example.test", "label": "Example"},
            "model_org": {"url": "https://example.test", "label": "Example"},
            "display_date": "Sep 1, 2026",
            "agent_display": {"url": "https://example.test/agent", "label": agent},
            "model_display": {"url": "https://example.test/model", "label": model},
            "reasoning_effort": effort,
        },
        "metrics": {
            "accuracy": accuracy,
            "n_trials": 330,
            "display_accuracy": f"**{accuracy}%** ± 3.0%",
            "accuracy_ci95_half_width": 3.0,
        },
        "status": "display",
    }


def row_2_0(model: str, accuracy: float, verified: bool = True, names: list | None = None) -> dict:
    """A 2.0 row: a bare-string model_display, and a verified flag."""
    return {
        "id": f"row-{model}",
        "rank": 1,
        "metadata": {
            "date": "2026-04-23",
            "verified": verified,
            "agent_org": "example",
            "model_org": "Example",
            "agent_display": {"url": "https://example.test/agent", "label": "Agent"},
            "model_display": model,
            "model_names": names or [model.lower()],
            "source_url": "https://www.tbench.ai/leaderboard/terminal-bench/2.0",
        },
        "metrics": {"accuracy": accuracy, "accuracy_ci95_half_width": 2.0},
    }


def board(key: str, rows: list[dict]) -> dict:
    spec = tb.BOARDS[key]
    return {
        "leaderboard": {
            "id": f"board-{key}",
            "name": spec["name"],
            "package": spec["package"],
            "title": f"Terminal-Bench {spec['version']}",
        },
        "rows": rows,
    }


BOARD_4_0 = board(
    "terminal_bench_4_0",
    [row("Alpha", 57.88), row("Alpha", 44.55, effort="high"), row("Beta", 41.82, agent="Codex")],
)
BOARD_2_1 = board("terminal_bench_2_1", [row("Alpha", 62.5), row("Gamma", 30.0)])
BOARD_2_0 = board(
    "terminal_bench_2_0",
    [
        row_2_0("Alpha", 70.0),
        row_2_0("Alpha", 84.7, verified=False),
        row_2_0("Multiple", 84.5, names=["alpha", "beta"]),
        row_2_0("Delta", 12.5),
    ],
)
BOARDS = {"terminal_bench_4_0": BOARD_4_0, "terminal_bench_2_1": BOARD_2_1,
          "terminal_bench_2_0": BOARD_2_0}


def served(boards: dict = BOARDS):
    """A fetch_board stand-in answering each (package, name) with its board."""
    by_pair = {(tb.BOARDS[k]["package"], tb.BOARDS[k]["name"]): b for k, b in boards.items()}
    return lambda package, name: by_pair[(package, name)]


def scores(boards: dict = BOARDS, keys: list | None = None) -> list[dict]:
    with mock.patch.object(tb, "fetch_board", side_effect=served(boards)):
        return tb.get_scores(keys)


def on(key: str, got: list[dict]) -> list[dict]:
    return [e for e in got if e["benchmark"] == key]


class TestBoards(unittest.TestCase):
    def test_every_board_feeds_a_llm_json_column(self):
        doc = json.loads(Path(__file__).resolve().with_name("llm.json").read_text())
        for key in tb.BOARDS:
            with self.subTest(key=key):
                self.assertIn(key, doc["benchmarks"])

    def test_each_board_is_credited_to_its_own_page(self):
        self.assertEqual(
            tb.board_url("terminal_bench_2_0"),
            "https://www.tbench.ai/leaderboard/terminal-bench/2.0",
        )
        self.assertEqual(tb.LEADERBOARD_URL, tb.board_url("terminal_bench_4_0"))
        self.assertEqual(len({tb.board_url(k) for k in tb.BOARDS}), len(tb.BOARDS))

    def test_every_board_is_read_by_default(self):
        self.assertEqual({e["benchmark"] for e in scores()}, set(tb.BOARDS))

    def test_one_board_can_be_read_alone(self):
        got = scores(keys=["terminal_bench_2_1"])
        self.assertEqual({e["benchmark"] for e in got}, {"terminal_bench_2_1"})


class TestVersionGuard(unittest.TestCase):
    def spec(self, key="terminal_bench_4_0"):
        return tb.BOARDS[key]["package"], tb.BOARDS[key]["name"]

    def test_accepts_the_board_it_asked_for(self):
        self.assertEqual(tb.check_version(BOARD_4_0, *self.spec())["name"], "4-0-0")

    def test_another_board_is_not_read_into_this_column(self):
        with self.assertRaises(ValueError) as ctx:
            tb.check_version(BOARD_2_1, *self.spec())
        self.assertIn("main", str(ctx.exception))

    def test_another_dataset_is_refused(self):
        other = {**BOARD_4_0, "leaderboard": {**BOARD_4_0["leaderboard"], "package": "other/bench"}}
        with self.assertRaises(ValueError):
            tb.check_version(other, *self.spec())

    def test_an_answer_without_a_descriptor_is_an_error(self):
        with self.assertRaises(ValueError):
            tb.check_version({"rows": []}, *self.spec())

    def test_get_scores_refuses_a_swapped_board(self):
        swapped = {**BOARDS, "terminal_bench_2_0": {**BOARD_2_0, "leaderboard": BOARD_2_1["leaderboard"]}}
        with self.assertRaises(ValueError):
            scores(swapped)


class TestFetchBoard(unittest.TestCase):
    def response(self, body: bytes):
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = body
        return resp

    def test_posts_the_package_and_name(self):
        with mock.patch.object(tb.urllib.request, "urlopen",
                               return_value=self.response(b'{"leaderboard": {}, "rows": []}')) as op:
            tb.fetch_board("terminal-bench/terminal-bench-2", "2-0")
        req = op.call_args[0][0]
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.full_url, tb.URL)
        self.assertEqual(
            json.loads(req.data), {"package": "terminal-bench/terminal-bench-2", "name": "2-0"}
        )

    def test_an_answer_without_rows_is_an_error(self):
        with mock.patch.object(tb.urllib.request, "urlopen",
                               return_value=self.response(b'{"error": {"message": "nope"}}')):
            with self.assertRaises(ValueError):
                tb.fetch_board("terminal-bench/terminal-bench", "4-0-0")


class TestGetScores(unittest.TestCase):
    def test_reads_accuracy_as_the_score(self):
        self.assertEqual(on("terminal_bench_4_0", scores())[0]["score"], 57.88)

    def test_every_agent_and_effort_row_is_kept(self):
        # update.py folds them onto one slug; the reader must not pre-empt that.
        alpha = [e for e in on("terminal_bench_4_0", scores()) if e["model"] == "Alpha"]
        self.assertEqual(sorted(e["effort"] for e in alpha), ["high", "max"])

    def test_the_model_label_is_the_mapping_key(self):
        self.assertEqual({e["model"] for e in on("terminal_bench_4_0", scores())}, {"Alpha", "Beta"})

    def test_agent_and_effort_are_kept_beside_the_score(self):
        beta = next(e for e in scores() if e["model"] == "Beta")
        self.assertEqual(beta["agent"], "Codex")
        self.assertEqual(beta["effort"], "max")

    def test_rows_are_ranked_best_first_within_their_board(self):
        for key in tb.BOARDS:
            with self.subTest(key=key):
                got = on(key, scores())
                self.assertEqual([e["rank"] for e in got], list(range(1, len(got) + 1)))
                self.assertEqual(
                    [e["score"] for e in got], sorted((e["score"] for e in got), reverse=True)
                )

    def test_a_row_without_an_accuracy_is_dropped(self):
        broken = row("Gamma", 0.0)
        broken["metrics"].pop("accuracy")
        boards = {**BOARDS, "terminal_bench_4_0": {**BOARD_4_0, "rows": [*BOARD_4_0["rows"], broken]}}
        self.assertNotIn("Gamma", {e["model"] for e in on("terminal_bench_4_0", scores(boards))})

    def test_a_boolean_is_not_an_accuracy(self):
        broken = row("Gamma", 0.0)
        broken["metrics"]["accuracy"] = True
        boards = {**BOARDS, "terminal_bench_4_0": {**BOARD_4_0, "rows": [*BOARD_4_0["rows"], broken]}}
        self.assertNotIn("Gamma", {e["model"] for e in on("terminal_bench_4_0", scores(boards))})

    def test_a_row_without_a_model_label_is_dropped(self):
        broken = row("Gamma", 12.0)
        broken["metadata"]["model_display"] = {"url": "https://example.test"}
        boards = {**BOARDS, "terminal_bench_4_0": {**BOARD_4_0, "rows": [*BOARD_4_0["rows"], broken]}}
        self.assertEqual(len(on("terminal_bench_4_0", scores(boards))), 3)

    def test_an_empty_board_is_an_error(self):
        with self.assertRaises(ValueError):
            scores({**BOARDS, "terminal_bench_2_1": board("terminal_bench_2_1", [])})


class TestTwoZeroBoard(unittest.TestCase):
    def test_a_bare_string_model_label_is_read(self):
        self.assertIn("Delta", {e["model"] for e in on("terminal_bench_2_0", scores())})

    def test_only_verified_runs_are_read(self):
        alpha = [e for e in on("terminal_bench_2_0", scores()) if e["model"] == "Alpha"]
        self.assertEqual([e["score"] for e in alpha], [70.0])

    def test_a_multi_model_run_is_dropped(self):
        self.assertNotIn("Multiple", {e["model"] for e in on("terminal_bench_2_0", scores())})

    def test_boards_without_the_flag_keep_every_row(self):
        self.assertEqual(len(on("terminal_bench_2_1", scores())), 2)


class TestIngest(unittest.TestCase):
    def test_each_column_gets_its_own_boards_best_run(self):
        payload = scores()
        proc = mock.MagicMock(returncode=0, stdout=json.dumps(payload), stderr="")
        mapping = {"Alpha": "alpha", "Beta": "beta", "Delta": "delta"}
        with mock.patch.object(update, "run_fetch", return_value=proc), \
                mock.patch.object(update, "load_tbench_to_slug_mapping", return_value=mapping):
            by_key = update.fetch_tbench_data(update.TBENCH_SCRIPT, Path("unused"))
        self.assertEqual(by_key["terminal_bench_4_0"]["alpha"]["score"], 57.88)
        self.assertEqual(by_key["terminal_bench_2_1"]["alpha"]["score"], 62.5)
        self.assertEqual(by_key["terminal_bench_2_0"]["alpha"]["score"], 70.0)
        self.assertNotIn("gamma", by_key["terminal_bench_2_1"])

    def test_scores_are_credited_to_their_boards_page(self):
        model = {"name": "alpha", "scores": {}, "scores_updated": {}, "scores_source": {}}
        doc = {"benchmarks": {k: {} for k in tb.BOARDS}, "models": [model]}
        by_key = {k: {"alpha": {"score": 50.0 + i}} for i, k in enumerate(tb.BOARDS)}
        matched, updated, _ = update.update_tbench_scores(doc, by_key)
        self.assertEqual((matched, updated), (1, 3))
        for key in tb.BOARDS:
            with self.subTest(key=key):
                self.assertEqual(model["scores_source"][key], tb.board_url(key))


if __name__ == "__main__":
    unittest.main()
