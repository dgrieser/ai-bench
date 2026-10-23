#!/usr/bin/env python3
"""Tests for the kernelbench.com reader. Run with ./test_kernelbench.py

Three things carry the weight here.

``test_a_cell_is_read_against_its_problems_best``: the native cell scores are
three different units (speedup, fraction of roofline, inverse latency), so a
reader that averaged them raw would let one problem's scale decide the ranking.
Each cell is divided by the best audited pass on its own problem first.

``test_a_failed_cell_counts_as_zero_over_the_full_deck``: the site averages
over every problem in the deck, a fail, hack or missing cell scoring zero; a
reader that averaged only the passes would put a model that solved one problem
well above one that solved all four.

``test_another_gpu_is_refused``: the column is the CUDA deck on the RTX PRO
6000, and a page that no longer carries that board raises instead of reading
whichever GPU is there.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

import fetch_kernelbench as kb


PROBLEMS = [{"id": "p1", "short": "a"}, {"id": "p2", "short": "b"}]


def chip(problem: str, kind: str = "pass", score: float | None = None) -> dict:
    return {"problem": problem, "kind": kind, "score": score}


def model(name: str, *chips: dict, subtitle: str = "claude · max") -> dict:
    return {"slug": name.lower().replace(" ", "-"), "name": name, "subtitle": subtitle,
            "flagged": 0, "chips": list(chips)}


def decks(rows: list[dict], gpu: str = kb.GPU, deck: str = kb.DECK) -> list[dict]:
    return [
        {"key": "mega", "byGpu": {gpu: {"problems": [{"id": "m"}], "rows": []}}},
        {"key": deck, "byGpu": {gpu: {"problems": PROBLEMS, "rows": rows}}},
    ]


def page(value: list[dict]) -> str:
    """A homepage whose flight payload carries `value` as its decks, split over
    two pushes the way Next.js streams it."""
    flight = f'1:"$Sreact.fragment"\n6:["$","div",null,{{"decks":{json.dumps(value)}}}]\n'
    half = len(flight) // 2
    pushes = "".join(
        f"<script>self.__next_f.push([1,{json.dumps(part)}])</script>"
        for part in (flight[:half], flight[half:])
    )
    return f"<html><body>{pushes}</body></html>"


def stub(html: str):
    return mock.patch.object(kb, "fetch_html", return_value=html)


def scores(*rows: dict) -> dict[str, dict]:
    with stub(page(decks(list(rows)))):
        return {entry["model"]: entry for entry in kb.get_scores()}


class TestShareOfBest(unittest.TestCase):
    def test_a_cell_is_read_against_its_problems_best(self) -> None:
        # p1 is a 30x speedup board and p2 a 0.4-of-roofline one; raw, A's 30
        # would swamp everything B does on p2.
        got = scores(
            model("A", chip("p1", score=30.0), chip("p2", score=0.2)),
            model("B", chip("p1", score=15.0), chip("p2", score=0.4)),
        )
        self.assertEqual(got["A"]["score"], 75.0)
        self.assertEqual(got["B"]["score"], 75.0)

    def test_a_failed_cell_counts_as_zero_over_the_full_deck(self) -> None:
        got = scores(
            model("A", chip("p1", score=10.0), chip("p2", "fail")),
            model("B", chip("p1", score=5.0), chip("p2", score=1.0)),
        )
        self.assertEqual(got["A"]["score"], 50.0)
        self.assertEqual(got["A"]["passed"], 1)
        self.assertEqual(got["B"]["score"], 75.0)

    def test_a_missing_cell_counts_as_zero(self) -> None:
        got = scores(model("A", chip("p1", score=10.0)), model("B", chip("p2", score=1.0)))
        self.assertEqual(got["A"]["score"], 50.0)

    def test_a_flagged_cell_neither_scores_nor_sets_the_best(self) -> None:
        # A hacked cell may still carry a number; it must not become the 1.00
        # the honest runs are measured against.
        got = scores(
            model("A", chip("p1", "hack", 100.0), chip("p2", score=1.0)),
            model("B", chip("p1", score=10.0), chip("p2", score=1.0)),
        )
        self.assertEqual(got["A"]["score"], 50.0)
        self.assertEqual(got["B"]["score"], 100.0)

    def test_a_problem_nobody_passed_scores_zero_for_everyone(self) -> None:
        got = scores(model("A", chip("p1", score=2.0), chip("p2", "fail")))
        self.assertEqual(got["A"]["score"], 50.0)

    def test_harness_and_effort_are_reported_beside_the_name(self) -> None:
        got = scores(
            model("A", chip("p1", score=1.0), subtitle="codex · xhigh"),
            model("B", chip("p1", score=1.0), subtitle="or-fable"),
        )
        self.assertEqual((got["A"]["harness"], got["A"]["effort"]), ("codex", "xhigh"))
        self.assertEqual((got["B"]["harness"], got["B"]["effort"]), ("or-fable", None))
        self.assertEqual(got["A"]["model"], "A")


class TestLayout(unittest.TestCase):
    def test_another_gpu_is_refused(self) -> None:
        with stub(page(decks([model("A", chip("p1", score=1.0))], gpu="h100"))):
            with self.assertRaisesRegex(ValueError, "no 'rtxpro6000' board"):
                kb.get_scores()

    def test_a_missing_deck_is_refused(self) -> None:
        with stub(page(decks([model("A", chip("p1", score=1.0))], deck="hard"))):
            with self.assertRaisesRegex(ValueError, "names deck 'cuda' 0 times"):
                kb.get_scores()

    def test_a_page_without_decks_is_refused(self) -> None:
        with stub("<html><body>nothing here</body></html>"):
            with self.assertRaisesRegex(ValueError, 'No "decks" array'):
                kb.get_scores()

    def test_two_different_deck_arrays_are_refused(self) -> None:
        one = page(decks([model("A", chip("p1", score=1.0))]))
        two = page(decks([model("B", chip("p1", score=1.0))]))
        with stub(one.replace("</body>", two.split("<body>")[1].split("</body>")[0] + "</body>")):
            with self.assertRaisesRegex(ValueError, "2 different"):
                kb.get_scores()

    def test_an_empty_board_is_refused(self) -> None:
        with stub(page(decks([]))):
            with self.assertRaisesRegex(ValueError, "No leaderboard rows"):
                kb.get_scores()


class TestSourceIdentity(unittest.TestCase):
    def test_the_board_ranks_as_the_benchmarks_own_leaderboard(self) -> None:
        from _precedence import KERNELBENCH_SOURCE_URL, RANK_BENCHMARK_SITE, source_rank

        self.assertEqual(source_rank(KERNELBENCH_SOURCE_URL), RANK_BENCHMARK_SITE)

    def test_the_column_is_declared(self) -> None:
        import pathlib

        doc = json.loads(pathlib.Path(kb.__file__).with_name("llm.json").read_text(encoding="utf-8"))
        self.assertIn("kernelbench_cuda", doc["benchmarks"])
        self.assertEqual(doc["benchmarks"]["kernelbench_cuda"]["urls"][0], kb.LEADERBOARD_URL)


if __name__ == "__main__":
    unittest.main()
