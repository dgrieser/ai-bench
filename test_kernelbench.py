#!/usr/bin/env python3
"""Tests for the kernelbench.com reader. Run with ./test_kernelbench.py

Three things carry the weight here.

``test_the_score_is_the_mean_fraction_of_roofline``: the column stores the
Hard deck's cells as they are -- every one a fraction of the GPU's roofline --
averaged over the deck, not the site's share of the best model. The two are far
apart (21.2 against 84.2 for the board's leader), so reading the wrong one would
move every value while still looking like a plausible score.

``test_a_failed_cell_counts_as_zero_over_the_full_deck``: the mean is over
every problem in the deck, a fail, hack or missing cell scoring zero; a reader
that averaged only the passes would put a model that solved one problem well
above one that solved all six.

``test_another_gpu_is_refused``: the column is the Hard deck on the RTX PRO
6000, and a page that no longer carries that board raises instead of reading
whichever GPU is there -- another GPU is another roofline.
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
        {"key": "cuda", "byGpu": {gpu: {"problems": [{"id": "m"}], "rows": []}}},
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


class TestScore(unittest.TestCase):
    def test_the_score_is_the_mean_fraction_of_roofline(self) -> None:
        got = scores(
            model("A", chip("p1", score=0.3), chip("p2", score=0.1)),
            model("B", chip("p1", score=0.15), chip("p2", score=0.2)),
        )
        self.assertEqual(got["A"]["score"], 20.0)
        self.assertEqual(got["B"]["score"], 17.5)

    def test_the_share_of_best_rides_along_unstored(self) -> None:
        got = scores(
            model("A", chip("p1", score=0.3), chip("p2", score=0.1)),
            model("B", chip("p1", score=0.15), chip("p2", score=0.2)),
        )
        self.assertEqual(got["A"]["share"], 75.0)
        self.assertEqual(got["B"]["share"], 75.0)

    def test_a_failed_cell_counts_as_zero_over_the_full_deck(self) -> None:
        got = scores(
            model("A", chip("p1", score=0.4), chip("p2", "fail")),
            model("B", chip("p1", score=0.2), chip("p2", score=0.2)),
        )
        self.assertEqual(got["A"]["score"], 20.0)
        self.assertEqual(got["A"]["passed"], 1)
        self.assertEqual(got["B"]["score"], 20.0)

    def test_a_missing_cell_counts_as_zero(self) -> None:
        got = scores(model("A", chip("p1", score=0.4)), model("B", chip("p2", score=0.2)))
        self.assertEqual(got["A"]["score"], 20.0)

    def test_a_flagged_cell_does_not_score(self) -> None:
        # A hacked cell may still carry a number; it must not count.
        got = scores(model("A", chip("p1", "hack", 0.9), chip("p2", score=0.2)))
        self.assertEqual(got["A"]["score"], 10.0)

    def test_a_cell_above_the_roofline_is_refused(self) -> None:
        # A cell over 1.0 is a board that switched units, not a kernel faster
        # than the hardware.
        with stub(page(decks([model("A", chip("p1", score=35.0))]))):
            with self.assertRaisesRegex(ValueError, "0-1 fraction"):
                kb.get_scores()

    def test_harness_and_effort_are_reported_beside_the_name(self) -> None:
        got = scores(
            model("A", chip("p1", score=0.1), subtitle="codex · xhigh"),
            model("B", chip("p1", score=0.1), subtitle="or-fable"),
        )
        self.assertEqual((got["A"]["harness"], got["A"]["effort"]), ("codex", "xhigh"))
        self.assertEqual((got["B"]["harness"], got["B"]["effort"]), ("or-fable", None))
        self.assertEqual(got["A"]["model"], "A")


class TestLayout(unittest.TestCase):
    def test_another_gpu_is_refused(self) -> None:
        with stub(page(decks([model("A", chip("p1", score=0.1))], gpu="h100"))):
            with self.assertRaisesRegex(ValueError, "no 'rtxpro6000' board"):
                kb.get_scores()

    def test_a_missing_deck_is_refused(self) -> None:
        with stub(page(decks([model("A", chip("p1", score=0.1))], deck="cuda"))):
            with self.assertRaisesRegex(ValueError, "names deck 'hard' 0 times"):
                kb.get_scores()

    def test_a_page_without_decks_is_refused(self) -> None:
        with stub("<html><body>nothing here</body></html>"):
            with self.assertRaisesRegex(ValueError, 'No "decks" array'):
                kb.get_scores()

    def test_two_different_deck_arrays_are_refused(self) -> None:
        one = page(decks([model("A", chip("p1", score=0.1))]))
        two = page(decks([model("B", chip("p1", score=0.1))]))
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
        self.assertIn("kernelbench_hard", doc["benchmarks"])
        self.assertEqual(doc["benchmarks"]["kernelbench_hard"]["urls"][0], kb.LEADERBOARD_URL)


if __name__ == "__main__":
    unittest.main()
