#!/usr/bin/env python3
"""Tests for the kernelbench.com reader. Run with ./test_kernelbench.py

The reader fills two columns from two pages: ``kernelbench_mega`` from the
homepage's Mega deck and ``kernelbench_hard`` from /hard, the frozen archive the
site moved the retired Hard deck to on 2026-11-23. Four things carry the weight.

``test_the_hard_score_is_the_mean_fraction_of_roofline``: Hard stores its
cells as they are -- every one a fraction of the GPU's roofline -- averaged over
the deck, not the site's share of the best model. The two are far apart (21.2
against 84.2 for the board's leader), so reading the wrong one would move every
value while still looking like a plausible score.

``test_a_failed_cell_counts_as_zero_over_the_full_deck``: the mean is over
every problem in the deck, a fail, hack or missing cell scoring zero; a reader
that averaged only the passes would put a model that solved one problem well
above one that solved all six.

``test_hard_is_read_from_its_archive_not_the_homepage``: the homepage no longer
carries Hard (the failure that retired the old reader), so each board is read
from its own page and cites it.

``test_a_cell_in_another_unit_is_refused``: a deck that mixes units -- CUDA's
NSA latency beside two fractions of roofline -- or switches unit raises rather
than being averaged.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

import fetch_kernelbench as kb


PROBLEMS = [{"id": "p1", "short": "a"}, {"id": "p2", "short": "b"}]
MEGA_PROBLEMS = [{"id": "m1", "short": "kimi"}]


def chip(problem: str, kind: str = "pass", score: float | None = None,
         title: str | None = None) -> dict:
    if title is None and score is not None:
        title = (f"{score:.2f}× vs baseline" if problem.startswith("m")
                 else f"{100 * score:.2f}% of roofline")
    return {"problem": problem, "kind": kind, "score": score, "title": title}


def model(name: str, *chips: dict, subtitle: str = "claude · max") -> dict:
    return {"slug": name.lower().replace(" ", "-"), "name": name, "subtitle": subtitle,
            "flagged": 0, "chips": list(chips)}


def home_decks(rows: list[dict], gpu: str = kb.GPU, deck: str = "mega") -> list[dict]:
    """The homepage since 2026-11-23: Mega, CUDA and an empty Multi; no Hard."""
    return [
        {"key": deck, "byGpu": {gpu: {"problems": MEGA_PROBLEMS, "rows": rows}}},
        {"key": "cuda", "byGpu": {gpu: {"problems": [{"id": "c"}], "rows": []}}},
        {"key": "multi", "byGpu": {}},
    ]


def hard_decks(rows: list[dict], gpu: str = kb.GPU, deck: str = "hard") -> list[dict]:
    """The /hard archive: one deck."""
    return [{"key": deck, "title": "Hard (archived)",
             "byGpu": {gpu: {"problems": PROBLEMS, "rows": rows}}}]


def page(value: list[dict]) -> str:
    """A page whose flight payload carries `value` as its decks, split over
    two pushes the way Next.js streams it."""
    flight = f'1:"$Sreact.fragment"\n6:["$","div",null,{{"decks":{json.dumps(value)}}}]\n'
    half = len(flight) // 2
    pushes = "".join(
        f"<script>self.__next_f.push([1,{json.dumps(part)}])</script>"
        for part in (flight[:half], flight[half:])
    )
    return f"<html><body>{pushes}</body></html>"


DEFAULT_MEGA = [model("M", chip("m1", score=20.0))]


def stub(hard_html: str, home_html: str | None = None):
    home_html = home_html if home_html is not None else page(home_decks(DEFAULT_MEGA))
    pages = {kb.LEADERBOARD_URL: home_html, kb.HARD_URL: hard_html}
    return mock.patch.object(kb, "fetch_html", side_effect=lambda url, *a, **k: pages[url])


def hard_scores(*rows: dict) -> dict[str, dict]:
    with stub(page(hard_decks(list(rows)))):
        return {e["model"]: e for e in kb.get_scores() if e["benchmark"] == kb.HARD_KEY}


def mega_scores(*rows: dict) -> dict[str, dict]:
    with stub(page(hard_decks([model("H", chip("p1", score=0.1))])),
              page(home_decks(list(rows)))):
        return {e["model"]: e for e in kb.get_scores() if e["benchmark"] == kb.MEGA_KEY}


class TestHardScore(unittest.TestCase):
    def test_the_hard_score_is_the_mean_fraction_of_roofline(self) -> None:
        got = hard_scores(
            model("A", chip("p1", score=0.3), chip("p2", score=0.1)),
            model("B", chip("p1", score=0.15), chip("p2", score=0.2)),
        )
        self.assertEqual(got["A"]["score"], 20.0)
        self.assertEqual(got["B"]["score"], 17.5)

    def test_the_share_of_best_rides_along_unstored(self) -> None:
        got = hard_scores(
            model("A", chip("p1", score=0.3), chip("p2", score=0.1)),
            model("B", chip("p1", score=0.15), chip("p2", score=0.2)),
        )
        self.assertEqual(got["A"]["share"], 75.0)
        self.assertEqual(got["B"]["share"], 75.0)

    def test_a_failed_cell_counts_as_zero_over_the_full_deck(self) -> None:
        got = hard_scores(
            model("A", chip("p1", score=0.4), chip("p2", "fail")),
            model("B", chip("p1", score=0.2), chip("p2", score=0.2)),
        )
        self.assertEqual(got["A"]["score"], 20.0)
        self.assertEqual(got["A"]["passed"], 1)
        self.assertEqual(got["B"]["score"], 20.0)

    def test_a_missing_cell_counts_as_zero(self) -> None:
        got = hard_scores(model("A", chip("p1", score=0.4)), model("B", chip("p2", score=0.2)))
        self.assertEqual(got["A"]["score"], 20.0)

    def test_a_flagged_cell_does_not_score(self) -> None:
        # A hacked cell may still carry a number; it must not count.
        got = hard_scores(model("A", chip("p1", "hack", 0.9), chip("p2", score=0.2)))
        self.assertEqual(got["A"]["score"], 10.0)

    def test_a_cell_above_the_roofline_is_refused(self) -> None:
        # A cell over 1.0 is a board that switched units, not a kernel faster
        # than the hardware.
        rows = [model("A", chip("p1", score=35.0, title="35% of roofline"))]
        with stub(page(hard_decks(rows))):
            with self.assertRaisesRegex(ValueError, "0-1 fraction"):
                kb.get_scores()

    def test_harness_and_effort_are_reported_beside_the_name(self) -> None:
        got = hard_scores(
            model("A", chip("p1", score=0.1), subtitle="codex · xhigh"),
            model("B", chip("p1", score=0.1), subtitle="or-fable"),
        )
        self.assertEqual((got["A"]["harness"], got["A"]["effort"]), ("codex", "xhigh"))
        self.assertEqual((got["B"]["harness"], got["B"]["effort"]), ("or-fable", None))
        self.assertEqual(got["A"]["model"], "A")


class TestMegaScore(unittest.TestCase):
    def test_the_mega_score_is_the_mean_speedup(self) -> None:
        got = mega_scores(
            model("A", chip("m1", score=36.655)),
            model("B", chip("m1", score=12.055)),
        )
        self.assertEqual(got["A"]["score"], 36.66)
        self.assertEqual(got["B"]["score"], 12.05)
        self.assertEqual(got["B"]["share"], round(100 * 12.055 / 36.655, 2))

    def test_a_failed_mega_cell_scores_zero(self) -> None:
        got = mega_scores(model("A", chip("m1", score=20.0)), model("B", chip("m1", "numerics")))
        self.assertEqual(got["B"]["score"], 0.0)
        self.assertEqual(got["B"]["passed"], 0)

    def test_a_cell_in_another_unit_is_refused(self) -> None:
        # CUDA-style: a latency caption on a deck read as speedup.
        rows = [model("A", chip("m1", score=1.36, title="0.072 ms geomean latency"))]
        with stub(page(hard_decks([model("H", chip("p1", score=0.1))])), page(home_decks(rows))):
            with self.assertRaisesRegex(ValueError, "unit changed"):
                kb.get_scores()

    def test_a_roofline_caption_on_the_hard_deck_is_required(self) -> None:
        rows = [model("A", chip("p1", score=0.5, title="0.50× vs baseline"))]
        with stub(page(hard_decks(rows))):
            with self.assertRaisesRegex(ValueError, "unit changed"):
                kb.get_scores()


class TestLayout(unittest.TestCase):
    def test_hard_is_read_from_its_archive_not_the_homepage(self) -> None:
        with stub(page(hard_decks([model("A", chip("p1", score=0.1))]))) as fetch:
            rows = kb.get_scores()
        self.assertEqual(
            sorted({call.args[0] for call in fetch.call_args_list}),
            sorted({kb.LEADERBOARD_URL, kb.HARD_URL}),
        )
        self.assertEqual({r["benchmark"] for r in rows}, {kb.HARD_KEY, kb.MEGA_KEY})
        self.assertEqual(kb.source_url(kb.HARD_KEY), "https://kernelbench.com/hard")
        self.assertEqual(kb.source_url(kb.MEGA_KEY), "https://kernelbench.com/")

    def test_another_gpu_is_refused(self) -> None:
        with stub(page(hard_decks([model("A", chip("p1", score=0.1))], gpu="h100"))):
            with self.assertRaisesRegex(ValueError, "no 'rtxpro6000' board"):
                kb.get_scores()

    def test_a_missing_deck_is_refused(self) -> None:
        with stub(page(hard_decks([model("A", chip("p1", score=0.1))], deck="cuda"))):
            with self.assertRaisesRegex(ValueError, "names deck 'hard' 0 times"):
                kb.get_scores()

    def test_a_page_without_decks_is_retried_then_refused(self) -> None:
        with stub("<html><body>Vercel checkpoint</body></html>"), \
                mock.patch.object(kb.time, "sleep") as sleep:
            with self.assertRaisesRegex(ValueError, 'No "decks" array'):
                kb.get_scores()
        self.assertEqual(sleep.call_count, 2)

    def test_a_checkpoint_page_is_retried(self) -> None:
        good = page(hard_decks([model("A", chip("p1", score=0.1))]))
        answers = iter(["<html>checkpoint</html>", good])
        with mock.patch.object(kb, "fetch_html", side_effect=lambda url: next(answers)), \
                mock.patch.object(kb.time, "sleep"):
            decks = kb.fetch_decks(kb.HARD_URL)
        self.assertEqual(decks[0]["key"], "hard")

    def test_two_different_deck_arrays_are_refused(self) -> None:
        one = page(hard_decks([model("A", chip("p1", score=0.1))]))
        two = page(hard_decks([model("B", chip("p1", score=0.1))]))
        with stub(one.replace("</body>", two.split("<body>")[1].split("</body>")[0] + "</body>")):
            with self.assertRaisesRegex(ValueError, "2 different"):
                kb.get_scores()

    def test_an_empty_board_is_refused(self) -> None:
        with stub(page(hard_decks([]))):
            with self.assertRaisesRegex(ValueError, "No leaderboard rows"):
                kb.get_scores()


class TestSourceIdentity(unittest.TestCase):
    def test_both_boards_rank_as_the_benchmarks_own_leaderboard(self) -> None:
        from _precedence import (
            KERNELBENCH_KEY_URLS, KERNELBENCH_SOURCE_URL, RANK_BENCHMARK_SITE, source_rank,
        )

        self.assertEqual(source_rank(KERNELBENCH_SOURCE_URL), RANK_BENCHMARK_SITE)
        for url in KERNELBENCH_KEY_URLS.values():
            self.assertEqual(source_rank(url), RANK_BENCHMARK_SITE)

    def test_the_columns_are_declared(self) -> None:
        import pathlib

        doc = json.loads(pathlib.Path(kb.__file__).with_name("llm.json").read_text(encoding="utf-8"))
        for key in kb.KEYS:
            self.assertIn(key, doc["benchmarks"])
            self.assertEqual(doc["benchmarks"][key]["urls"][0], kb.source_url(key))

    def test_mega_is_not_on_a_percentage_scale(self) -> None:
        # Speedups run past 100x in principle; the default 0-100 range would
        # refuse them.
        import pathlib

        doc = json.loads(pathlib.Path(kb.__file__).with_name("llm.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(doc["benchmarks"][kb.MEGA_KEY]["range"][1], kb.MAX_SPEEDUP)


if __name__ == "__main__":
    unittest.main()
