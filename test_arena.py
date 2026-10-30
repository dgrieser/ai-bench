#!/usr/bin/env python3
"""fetch_arena.py: which rows of Arena's leaderboard dataset reach a column.

The dataset holds every category of every arena in one table per subset, so
the parse has to pick its column's category, keep the battle-count floor on the
rating boards, scale Agent Arena's signed effects, and refuse a file whose
scale or layout moved. All of it runs on in-memory rows: no download, and no
pyarrow, which only the download path needs.
"""

from __future__ import annotations

import json
import unittest

import fetch_arena as fa

SOURCE = "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/resolve/main/x.parquet"


def rating_row(name: str, rating: float, votes: int = 5000, category: str = "overall",
               license: str = "MIT") -> dict:
    return {
        "model_name": name, "organization": "org", "license": license, "rating": rating,
        "rating_lower": rating - 5, "rating_upper": rating + 5, "variance": 1.0,
        "vote_count": votes, "rank": 1, "category": category,
        "leaderboard_publish_date": "2026-10-28",
    }


def ips_row(name: str, score: float) -> dict:
    return {
        "model_name": name, "organization": "org", "license": "MIT", "score": score,
        "score_ci_lower": score - 0.01, "score_ci_upper": score + 0.01,
        "observation_count": 100000, "session_count": 1000, "rank": 1,
        "category": "overall", "leaderboard_publish_date": "2026-10-28",
    }


class TestNormalize(unittest.TestCase):
    def test_an_effort_parenthetical_is_dropped(self) -> None:
        self.assertEqual(fa.normalize_model("GLM-5.3 (Max)"), "glm-5.3")
        self.assertEqual(fa.normalize_model("Claude Opus 5.5 (High)"), "claude opus 5.5")
        self.assertEqual(fa.normalize_model("Kimi K2.5 (Thinking)"), "kimi k2.5")

    def test_a_parenthetical_naming_the_model_is_kept(self) -> None:
        self.assertEqual(fa.normalize_model("DeepSeek-R1 (0528)"), "deepseek-r1 (0528)")
        self.assertEqual(
            fa.normalize_model("GPT-5.6 Sol (XHigh, Codex Harness)"), "gpt-5.6 sol (xhigh, codex harness)"
        )
        self.assertEqual(fa.normalize_model("Nemotron (NVFP4)"), "nemotron (nvfp4)")


class TestParseBoard(unittest.TestCase):
    def test_only_the_columns_category_is_read(self) -> None:
        rows = [
            rating_row("A (Max)", 1500.4),
            rating_row("A (Max)", 1480.0, category="non_english"),
            rating_row("B", 1400.0, category="coding"),
        ]
        out = fa.parse_board("arena_text", rows, SOURCE)
        self.assertEqual([(r["model"], r["score"]) for r in out], [("a", 1500.4)])
        self.assertEqual(out[0]["source"], "https://arena.ai/leaderboard/text")
        other = fa.parse_board("arena_text_non_english", rows, SOURCE)
        self.assertEqual([r["score"] for r in other], [1480.0])

    def test_a_rating_with_too_few_votes_is_left_out(self) -> None:
        rows = [rating_row("A", 1500.0), rating_row("B", 1490.0, votes=fa.MIN_VOTES - 1)]
        out = fa.parse_board("arena_text", rows, SOURCE)
        self.assertEqual([r["model"] for r in out], ["a"])

    def test_agent_scores_are_stored_times_100(self) -> None:
        out = fa.parse_board("arena_agent", [ips_row("A", 0.2094), ips_row("B", -0.1579)], SOURCE)
        self.assertEqual([(r["model"], r["score"]) for r in out], [("a", 20.94), ("b", -15.79)])
        self.assertEqual(out[0]["source"], "https://arena.ai/leaderboard/agent")

    def test_a_rating_off_its_scale_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            fa.parse_board("arena_text", [rating_row("A", 15000.0)], SOURCE)
        with self.assertRaisesRegex(ValueError, "outside"):
            fa.parse_board("arena_agent", [ips_row("A", 20.9)], SOURCE)

    def test_a_missing_column_or_category_stops_the_read(self) -> None:
        renamed = [{**rating_row("A", 1500.0)}]
        del renamed[0]["vote_count"]
        with self.assertRaises(ValueError):
            fa.parse_board("arena_text", renamed, SOURCE)
        with self.assertRaises(ValueError):
            fa.parse_board("arena_text", [rating_row("A", 1500.0, category="coding")], SOURCE)


class TestLatestPath(unittest.TestCase):
    def test_the_one_latest_file_is_found(self) -> None:
        listing = [
            {"type": "file", "path": "webdev/full-00000-of-00001.parquet"},
            {"type": "file", "path": "webdev/latest-00000-of-00001.parquet"},
        ]
        self.assertEqual(
            fa.latest_path("webdev", fetch_json=lambda url: listing),
            "webdev/latest-00000-of-00001.parquet",
        )

    def test_none_or_several_is_refused(self) -> None:
        for listing in ([], [{"path": "webdev/latest-00000-of-00002.parquet"},
                             {"path": "webdev/latest-00001-of-00002.parquet"}]):
            with self.subTest(n=len(listing)):
                with self.assertRaises(ValueError):
                    fa.latest_path("webdev", fetch_json=lambda url, l=listing: l)


class TestGetScores(unittest.TestCase):
    def test_each_subset_is_downloaded_once(self) -> None:
        downloads: list[str] = []
        tables = {
            "text_style_control": [rating_row("A", 1500.0), rating_row("A", 1490.0, category="non_english")],
        }

        def fetch(url: str) -> bytes:
            downloads.append(url)
            subset = url.rsplit("/", 2)[-2]
            return json.dumps(tables[subset]).encode()

        listing = lambda url: [{"path": f"{url.rsplit('/', 1)[-1]}/latest-00000-of-00001.parquet"}]
        original = fa.read_parquet
        fa.read_parquet = lambda data: json.loads(data)
        try:
            rows = fa.get_scores(["arena_text", "arena_text_non_english"], fetch=fetch, fetch_json=listing)
        finally:
            fa.read_parquet = original
        self.assertEqual(len(downloads), 1)
        self.assertEqual(sorted(r["key"] for r in rows), ["arena_text", "arena_text_non_english"])


class TestMapping(unittest.TestCase):
    def test_every_mapped_slug_is_a_model(self) -> None:
        from pathlib import Path

        here = Path(__file__).resolve().parent
        slugs = {m["name"] for m in json.loads((here / "llm.json").read_text(encoding="utf-8"))["models"]}
        mapping = json.loads(
            (here / "model-name-mapping-arena-to-artificialanalysis.json").read_text(encoding="utf-8")
        )
        for name, slug in mapping.items():
            with self.subTest(name=name):
                self.assertTrue(slug.startswith("__") or slug in slugs, slug)

    def test_a_harness_run_is_never_the_model(self) -> None:
        # A row run inside a vendor's agent harness is the harness, not the
        # model, the line every other board here draws.
        import _arena_mapping

        for name, slug in _arena_mapping._load_raw_mapping().items():
            if "harness" in name:
                with self.subTest(name=name):
                    self.assertTrue(slug.startswith("__"), slug)


if __name__ == "__main__":
    unittest.main()
