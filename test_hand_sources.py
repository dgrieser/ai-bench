#!/usr/bin/env python3
"""Tests for where a score may say it came from. Run with ./test_hand_sources.py

A hand entry is the weakest precedence rung, but in a column no scraper
reaches nothing ever replaces it, so whatever page it cites is the column's
word on that number -- and several of them led their column from a news alert,
a blog post, a gist, a personal tracker or a tweet (issue #229). Two more cited
nothing at all.

edit.py and the admin answers now refuse both: a written score needs the page
it was published on, and a news, blog or social host is not that page
(_source_quality.py). These tests hold the committed llm.json to the same rule,
so a value that reaches the file some other way -- a hand edit of the JSON, a
future writer that forgets -- is caught here rather than on the site.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from _scores import editable_benchmarks, score_source
from _source_quality import SECONDARY_HOSTS, is_secondary_source

HERE = Path(__file__).resolve().parent
LLM = HERE / "llm.json"
ADMIN = HERE / "_admin" / "index.html"


class TestIsSecondarySource(unittest.TestCase):
    def test_retellings_are_secondary(self) -> None:
        for url in (
            "https://x.com/AiBattle_/status/2061412711657857253",
            "https://twitter.com/someone/status/1",
            "https://gist.github.com/igorrivin/d594981a7712a8204932a1ab2f1b9d04",
            "https://www.datacamp.com/blog/qwen3-8-max",
            "https://aiweekly.co/alerts/something",
            "https://thenextweb.com/news/something",
            "https://atlas.kevinhu.io/models/deepseek-v4-flash",
            "https://someone.substack.com/p/scores",
            "https://WWW.EDENAI.CO/post/x",
        ):
            with self.subTest(url=url):
                self.assertTrue(is_secondary_source(url))

    def test_publications_are_not(self) -> None:
        for url in (
            None,
            "",
            "https://huggingface.co/upstage/Solar-Open2-250B",
            "https://github.com/deepseek-ai/DeepSeek-V4",
            "https://arxiv.org/abs/2508.10925",
            "https://www.anthropic.com/news/claude-sonnet-5",
            "https://cognition.ai/blog/frontier-code",
            "https://www.tbench.ai/leaderboard/terminal-bench/2.0",
            # A lookalike is not the host it contains.
            "https://notx.com/post",
            "https://x.com.example.org/post",
        ):
            with self.subTest(url=url):
                self.assertFalse(is_secondary_source(url))


class TestCommittedScores(unittest.TestCase):
    """Every number on the site names a page, and not a retelling."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = json.loads(LLM.read_text(encoding="utf-8"))
        cls.keys = set(editable_benchmarks(cls.doc))

    def scored(self):
        for model in self.doc["models"]:
            for key, value in (model.get("scores") or {}).items():
                if key in self.keys and value is not None:
                    yield model["name"], key, score_source(model, key)

    def test_every_score_names_its_source(self) -> None:
        missing = [f"{name} {key}" for name, key, url in self.scored() if url is None]
        self.assertEqual(missing, [], "scores with no source page")

    def test_no_score_cites_a_retelling(self) -> None:
        cited = [
            f"{name} {key}: {url}" for name, key, url in self.scored() if is_secondary_source(url)
        ]
        self.assertEqual(cited, [], "scores sourced from a news, blog or social page")


class TestAdminMirror(unittest.TestCase):
    def test_the_admin_ui_refuses_the_same_hosts(self) -> None:
        """A page the runner refuses takes the whole batch down, so the UI must
        catch every host answer.py would."""
        html = ADMIN.read_text(encoding="utf-8")
        block = re.search(r"const SECONDARY_HOSTS = new Set\(\[(.*?)\]\);", html, re.DOTALL)
        self.assertIsNotNone(block, "SECONDARY_HOSTS not found in _admin/index.html")
        self.assertEqual(set(re.findall(r'"([^"]+)"', block.group(1))), set(SECONDARY_HOSTS))


if __name__ == "__main__":
    unittest.main()
