#!/usr/bin/env python3
"""Tests for the HLE-Diamond reader. Run with ./test_hle_diamond.py

The announcement post carries three views of one benchmark -- each vendor at
its highest reasoning effort, every model at reasoning high, and with tools --
and only the first is the column, because every column in the table carries a
model at its highest setting. What these tests pin is that the reader takes the
post's headline results table (highest effort, no tools), never the chart's
reasoning-high literal or the with-tools table, keeps rows run on the
text-only subset but flags them, and stops rather than guesses when the table it reads is
missing or doubled.
"""

from __future__ import annotations

import json
import unittest

import fetch_hle_diamond as hd


CHUNK_URL = "https://lastexam.ai/_next/static/chunks/app/blog/hle-diamond/page-66fb34af6436104e.js"

MODELS = {
    "GPT-6 Astra": {"color": "#22c55e", "logo": "/provider-logos/openai_logo.svg"},
    "Kimi K3": {"color": "#000", "logo": "/provider-logos/kimi_logo.svg"},
    "DeepSeek V4 Pro": {"color": "#000", "logo": "/provider-logos/deepseek_logo.svg"},
    "GLM 5.3": {"color": "#3853de", "logo": "/provider-logos/zai_logo.svg", "textOnly": True},
}
RESULTS = [
    ["GPT-6 Astra", "66.2%"],
    ["GLM 5.3", "22.6%"],
    ["Kimi K3", "22.2%"],
    ["DeepSeek V4 Pro", "19.1%"],
]
SPLIT = [
    ["GPT-6 Astra", "81.6%", "50.8%", "66.2%"],
    ["GLM 5.3", "30.5%", "15.5%", "22.6%"],
    ["Kimi K3", "25.4%", "19.0%", "22.2%"],
    # Still the reasoning-high run on the live post.
    ["DeepSeek V4 Pro", "14.4%", "12.4%", "13.4%"],
]
WITH_TOOLS = [["GPT-6 Astra", "60.6%", "82.9%"]]
# The chart's reasoning-high literal, which must never be the score.
HIGH = {"GPT-6 Astra": [60.6, 75.6, 45.6], "DeepSeek V4 Pro": [13.4, 14.4, 12.4]}


def table(headers: list[str], rows: list[list[str]]) -> list:
    return ["$", "$L13", None, {"headers": headers, "rows": rows}]


def page(*tables: list) -> str:
    # How Next.js ships it: flight chunks as escaped JS strings.
    flight = json.dumps(["$", "article", None, {"children": list(tables)}])
    pushed = json.dumps("5:" + flight)[1:-1]
    return (
        '<script src="/_next/static/chunks/app/blog/hle-diamond/page-66fb34af6436104e.js" async>'
        f'</script><script>self.__next_f.push([1,"{pushed}"])</script>'
    )


def literal(value) -> str:
    return "JSON.parse('" + json.dumps(value).replace("'", "\\'") + "')"


CHUNK = f"var f={literal(MODELS)},x={literal(HIGH)};"
PAGE = page(
    table(["Model", "Accuracy"], RESULTS),
    table(["Model", "Reasoning", "Knowledge", "HLE-Diamond"], SPLIT),
    table(["Model", "Without tools", "With tools"], WITH_TOOLS),
)


def fetcher(html: str = PAGE, js: str = CHUNK):
    def fetch(url: str) -> str:
        return html if url == hd.URL else js if url == CHUNK_URL else ""

    return fetch


class ReadsTheHighestEffortTable(unittest.TestCase):
    def test_headline_score_per_model(self):
        rows = hd.get_scores(fetch=fetcher())
        self.assertEqual(
            {r["model"]: r["score"] for r in rows},
            {"GPT-6 Astra": 66.2, "GLM 5.3": 22.6, "Kimi K3": 22.2, "DeepSeek V4 Pro": 19.1},
        )

    def test_neither_reasoning_high_nor_tools_lands(self):
        scores = [r["score"] for r in hd.get_scores(fetch=fetcher())]
        for value in (60.6, 13.4, 82.9):
            self.assertNotIn(value, scores)

    def test_text_only_rows_are_kept_and_flagged(self):
        rows = {r["model"]: r for r in hd.get_scores(fetch=fetcher())}
        self.assertTrue(rows["GLM 5.3"]["text_only"])
        self.assertFalse(rows["GPT-6 Astra"]["text_only"])

    def test_split_kept_only_where_it_agrees(self):
        rows = {r["model"]: r for r in hd.get_scores(fetch=fetcher())}
        self.assertEqual((rows["GPT-6 Astra"]["reasoning"], rows["GPT-6 Astra"]["knowledge"]), (81.6, 50.8))
        self.assertIsNone(rows["DeepSeek V4 Pro"]["reasoning"])
        self.assertIsNone(rows["DeepSeek V4 Pro"]["knowledge"])

    def test_a_missing_results_table_stops_the_read(self):
        html = page(table(["Model", "Without tools", "With tools"], WITH_TOOLS))
        with self.assertRaises(ValueError):
            hd.get_scores(fetch=fetcher(html=html))

    def test_a_second_results_table_stops_the_read(self):
        html = page(table(["Model", "Accuracy"], RESULTS), table(["Model", "Accuracy"], RESULTS))
        with self.assertRaises(ValueError):
            hd.get_scores(fetch=fetcher(html=html))

    def test_a_missing_model_table_stops_the_read(self):
        with self.assertRaises(ValueError):
            hd.get_scores(fetch=fetcher(js=f"var x={literal(HIGH)};"))

    def test_a_missing_chunk_link_stops_the_read(self):
        with self.assertRaises(ValueError):
            hd.get_scores(fetch=lambda url: "<html></html>")

    def test_credited_to_the_announcement(self):
        self.assertEqual(hd.URL, "https://lastexam.ai/blog/hle-diamond")
        self.assertEqual(hd.KEY, "hle_diamond")


if __name__ == "__main__":
    unittest.main()
