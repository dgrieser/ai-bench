#!/usr/bin/env python3
"""Tests for the HLE-Diamond reader. Run with ./test_hle_diamond.py

The announcement post's chart carries three views of one benchmark -- every
model at reasoning high, each vendor at its own highest effort, and with tools
-- and only the first is the column. What these tests pin is that the reader
takes the reasoning-high table and its overall score only, drops the rows run
on the text-only subset, and stops rather than guesses when the chunk no longer
carries exactly one table of that shape or a triple's order has changed.
"""

from __future__ import annotations

import json
import unittest

import fetch_hle_diamond as hd


PAGE = (
    '<script src="/_next/static/chunks/app/blog/hle-diamond/page-66fb34af6436104e.js" async>'
    "</script>"
)
CHUNK_URL = "https://lastexam.ai/_next/static/chunks/app/blog/hle-diamond/page-66fb34af6436104e.js"

MODELS = {
    "GPT-6 Astra": {"color": "#22c55e", "logo": "/provider-logos/openai_logo.svg"},
    "Kimi K3": {"color": "#000", "logo": "/provider-logos/kimi_logo.svg"},
    "GLM 5.3": {"color": "#3853de", "logo": "/provider-logos/zai_logo.svg", "textOnly": True},
}
SCORES = {
    "GPT-6 Astra": [60.6, 75.6, 45.6],
    "Kimi K3": [22.2, 25.4, 19],
    # The text-only halves are not 500 questions each, so this is not a mean.
    "GLM 5.3": [22.6, 30.5, 15.5],
}
TOKENS = {"GPT-6 Astra": {"overall": {"mean": 11485.1, "count": 997}}}


def literal(value) -> str:
    # How the build inlines it: a single-quoted JS string, quotes escaped.
    return "JSON.parse('" + json.dumps(value).replace("'", "\\'") + "')"


def chunk(scores=SCORES, models=MODELS, extra: str = "") -> str:
    return (
        f"var f={literal(models)},x={literal(scores)},p={literal(TOKENS)};"
        f"let k=e=>e;{extra}"
    )


def fetcher(js: str):
    def fetch(url: str) -> str:
        return PAGE if url == hd.URL else js if url == CHUNK_URL else ""

    return fetch


class ReadsTheReasoningHighView(unittest.TestCase):
    def test_overall_score_per_model(self):
        rows = hd.get_scores(fetch=fetcher(chunk()))
        self.assertEqual({r["model"]: r["score"] for r in rows}, {"GPT-6 Astra": 60.6, "Kimi K3": 22.2})

    def test_text_only_rows_are_dropped(self):
        rows = hd.get_scores(fetch=fetcher(chunk()))
        self.assertNotIn("GLM 5.3", [r["model"] for r in rows])

    def test_halves_are_kept_beside_the_score(self):
        (astra,) = [r for r in hd.get_scores(fetch=fetcher(chunk())) if r["model"] == "GPT-6 Astra"]
        self.assertEqual((astra["reasoning"], astra["knowledge"]), (75.6, 45.6))

    def test_escaped_quotes_decode(self):
        models = dict(MODELS, **{"Claude Opus 5.5": {"logo": "x", "note": "it's"}})
        scores = dict(SCORES, **{"Claude Opus 5.5": [55, 63.2, 46.8]})
        rows = hd.get_scores(fetch=fetcher(chunk(scores, models)))
        self.assertIn("Claude Opus 5.5", [r["model"] for r in rows])

    def test_a_reordered_triple_stops_the_read(self):
        scores = dict(SCORES, **{"GPT-6 Astra": [75.6, 45.6, 60.6]})
        with self.assertRaises(ValueError):
            hd.get_scores(fetch=fetcher(chunk(scores)))

    def test_a_second_score_table_stops_the_read(self):
        # A max-effort table next to the high one: which is which is a guess.
        extra = "var y=" + literal({"GPT-6 Astra": [66.2, 81.6, 50.8]}) + ";"
        with self.assertRaises(ValueError):
            hd.get_scores(fetch=fetcher(chunk(extra=extra)))

    def test_a_missing_chunk_link_stops_the_read(self):
        with self.assertRaises(ValueError):
            hd.get_scores(fetch=lambda url: "<html></html>")

    def test_credited_to_the_announcement(self):
        self.assertEqual(hd.URL, "https://lastexam.ai/blog/hle-diamond")
        self.assertEqual(hd.KEY, "hle_diamond")


if __name__ == "__main__":
    unittest.main()
