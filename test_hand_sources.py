#!/usr/bin/env python3
"""Tests that every score says where it came from. Run with ./test_hand_sources.py

A hand entry is the weakest precedence rung, but in a column no scraper
reaches nothing ever replaces it, so the page it cites is all a reader can
check the number by -- and some cited nothing at all (issue #229).

edit.py and the admin answers now refuse a written score without a page. This
test holds the committed llm.json to the same rule, so a value that reaches
the file some other way -- a hand edit of the JSON, a future writer that
forgets -- is caught here rather than on the site.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from _scores import editable_benchmarks, score_source

LLM = Path(__file__).resolve().parent / "llm.json"


class TestCommittedScores(unittest.TestCase):
    def test_every_score_names_its_source(self) -> None:
        doc = json.loads(LLM.read_text(encoding="utf-8"))
        keys = set(editable_benchmarks(doc))
        missing = [
            f"{model['name']} {key}"
            for model in doc["models"]
            for key, value in (model.get("scores") or {}).items()
            if key in keys and value is not None and score_source(model, key) is None
        ]
        self.assertEqual(missing, [], "scores with no source page")


if __name__ == "__main__":
    unittest.main()
