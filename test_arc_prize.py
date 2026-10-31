#!/usr/bin/env python3
"""fetch_arc_prize.py: which of ARC Prize's evaluation rows reach a column.

The two files the leaderboard loads hold every dataset ARC Prize scores --
public, semi-private and private sets of three ARC-AGI versions -- and every
harness a model was run in, so the parse has to keep the semi-private boards,
drop the human panel and the Provider Adapter runs, and refuse a file whose
scale moved. All of it on in-memory rows.
"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import fetch_arc_prize as ap

MODELS = [
    {"id": "openai-gpt-6-astra-max", "displayName": "GPT-6 Astra (Max)"},
    {"id": "openai-gpt-6-astra-max-provider-adapter", "displayName": "GPT-6 Astra - Provider Adapter (Max)"},
    {"id": "deepseek-r1-0528", "displayName": "DeepSeek R1 (05/28)"},
    {"id": "2025_human_panel", "displayName": "Human Panel"},
]


def ev(dataset: str, model_id: str, score: float, display: bool = True) -> dict:
    return {"datasetId": dataset, "modelId": model_id, "score": score, "display": display}


class TestNormalize(unittest.TestCase):
    def test_an_effort_parenthetical_is_dropped_and_others_kept(self) -> None:
        self.assertEqual(ap.normalize_model("GPT-6 Astra (Max)"), "gpt-6 astra")
        self.assertEqual(ap.normalize_model("DeepSeek V4.1 Flash (None)"), "deepseek v4.1 flash")
        self.assertEqual(ap.normalize_model("DeepSeek R1 (05/28)"), "deepseek r1 (05/28)")


class TestParseRows(unittest.TestCase):
    def test_each_semi_private_set_feeds_its_version(self) -> None:
        rows = ap.parse_rows([
            ev("v3_Semi_Private", "openai-gpt-6-astra-max", 0.6271),
            ev("v2_Semi_Private", "deepseek-r1-0528", 0.011),
            ev("v2_Public_Eval", "deepseek-r1-0528", 0.02),
        ], MODELS)
        self.assertEqual(
            sorted((r["key"], r["model"], r["score"]) for r in rows),
            [("arc_agi_2", "deepseek r1 (05/28)", 1.1), ("arc_agi_3", "gpt-6 astra", 62.71)],
        )

    def test_adapter_runs_humans_and_hidden_rows_are_dropped(self) -> None:
        rows = ap.parse_rows([
            ev("v3_Semi_Private", "openai-gpt-6-astra-max-provider-adapter", 0.9855),
            ev("v1_Semi_Private", "2025_human_panel", 0.98),
            ev("v3_Semi_Private", "openai-gpt-6-astra-max", 0.62, display=False),
        ], MODELS)
        self.assertEqual(rows, [])

    def test_a_new_semi_private_version_is_reported_not_read(self) -> None:
        err = io.StringIO()
        with redirect_stderr(err):
            rows = ap.parse_rows([ev("v4_Semi_Private", "openai-gpt-6-astra-max", 0.1)], MODELS)
        self.assertEqual(rows, [])
        self.assertIn("v4_Semi_Private has no llm.json column", err.getvalue())

    def test_a_percentage_scale_is_refused(self) -> None:
        files = {
            ap.EVALUATIONS_URL: [ev(d, "openai-gpt-6-astra-max", 62.7) for d in ap.DATASETS],
            ap.MODELS_URL: MODELS,
        }
        with self.assertRaisesRegex(ValueError, "fraction"):
            ap.get_scores(files.__getitem__)

    def test_every_version_needs_rows(self) -> None:
        files = {ap.EVALUATIONS_URL: [ev("v3_Semi_Private", "openai-gpt-6-astra-max", 0.6)],
                 ap.MODELS_URL: MODELS}
        with self.assertRaises(ValueError):
            ap.get_scores(files.__getitem__)


class TestMapping(unittest.TestCase):
    def test_every_mapped_slug_is_a_model_and_no_adapter_is_mapped(self) -> None:
        here = Path(__file__).resolve().parent
        slugs = {m["name"] for m in json.loads((here / "llm.json").read_text(encoding="utf-8"))["models"]}
        mapping = json.loads(
            (here / "model-name-mapping-arc-prize-to-artificialanalysis.json").read_text(encoding="utf-8")
        )
        for name, slug in mapping.items():
            with self.subTest(name=name):
                self.assertTrue(slug.startswith("__") or slug in slugs, slug)
                self.assertNotIn("adapter", name)


if __name__ == "__main__":
    unittest.main()
