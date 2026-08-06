#!/usr/bin/env python3
"""Tests for the interactive provenance backfill. Run with ./test_fill_missing_source_urls.py

fill_missing_source_urls.py asks a person for the dates and source URLs nothing in the
pipeline can derive. These tests pin what counts as a gap (a non-null score with
no date or no source, a VRAM figure with no page, a missing model/creator/
benchmark URL), the candidate ranking -- above all that another model's own page
is never offered -- and that each answer lands in the right place.
"""

from __future__ import annotations

import unittest
from datetime import date

import fill_missing_source_urls as fp

HF_A = "https://huggingface.co/org/Model-A"
HF_B = "https://huggingface.co/org/Model-B"
BOARD = "https://example.com/leaderboard"
CREATOR = "https://example.com/creator"


def doc_with(*models: dict, benchmarks: dict | None = None) -> dict:
    return {
        "benchmarks": benchmarks or {"bench": {"name": "Bench", "urls": [BOARD]}},
        "models": list(models),
    }


def model(name: str, **overrides) -> dict:
    base = {
        "name": name,
        "date_added": "2026-01-01",
        "url": f"https://huggingface.co/org/{name}",
        "creator": {"name": "Org", "url": CREATOR},
        "scores": {"bench": 50.0},
        "scores_updated": {"bench": "2026-01-01"},
        "scores_source": {"bench": BOARD},
    }
    base.update(overrides)
    return base


def gaps_of(doc: dict, names: list[str] | None = None) -> list[fp.Gap]:
    return fp.collect_gaps(doc, names or [], set(fp.KINDS))


class TestGapDetection(unittest.TestCase):
    def test_no_gaps_on_a_complete_model(self) -> None:
        self.assertEqual(gaps_of(doc_with(model("a"))), [])

    def test_null_score_needs_neither_date_nor_source(self) -> None:
        m = model("a", scores={"bench": None}, scores_updated={"bench": None},
                  scores_source={"bench": None})
        self.assertEqual(gaps_of(doc_with(m)), [])

    def test_score_without_date_or_source_is_two_gaps(self) -> None:
        m = model("a", scores_updated={"bench": None}, scores_source={"bench": None})
        self.assertEqual(
            {gap.kind for gap in gaps_of(doc_with(m))}, {"score-date", "score-source"}
        )

    def test_blank_string_counts_as_missing(self) -> None:
        m = model("a", scores_source={"bench": "   "})
        self.assertEqual([gap.kind for gap in gaps_of(doc_with(m))], ["score-source"])

    def test_derived_benchmark_is_never_a_gap(self) -> None:
        doc = doc_with(
            model("a", scores={"idx": 1.0}, scores_updated={"idx": None},
                  scores_source={"idx": None}),
            benchmarks={"idx": {"name": "Index", "urls": [BOARD], "derived": True}},
        )
        self.assertEqual(gaps_of(doc), [])

    def test_vram_figure_without_a_page_is_a_gap(self) -> None:
        m = model("a", vram={"fp16": 20, "int8": 10}, vram_source={"fp16": BOARD})
        gaps = gaps_of(doc_with(m))
        self.assertEqual([gap.label for gap in gaps], ["a.vram.int8"])

    def test_null_vram_figure_needs_no_page(self) -> None:
        m = model("a", vram={"fp16": None}, vram_source={})
        self.assertEqual(gaps_of(doc_with(m)), [])

    def test_missing_model_creator_and_benchmark_urls(self) -> None:
        doc = doc_with(
            model("a", url="", creator={"name": "Org", "url": None}),
            benchmarks={"bench": {"name": "Bench", "urls": []}},
        )
        self.assertEqual(
            {gap.kind for gap in gaps_of(doc)},
            {"model-url", "creator-url", "benchmark-urls"},
        )

    def test_missing_date_added_is_left_to_add_py(self) -> None:
        m = model("a")
        del m["date_added"]
        self.assertEqual(gaps_of(doc_with(m)), [])

    def test_legacy_scalar_benchmark_url_is_not_a_gap(self) -> None:
        doc = doc_with(model("a"), benchmarks={"bench": {"name": "Bench", "url": BOARD}})
        self.assertEqual(gaps_of(doc), [])

    def test_model_filter_drops_other_models_and_benchmarks(self) -> None:
        doc = doc_with(
            model("a", scores_source={"bench": None}),
            model("b", scores_source={"bench": None}),
            benchmarks={"bench": {"name": "Bench", "urls": []}},
        )
        self.assertEqual([gap.label for gap in gaps_of(doc, ["a"])], ["a.bench"])

    def test_kind_filter_keeps_only_that_kind(self) -> None:
        m = model("a", scores_updated={"bench": None}, scores_source={"bench": None})
        gaps = fp.collect_gaps(doc_with(m), [], {"score-date"})
        self.assertEqual([gap.kind for gap in gaps], ["score-date"])


class TestCandidates(unittest.TestCase):
    def source_candidates(self, doc: dict, label: str) -> list[str]:
        gap = next(g for g in gaps_of(doc) if g.label == label and g.kind == "score-source")
        return [value for value, _ in gap.candidates]

    def test_benchmark_page_is_offered_first(self) -> None:
        doc = doc_with(model("a", scores_source={"bench": None}))
        self.assertEqual(self.source_candidates(doc, "a.bench")[0], BOARD)

    def test_another_models_page_is_never_offered(self) -> None:
        # b sourced its score from its own HuggingFace card, so that page is the
        # most-cited source for "bench" -- and wrong for every other model.
        doc = doc_with(
            model("a", url=HF_A, scores_source={"bench": None}),
            model("b", url=HF_B, scores_source={"bench": HF_B}),
            benchmarks={"bench": {"name": "Bench", "urls": []}},
        )
        candidates = self.source_candidates(doc, "a.bench")
        self.assertNotIn(HF_B, candidates)
        self.assertIn(HF_A, candidates)

    def test_shared_leaderboard_of_other_models_is_offered(self) -> None:
        doc = doc_with(
            model("a", scores={"bench": 1.0, "other": 2.0},
                  scores_updated={"bench": "2026-01-01", "other": "2026-01-01"},
                  scores_source={"bench": None, "other": None}),
            model("b", scores={"bench": 2.0}, scores_updated={"bench": "2026-01-01"},
                  scores_source={"bench": BOARD}),
            benchmarks={"bench": {"name": "Bench", "urls": []},
                        "other": {"name": "Other", "urls": []}},
        )
        self.assertIn(BOARD, self.source_candidates(doc, "a.bench"))

    def test_own_source_urls_are_offered_for_the_next_score(self) -> None:
        m = model(
            "a",
            scores={"bench": 1.0, "other": 2.0},
            scores_updated={"bench": "2026-01-01", "other": "2026-01-01"},
            scores_source={"bench": BOARD, "other": None},
        )
        doc = doc_with(m, benchmarks={"bench": {"name": "Bench", "urls": []},
                                      "other": {"name": "Other", "urls": []}})
        self.assertIn(BOARD, self.source_candidates(doc, "a.other"))

    def test_creator_url_comes_from_the_same_creator(self) -> None:
        doc = doc_with(
            model("a", creator={"name": "Org", "url": None}),
            model("b"),
        )
        gap = next(g for g in gaps_of(doc) if g.kind == "creator-url")
        self.assertEqual([value for value, _ in gap.candidates], [CREATOR])

    def test_spheron_page_is_offered_for_a_vram_gap(self) -> None:
        doc = doc_with(model("a", url=HF_A, vram={"fp16": 20}, vram_source={}))
        gap = next(g for g in gaps_of(doc) if g.kind == "vram-source")
        self.assertIn(
            "https://www.spheron.network/tools/gpu-recommender/org/Model-A/",
            [value for value, _ in gap.candidates],
        )

    def test_today_is_offered_for_a_missing_date(self) -> None:
        doc = doc_with(model("a", scores_updated={"bench": None}))
        gap = next(g for g in gaps_of(doc) if g.kind == "score-date")
        self.assertEqual(gap.candidates[0][0], date.today().isoformat())

    def test_candidates_are_deduped_and_capped(self) -> None:
        self.assertEqual(
            fp.dedupe([(BOARD, "first"), (BOARD, "second"), ("", "blank")]),
            [(BOARD, "first")],
        )
        many = [(f"https://example.com/{i}", "why") for i in range(20)]
        self.assertEqual(len(fp.dedupe(many)), fp.MAX_CANDIDATES)


class TestApply(unittest.TestCase):
    def apply_one(self, doc: dict, kind: str, value: str) -> None:
        gap = next(g for g in gaps_of(doc) if g.kind == kind)
        gap.apply(value)

    def test_score_date_lands_in_scores_updated(self) -> None:
        m = model("a", scores_updated={"bench": None})
        self.apply_one(doc_with(m), "score-date", "2026-02-03")
        self.assertEqual(m["scores_updated"]["bench"], "2026-02-03")

    def test_score_source_lands_in_scores_source(self) -> None:
        m = model("a", scores_source={"bench": None})
        self.apply_one(doc_with(m), "score-source", BOARD)
        self.assertEqual(m["scores_source"]["bench"], BOARD)

    def test_vram_source_lands_next_to_its_quant(self) -> None:
        m = model("a", vram={"fp16": 20}, vram_source={})
        self.apply_one(doc_with(m), "vram-source", BOARD)
        self.assertEqual(m["vram_source"], {"fp16": BOARD})

    def test_vram_source_map_is_created_when_absent(self) -> None:
        m = model("a", vram={"fp16": 20})
        self.apply_one(doc_with(m), "vram-source", BOARD)
        self.assertEqual(m["vram_source"], {"fp16": BOARD})

    def test_non_dict_vram_source_raises(self) -> None:
        m = model("a", vram={"fp16": 20}, vram_source=[])
        with self.assertRaises(TypeError):
            fp.set_vram_source(m, "fp16", BOARD)

    def test_model_and_creator_urls_land_on_the_model(self) -> None:
        m = model("a", url="", creator={"name": "Org", "url": None})
        doc = doc_with(m)
        self.apply_one(doc, "model-url", HF_A)
        self.apply_one(doc, "creator-url", CREATOR)
        self.assertEqual(m["url"], HF_A)
        self.assertEqual(m["creator"]["url"], CREATOR)

    def test_benchmark_url_replaces_the_legacy_scalar(self) -> None:
        entry = {"name": "Bench", "urls": [], "url": ""}
        fp.set_benchmark_url(entry, BOARD)
        self.assertEqual(entry["urls"], [BOARD])
        self.assertNotIn("url", entry)


class TestParsers(unittest.TestCase):
    def test_url_must_be_http(self) -> None:
        self.assertEqual(fp.parse_url(f"  {BOARD} "), BOARD)
        with self.assertRaises(ValueError):
            fp.parse_url("example.com/x")

    def test_date_must_be_iso(self) -> None:
        self.assertEqual(fp.parse_date(" 2026-02-03 "), "2026-02-03")
        with self.assertRaises(ValueError):
            fp.parse_date("03.02.2026")


if __name__ == "__main__":
    unittest.main()
