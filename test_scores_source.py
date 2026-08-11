#!/usr/bin/env python3
"""Tests for per-score source URL bookkeeping. Run with ./test_scores_source.py

Every score write stamps models[].scores_source with the page the number was
read from, and `update.py --fill-source-urls` backfills the map for scores
written before it existed: the first source in the usual update order whose
fetched value equals the stored score claims it. These tests pin the write
rules in apply_score, the map materialization, the AA per-benchmark origin
tracking, and the repo page the derived Coding index cites instead of a
leaderboard.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

import derive_coding_index
import update
from _scores import stamp_score_source

# apply_score() reads the benchmark grid off the document; a key llm.json
# does not describe rounds to the default one printed digit.
DOC: dict = {"benchmarks": {}}
URL_A = "https://example.com/leaderboard-a"
URL_B = "https://example.com/leaderboard-b"


def model_with(score=None, source=None, key="bench") -> dict:
    return {
        "name": "m",
        "scores": {key: score},
        "scores_updated": {key: None},
        "scores_source": {key: source},
    }


class TestApplyScoreNormalMode(unittest.TestCase):
    def test_write_stamps_value_date_and_source(self) -> None:
        model = model_with()
        changes: list = []
        n = update.apply_score(DOC, model, "m", "bench", 45.0, URL_A, changes)
        self.assertEqual(n, 1)
        self.assertEqual(model["scores"]["bench"], 45.0)
        self.assertIsNotNone(model["scores_updated"]["bench"])
        self.assertEqual(model["scores_source"]["bench"], URL_A)
        self.assertEqual(changes, [("m", "bench", None, 45.0)])

    def test_overwrite_moves_the_source_with_the_score(self) -> None:
        model = model_with(score=30.0, source=URL_A)
        n = update.apply_score(DOC, model, "m", "bench", 45.0, URL_B, [])
        self.assertEqual(n, 1)
        self.assertEqual(model["scores"]["bench"], 45.0)
        self.assertEqual(model["scores_source"]["bench"], URL_B)

    def test_null_never_clobbers_value_or_source(self) -> None:
        model = model_with(score=30.0, source=URL_A)
        n = update.apply_score(DOC, model, "m", "bench", None, URL_B, [])
        self.assertEqual(n, 0)
        self.assertEqual(model["scores"]["bench"], 30.0)
        self.assertEqual(model["scores_source"]["bench"], URL_A)

    def test_equal_value_touches_nothing(self) -> None:
        model = model_with(score=45.0, source=URL_A)
        n = update.apply_score(DOC, model, "m", "bench", 45.0, URL_B, [])
        self.assertEqual(n, 0)
        self.assertEqual(model["scores_source"]["bench"], URL_A)

    def test_fill_only_never_overwrites(self) -> None:
        model = model_with(score=30.0, source=URL_A)
        n = update.apply_score(DOC, model, "m", "bench", 45.0, URL_B, [], fill_only=True)
        self.assertEqual(n, 0)
        self.assertEqual(model["scores"]["bench"], 30.0)
        self.assertEqual(model["scores_source"]["bench"], URL_A)

    def test_fill_only_fills_a_null(self) -> None:
        model = model_with()
        n = update.apply_score(DOC, model, "m", "bench", 45.0, URL_B, [], fill_only=True)
        self.assertEqual(n, 1)
        self.assertEqual(model["scores"]["bench"], 45.0)
        self.assertEqual(model["scores_source"]["bench"], URL_B)


class TestApplyScoreFillUrlsOnly(unittest.TestCase):
    def test_matching_value_fills_a_missing_url(self) -> None:
        model = model_with(score=45.0)
        changes: list = []
        n = update.apply_score(
            DOC, model, "m", "bench", 45.0, URL_A, changes, fill_urls_only=True
        )
        self.assertEqual(n, 1)
        self.assertEqual(model["scores_source"]["bench"], URL_A)
        # The score and its date stay untouched.
        self.assertEqual(model["scores"]["bench"], 45.0)
        self.assertIsNone(model["scores_updated"]["bench"])
        self.assertEqual(changes, [("m", "bench", 45.0, URL_A)])

    def test_int_float_drift_still_matches(self) -> None:
        model = model_with(score=80)
        n = update.apply_score(DOC, model, "m", "bench", 80.0, URL_A, [], fill_urls_only=True)
        self.assertEqual(n, 1)

    def test_mismatching_value_fills_nothing(self) -> None:
        model = model_with(score=45.0)
        n = update.apply_score(DOC, model, "m", "bench", 30.0, URL_A, [], fill_urls_only=True)
        self.assertEqual(n, 0)
        self.assertIsNone(model["scores_source"]["bench"])

    def test_null_fetch_fills_nothing(self) -> None:
        model = model_with(score=None)
        n = update.apply_score(DOC, model, "m", "bench", None, URL_A, [], fill_urls_only=True)
        self.assertEqual(n, 0)
        self.assertIsNone(model["scores_source"]["bench"])

    def test_existing_url_is_never_replaced_so_the_first_source_wins(self) -> None:
        model = model_with(score=45.0)
        for url in (URL_A, URL_B):  # sources in update order
            update.apply_score(DOC, model, "m", "bench", 45.0, url, [], fill_urls_only=True)
        self.assertEqual(model["scores_source"]["bench"], URL_A)

    def test_scores_and_dates_never_move_even_on_new_values(self) -> None:
        model = model_with(score=45.0)
        update.apply_score(DOC, model, "m", "bench", 99.0, URL_A, [], fill_urls_only=True)
        self.assertEqual(model["scores"]["bench"], 45.0)
        self.assertIsNone(model["scores_updated"]["bench"])


class TestEnsureScoresSource(unittest.TestCase):
    KEYS = ["alpha", "beta", "gamma"]

    def test_map_lands_directly_after_scores_updated(self) -> None:
        model = {
            "name": "m",
            "scores": {"alpha": 1},
            "scores_updated": {"alpha": "2026-01-01"},
            "vram": {},
        }
        update.ensure_scores_source(model, self.KEYS)
        self.assertEqual(
            list(model.keys()),
            ["name", "scores", "scores_updated", "scores_source", "vram"],
        )
        self.assertEqual(model["scores_source"], {k: None for k in self.KEYS})

    def test_existing_values_survive_and_keys_are_completed(self) -> None:
        model = {
            "name": "m",
            "scores_updated": {},
            "scores_source": {"beta": URL_A},
        }
        update.ensure_scores_source(model, self.KEYS)
        self.assertEqual(
            model["scores_source"], {"alpha": None, "beta": URL_A, "gamma": None}
        )

    def test_model_without_scores_updated_still_gets_the_map(self) -> None:
        model = {"name": "m"}
        update.ensure_scores_source(model, self.KEYS)
        self.assertEqual(model["scores_source"], {k: None for k in self.KEYS})


class TestMergeAaModelsOrigins(unittest.TestCase):
    def test_each_evaluation_remembers_the_slug_that_supplied_it(self) -> None:
        records = [
            {"slug": "m-base", "evaluations": {"gpqa": 0.5, "hle": None}},
            {"slug": "m-high", "evaluations": {"gpqa": 0.9, "hle": 0.3}},
        ]
        merged = update.merge_aa_models(records)
        # The leading record keeps every value it measured; the second only
        # fills the gap it left.
        self.assertEqual(merged["evaluations"], {"gpqa": 0.5, "hle": 0.3})
        self.assertEqual(
            merged["_eval_origins"], {"gpqa": "m-base", "hle": "m-high"}
        )

    def test_single_record_attributes_everything_to_itself(self) -> None:
        merged = update.merge_aa_models([{"slug": "m", "evaluations": {"gpqa": 0.5}}])
        self.assertEqual(merged["_eval_origins"], {"gpqa": "m"})


class TestUpdateSourceOrderBackfill(unittest.TestCase):
    def test_first_source_in_update_order_claims_the_score(self) -> None:
        # deepswe runs before frontierswe in update.py's order; when both would
        # produce the stored value, running the backfill in that order must
        # leave deepswe's page on the score it filled first.
        doc = {
            "models": [
                {
                    "name": "m",
                    "scores": {"deepswe": 45.0},
                    "scores_updated": {"deepswe": "2026-01-01"},
                    "scores_source": {"deepswe": None},
                }
            ]
        }
        update.update_deepswe_scores(doc, {"m": {"score": 45.0}}, fill_urls_only=True)
        # A later source reporting the same number must not re-attribute it.
        update.update_swe_rebench_scores(
            doc, {"m": {"resolved_rate": 45.0}}, fill_urls_only=True
        )
        model = doc["models"][0]
        self.assertEqual(
            model["scores_source"]["deepswe"], update.DEEPSWE_SOURCE_URL
        )
        self.assertEqual(model["scores"]["deepswe"], 45.0)
        self.assertEqual(model["scores_updated"]["deepswe"], "2026-01-01")


class TestHandEditClearsAttribution(unittest.TestCase):
    def test_stamp_score_source_accepts_none(self) -> None:
        model = model_with(score=30.0, source=URL_A)
        stamp_score_source(model, "bench", None)
        self.assertIsNone(model["scores_source"]["bench"])

    def test_non_dict_map_fails_loudly(self) -> None:
        model = {"name": "m", "scores_source": []}
        with self.assertRaises(TypeError):
            stamp_score_source(model, "bench", URL_A)


class TestPerBenchmarkUrls(unittest.TestCase):
    def test_swe_atlas_and_evals_report_resolve_specific_pages(self) -> None:
        import fetch_evals_report
        import fetch_swe_atlas

        self.assertEqual(
            set(update.SWE_ATLAS_KEY_URLS), set(fetch_swe_atlas.TRACKS.values())
        )
        self.assertEqual(
            set(update.EVALS_REPORT_KEY_URLS),
            set(fetch_evals_report.BENCHMARKS.values()),
        )
        for url in (*update.SWE_ATLAS_KEY_URLS.values(),
                    *update.EVALS_REPORT_KEY_URLS.values()):
            self.assertNotIn("?", url)
            self.assertFalse(url.endswith("/"))


class TestDerivedIndexSource(unittest.TestCase):
    """The Coding index has no leaderboard, so it cites the repo page that
    documents how it is computed -- read off the column's own urls in llm.json,
    and null on a model the index leaves unranked."""

    KEY = derive_coding_index.INDEX_KEY
    REPO_URL = "https://github.com/dgrieser/ai-bench#coding-index"

    def doc_with(self, urls: object) -> dict:
        benchmarks = {key: {} for key, _ in derive_coding_index.CONTRIBUTING}
        entry: dict = {"derived": True}
        if urls is not None:
            entry["urls"] = urls
        return {"benchmarks": {self.KEY: entry, **benchmarks}}

    def test_declared_url_is_stamped_on_ranked_models(self) -> None:
        doc = self.doc_with([self.REPO_URL])
        doc["models"] = [
            {"name": "a", "scores": {"deepswe": 40.0}},
            {"name": "b", "scores": {"deepswe": 20.0}},
        ]
        derive_coding_index.refresh(doc)
        for model in doc["models"]:
            self.assertIsNotNone(model["scores"][self.KEY])
            self.assertEqual(model["scores_source"][self.KEY], self.REPO_URL)

    def test_unranked_model_reports_no_source(self) -> None:
        doc = self.doc_with([self.REPO_URL])
        # Two scored models make deepswe rankable; the third is measured on
        # nothing, so it stays unranked and must carry neither value nor source.
        doc["models"] = [
            {"name": "a", "scores": {"deepswe": 40.0}},
            {"name": "b", "scores": {"deepswe": 20.0}},
            {"name": "c", "scores": {}, "scores_source": {self.KEY: self.REPO_URL}},
        ]
        derive_coding_index.refresh(doc)
        unranked = doc["models"][2]
        self.assertIsNone(unranked["scores"][self.KEY])
        self.assertIsNone(unranked["scores_source"][self.KEY])

    def test_falls_back_when_the_column_declares_no_url(self) -> None:
        for urls in (None, [], ["  "], "not-a-list"):
            with self.subTest(urls=urls):
                self.assertEqual(
                    derive_coding_index.source_url(self.doc_with(urls)),
                    derive_coding_index.FALLBACK_SOURCE_URL,
                )

    def test_llm_json_declares_the_repo_page(self) -> None:
        """llm.json is the one place the link is spelled; the fallback in the
        script has to agree with it or a restamp would silently repoint."""
        doc = json.loads(
            (Path(__file__).resolve().parent / "llm.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            derive_coding_index.source_url(doc),
            derive_coding_index.FALLBACK_SOURCE_URL,
        )


if __name__ == "__main__":
    unittest.main()
