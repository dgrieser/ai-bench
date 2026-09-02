#!/usr/bin/env python3
"""Tests for source precedence. Run with ./test_precedence.py

Most columns in llm.json have more than one publisher, so every refresh has to
decide whose number lands. That used to be settled by the order update.py calls
its ingests -- last writer wins -- which made the file depend on which ingests
ran: the AA-only pass in 2ab9ab0 replaced four evals.report values with
Artificial Analysis' own, and the next full refresh put evals.report back.

_precedence.py declares the rank instead and apply_score() enforces it. These
tests pin the rungs, the two same-host families that have to be told apart, and
the property the change exists for: the stored value is the same whichever
subset of the ingests runs, and in whatever order.
"""

from __future__ import annotations

import unittest

import _precedence as precedence
import update
from _precedence import (
    AA_CODING_AGENTS_SOURCE_URL,
    BFCL_SOURCE_URL,
    DATACURVE_SOURCE_URL,
    DEEPSWE_SOURCE_URL,
    EVALS_REPORT_KEY_URLS,
    FRONTIERCODE_SOURCE_URL,
    FRONTIERSWE_SOURCE_URL,
    HUGGING_FACE_PREFIX,
    LLMSTATS_SOURCE_URL,
    MCP_ATLAS_SOURCE_URL,
    OSWORLD_SOURCE_URL,
    RANK_AA,
    RANK_AA_CODING_AGENTS,
    RANK_AGGREGATE,
    RANK_BENCHMARK_SITE,
    RANK_CURATED,
    RANK_HAND_ENTERED,
    RANKED_PREFIXES,
    SWE_ATLAS_KEY_URLS,
    SWE_MARATHON_SOURCE_URL,
    TOOLATHLON_SOURCE_URL,
    may_overwrite,
    source_rank,
)

# apply_score() reads the benchmark grid off the document; a key llm.json does
# not describe rounds to the default one printed digit.
DOC: dict = {"benchmarks": {}}

AA_PAGE = "https://artificialanalysis.ai/models/gemma-4-31b"
EVALS_IFBENCH = EVALS_REPORT_KEY_URLS["ifbench"]
HF_CARD = f"{HUGGING_FACE_PREFIX}/google/gemma-4-31b"
HAND_ENTERED = "https://thenextweb.com/news/some-benchmark-writeup"


def model_with(score=None, source=None, key="ifbench") -> dict:
    return {
        "name": "m",
        "scores": {key: score},
        "scores_updated": {key: None},
        "scores_source": {key: source},
    }


class TestRungs(unittest.TestCase):
    def test_artificial_analysis_leads(self) -> None:
        self.assertEqual(source_rank(AA_PAGE), RANK_AA)
        self.assertEqual(RANK_AA, min(rank for _, rank in RANKED_PREFIXES))

    def test_benchmarks_own_leaderboards(self) -> None:
        for url in (
            OSWORLD_SOURCE_URL,
            TOOLATHLON_SOURCE_URL,
            MCP_ATLAS_SOURCE_URL,
            BFCL_SOURCE_URL,
            DATACURVE_SOURCE_URL,
            FRONTIERSWE_SOURCE_URL,
            FRONTIERCODE_SOURCE_URL,
            SWE_MARATHON_SOURCE_URL,
            *SWE_ATLAS_KEY_URLS.values(),
        ):
            with self.subTest(url=url):
                self.assertEqual(source_rank(url), RANK_BENCHMARK_SITE)

    def test_curated_third_parties(self) -> None:
        for url in (*EVALS_REPORT_KEY_URLS.values(), DEEPSWE_SOURCE_URL):
            with self.subTest(url=url):
                self.assertEqual(source_rank(url), RANK_CURATED)

    def test_aggregates(self) -> None:
        self.assertEqual(source_rank(LLMSTATS_SOURCE_URL), RANK_AGGREGATE)
        self.assertEqual(source_rank(HF_CARD), RANK_AGGREGATE)

    def test_unattributed_and_unknown_rank_as_hand_entered(self) -> None:
        self.assertEqual(source_rank(None), RANK_HAND_ENTERED)
        self.assertEqual(source_rank(""), RANK_HAND_ENTERED)
        self.assertEqual(source_rank(HAND_ENTERED), RANK_HAND_ENTERED)

    def test_ranked_pages_are_stored_form(self) -> None:
        # Ranks are matched against canonicalized URLs, so a prefix carrying a
        # query or a trailing slash could never match what is stored.
        for url, _ in RANKED_PREFIXES:
            with self.subTest(url=url):
                self.assertNotIn("?", url)
                self.assertFalse(url.endswith("/"))


class TestSameHostFamilies(unittest.TestCase):
    """AA publishes on one host at two standings, and the Coding Agent Index
    must not inherit the lock the model pages carry."""

    def test_coding_agent_index_is_not_aa_rank(self) -> None:
        self.assertEqual(source_rank(AA_CODING_AGENTS_SOURCE_URL), RANK_AA_CODING_AGENTS)
        self.assertEqual(source_rank(AA_PAGE), RANK_AA)

    def test_coding_agent_index_outranks_the_aggregates(self) -> None:
        self.assertTrue(may_overwrite(AA_CODING_AGENTS_SOURCE_URL, HF_CARD))
        self.assertFalse(may_overwrite(HF_CARD, AA_CODING_AGENTS_SOURCE_URL))

    def test_prefix_matches_only_on_a_path_boundary(self) -> None:
        # A future leaderboard at .../models-v2 is a different page, not a
        # longer spelling of the one that holds rank 1.
        self.assertEqual(
            source_rank("https://artificialanalysis.ai/models-v2/x"),
            RANK_HAND_ENTERED,
        )


class TestMayOverwrite(unittest.TestCase):
    def test_better_rank_wins(self) -> None:
        self.assertTrue(may_overwrite(AA_PAGE, EVALS_IFBENCH))
        self.assertTrue(may_overwrite(EVALS_IFBENCH, HF_CARD))
        self.assertTrue(may_overwrite(HF_CARD, HAND_ENTERED))

    def test_worse_rank_refused(self) -> None:
        self.assertFalse(may_overwrite(EVALS_IFBENCH, AA_PAGE))
        self.assertFalse(may_overwrite(HF_CARD, EVALS_IFBENCH))
        self.assertFalse(may_overwrite(HAND_ENTERED, HF_CARD))

    def test_equal_rank_passes_so_a_source_can_refresh_itself(self) -> None:
        self.assertTrue(may_overwrite(AA_PAGE, AA_PAGE))
        self.assertTrue(
            may_overwrite(TOOLATHLON_SOURCE_URL, SWE_ATLAS_KEY_URLS["swe_atlas_qna"])
        )


class TestApplyScoreHonoursRank(unittest.TestCase):
    def test_outranked_source_cannot_change_a_stored_value(self) -> None:
        model = model_with(score=39.6, source=AA_PAGE)
        changes: list = []
        n = update.apply_score(DOC, model, "m", "ifbench", 38.0, EVALS_IFBENCH, changes)
        self.assertEqual(n, 0)
        self.assertEqual(model["scores"]["ifbench"], 39.6)
        self.assertEqual(model["scores_source"]["ifbench"], AA_PAGE)
        self.assertEqual(changes, [])

    def test_outranked_source_does_not_restamp_the_date(self) -> None:
        model = model_with(score=39.6, source=AA_PAGE)
        model["scores_updated"]["ifbench"] = "2026-08-27"
        update.apply_score(DOC, model, "m", "ifbench", 38.0, EVALS_IFBENCH, [])
        self.assertEqual(model["scores_updated"]["ifbench"], "2026-08-27")

    def test_better_ranked_source_overwrites(self) -> None:
        model = model_with(score=38.0, source=EVALS_IFBENCH)
        changes: list = []
        n = update.apply_score(DOC, model, "m", "ifbench", 39.6, AA_PAGE, changes)
        self.assertEqual(n, 1)
        self.assertEqual(model["scores"]["ifbench"], 39.6)
        self.assertEqual(model["scores_source"]["ifbench"], AA_PAGE)
        self.assertEqual(changes, [("m", "ifbench", 38.0, 39.6)])

    def test_same_source_refreshes_its_own_value(self) -> None:
        model = model_with(score=1765, source=AA_PAGE, key="gdpval_aa")
        n = update.apply_score(DOC, model, "m", "gdpval_aa", 1725, AA_PAGE, [])
        self.assertEqual(n, 1)
        self.assertEqual(model["scores"]["gdpval_aa"], 1725)

    def test_rank_never_blocks_filling_a_null(self) -> None:
        model = model_with(score=None, source=None)
        n = update.apply_score(DOC, model, "m", "ifbench", 38.0, EVALS_IFBENCH, [])
        self.assertEqual(n, 1)
        self.assertEqual(model["scores"]["ifbench"], 38.0)

    def test_any_scraper_may_overwrite_a_hand_entry(self) -> None:
        for url in (AA_PAGE, TOOLATHLON_SOURCE_URL, EVALS_IFBENCH, HF_CARD):
            with self.subTest(url=url):
                model = model_with(score=10.0, source=HAND_ENTERED)
                n = update.apply_score(DOC, model, "m", "ifbench", 38.0, url, [])
                self.assertEqual(n, 1)
                self.assertEqual(model["scores_source"]["ifbench"], url)

    def test_a_cleared_attribution_is_overwritable_too(self) -> None:
        # edit.py stamps None when a value is typed in by hand.
        model = model_with(score=10.0, source=None)
        n = update.apply_score(DOC, model, "m", "ifbench", 38.0, HF_CARD, [])
        self.assertEqual(n, 1)


class TestOrderIndependence(unittest.TestCase):
    """The regression this exists for: 2ab9ab0 ran the AA ingest alone and
    displaced evals.report's ifbench and mmlu_pro values, and 0c24cc9 -- the
    next full refresh -- put them back. Neither the order the ingests run in
    nor which of them run may change where a score settles."""

    def apply_both(self, urls) -> dict:
        model = model_with()
        values = {AA_PAGE: 39.6, EVALS_IFBENCH: 38.0}
        for url in urls:
            update.apply_score(DOC, model, "m", "ifbench", values[url], url, [])
        return model

    def test_either_order_settles_on_the_same_value(self) -> None:
        aa_last = self.apply_both([EVALS_IFBENCH, AA_PAGE])
        aa_first = self.apply_both([AA_PAGE, EVALS_IFBENCH])
        self.assertEqual(aa_first["scores"], aa_last["scores"])
        self.assertEqual(aa_first["scores_source"], aa_last["scores_source"])
        self.assertEqual(aa_first["scores"]["ifbench"], 39.6)

    def test_a_partial_run_leaves_what_a_full_run_would(self) -> None:
        full = self.apply_both([AA_PAGE, EVALS_IFBENCH])
        aa_only = self.apply_both([AA_PAGE])
        self.assertEqual(aa_only["scores"], full["scores"])
        self.assertEqual(aa_only["scores_source"], full["scores_source"])

    def test_a_second_pass_changes_nothing(self) -> None:
        model = self.apply_both([AA_PAGE, EVALS_IFBENCH])
        before = (dict(model["scores"]), dict(model["scores_source"]))
        changes: list = []
        for url, value in ((AA_PAGE, 39.6), (EVALS_IFBENCH, 38.0)):
            update.apply_score(DOC, model, "m", "ifbench", value, url, changes)
        self.assertEqual(changes, [])
        self.assertEqual((model["scores"], model["scores_source"]), before)


class TestIngestsGoThroughTheGate(unittest.TestCase):
    """Rank lives in apply_score, the one writer every ingest shares, so no
    ingest can route around it. Exercised through the evals.report ingest,
    which is the one the ranking actually took a column away from."""

    @staticmethod
    def doc_with(score, source) -> dict:
        return {
            "benchmarks": {},
            "models": [
                {
                    "name": "deepseek-r1",
                    "scores": {"ifbench": score},
                    "scores_updated": {"ifbench": "2026-08-27" if score else None},
                    "scores_source": {"ifbench": source},
                }
            ],
        }

    def test_evals_report_leaves_an_aa_value_alone(self) -> None:
        doc = self.doc_with(39.6, AA_PAGE)
        matched, updated, changes = update.update_evals_report_scores(
            doc, {"deepseek-r1": {"ifbench": 38.0}}
        )
        self.assertEqual((matched, updated, changes), (1, 0, []))
        model = doc["models"][0]
        self.assertEqual(model["scores"]["ifbench"], 39.6)
        self.assertEqual(model["scores_source"]["ifbench"], AA_PAGE)

    def test_evals_report_still_fills_a_column_aa_has_not_measured(self) -> None:
        doc = self.doc_with(None, None)
        _, updated, changes = update.update_evals_report_scores(
            doc, {"deepseek-r1": {"ifbench": 38.0}}
        )
        self.assertEqual(updated, 1)
        self.assertEqual(changes, [("deepseek-r1", "ifbench", None, 38)])
        model = doc["models"][0]
        self.assertEqual(model["scores"]["ifbench"], 38)
        self.assertEqual(model["scores_source"]["ifbench"], EVALS_IFBENCH)


class TestEveryScrapedPageIsRanked(unittest.TestCase):
    """A source whose page is not in the table ranks as hand-entered, which
    would quietly let the aggregates overwrite it. Every URL update.py stamps
    has to resolve to a real rung."""

    def test_no_ingest_falls_through_to_hand_entered(self) -> None:
        stamped = [
            update.aa_model_page_url("gemma-4-31b"),
            AA_CODING_AGENTS_SOURCE_URL,
            OSWORLD_SOURCE_URL,
            LLMSTATS_SOURCE_URL,
            TOOLATHLON_SOURCE_URL,
            MCP_ATLAS_SOURCE_URL,
            BFCL_SOURCE_URL,
            DEEPSWE_SOURCE_URL,
            DATACURVE_SOURCE_URL,
            FRONTIERSWE_SOURCE_URL,
            FRONTIERCODE_SOURCE_URL,
            SWE_MARATHON_SOURCE_URL,
            HF_CARD,
            *SWE_ATLAS_KEY_URLS.values(),
            *EVALS_REPORT_KEY_URLS.values(),
        ]
        for url in stamped:
            with self.subTest(url=url):
                self.assertLess(source_rank(url), RANK_HAND_ENTERED)

    def test_rungs_are_contiguous_and_ordered(self) -> None:
        self.assertEqual(
            [
                RANK_AA,
                RANK_BENCHMARK_SITE,
                RANK_CURATED,
                RANK_AA_CODING_AGENTS,
                RANK_AGGREGATE,
                RANK_HAND_ENTERED,
            ],
            [1, 2, 3, 4, 5, 6],
        )
        self.assertEqual(
            sorted({rank for _, rank in RANKED_PREFIXES}),
            [RANK_AA, RANK_BENCHMARK_SITE, RANK_CURATED, RANK_AA_CODING_AGENTS,
             RANK_AGGREGATE],
        )

    def test_prefixes_are_tried_longest_first(self) -> None:
        lengths = [len(url) for url, _ in precedence.RANKED_PREFIXES]
        self.assertEqual(lengths, sorted(lengths, reverse=True))


if __name__ == "__main__":
    unittest.main()
