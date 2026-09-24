#!/usr/bin/env python3
"""Tests for source precedence. Run with ./test_precedence.py

Most columns in llm.json have more than one publisher, so every refresh has to
decide whose number lands. That used to be settled by the order update.py calls
its ingests -- last writer wins -- which made the file depend on which ingests
ran: the AA-only pass in 2ab9ab0 replaced four evals.report values with
Artificial Analysis' own, and the next full refresh put evals.report back.

_precedence.py declares the rank instead and apply_score() enforces it. These
tests pin the rungs, the shared authority of AA's publication surfaces, and
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
    VALS_KEY_URLS,
    VALS_OWN_KEY_URLS,
    VALS_RERUN_KEY_URLS,
    may_overwrite,
    source_rank,
)

# apply_score() reads the benchmark grid and range off the document; a key this
# does not describe rounds to the default one printed digit and is held to a
# percentage. GDPval-AA is an Elo, so it declares llm.json's range for it.
DOC: dict = {"benchmarks": {"gdpval_aa": {"range": [-1000, 3000]}}}

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
        for url in (
            *EVALS_REPORT_KEY_URLS.values(),
            *VALS_RERUN_KEY_URLS.values(),
            DEEPSWE_SOURCE_URL,
        ):
            with self.subTest(url=url):
                self.assertEqual(source_rank(url), RANK_CURATED)

    def test_vals_own_boards_rank_as_first_party(self) -> None:
        # Vals is curated for the boards it re-runs and first-party for the ones
        # it authored, so the split is per board rather than per source. Both
        # halves are checked here, and the split itself in the two tests below.
        self.assertTrue(VALS_OWN_KEY_URLS)
        for url in VALS_OWN_KEY_URLS.values():
            with self.subTest(url=url):
                self.assertEqual(source_rank(url), RANK_BENCHMARK_SITE)

    def test_every_vals_board_is_on_exactly_one_side_of_the_split(self) -> None:
        self.assertEqual(
            set(VALS_OWN_KEY_URLS) | set(VALS_RERUN_KEY_URLS), set(VALS_KEY_URLS)
        )
        self.assertEqual(set(VALS_OWN_KEY_URLS) & set(VALS_RERUN_KEY_URLS), set())

    def test_a_vals_own_board_outranks_a_vals_rerun(self) -> None:
        # Not a statement about the source: it is what lets the Vibe Code Bench
        # page lead its column the way any other benchmark's own board does.
        own = next(iter(VALS_OWN_KEY_URLS.values()))
        rerun = next(iter(VALS_RERUN_KEY_URLS.values()))
        self.assertTrue(may_overwrite(own, rerun))
        self.assertFalse(may_overwrite(rerun, own))

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
    """Every AA publication has the same authority."""

    def test_coding_agent_index_is_aa_rank(self) -> None:
        self.assertEqual(source_rank(AA_CODING_AGENTS_SOURCE_URL), RANK_AA_CODING_AGENTS)
        self.assertEqual(RANK_AA_CODING_AGENTS, RANK_AA)
        self.assertEqual(source_rank(AA_PAGE), RANK_AA)

    def test_coding_agent_index_outranks_the_aggregates(self) -> None:
        self.assertTrue(may_overwrite(AA_CODING_AGENTS_SOURCE_URL, HF_CARD))
        self.assertFalse(may_overwrite(HF_CARD, AA_CODING_AGENTS_SOURCE_URL))

    def test_prefix_matches_only_on_a_path_boundary(self) -> None:
        self.assertEqual(source_rank("https://artificialanalysis.ai/models-v2/x"), RANK_AA)
        self.assertEqual(source_rank("https://x.com/ArtificialAnlys/status/123"), RANK_AA)
        self.assertEqual(
            source_rank("https://artificialanalysis.ai.example.com/models/x"),
            RANK_HAND_ENTERED,
        )
        self.assertEqual(source_rank("https://x.com/ArtificialAnlysFake/status/123"), RANK_HAND_ENTERED)


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
    def test_equal_aa_value_takes_authority_without_changing_date(self) -> None:
        model = model_with(score=39.6, source=EVALS_IFBENCH)
        model["scores_updated"]["ifbench"] = "2026-08-27"
        self.assertEqual(update.apply_score(DOC, model, "m", "ifbench", 39.6, AA_PAGE, []), 1)
        self.assertEqual(model["scores_source"]["ifbench"], AA_PAGE)
        self.assertEqual(model["scores_updated"]["ifbench"], "2026-08-27")
        self.assertEqual(update.apply_score(DOC, model, "m", "ifbench", 38, EVALS_IFBENCH, []), 0)
        self.assertEqual(update.apply_score(DOC, model, "m", "ifbench", 39.6, AA_PAGE, []), 0)

    def test_aa_agent_ingest_overwrites_other_sources_and_itself(self) -> None:
        for source in (AA_PAGE, AA_CODING_AGENTS_SOURCE_URL, HF_CARD,
                       SWE_ATLAS_KEY_URLS["swe_atlas_qna"], HAND_ENTERED):
            with self.subTest(source=source):
                model = model_with(55.6, source, "swe_atlas_qna")
                doc = {"benchmarks": {}, "models": [model]}
                _, count, _ = update.update_aa_coding_agents_scores(doc, {"m": {"swe_atlas_qna": 51.34}})
                self.assertEqual(count, 1)
                self.assertEqual(model["scores"]["swe_atlas_qna"], 51.3)
                self.assertEqual(model["scores_source"]["swe_atlas_qna"], AA_CODING_AGENTS_SOURCE_URL)
                self.assertEqual(update.apply_score(doc, model, "m", "swe_atlas_qna", 60, HF_CARD, []), 0)

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


class TestCustomSources(unittest.TestCase):
    """A page no fetcher writes is a custom source, and every fetcher outranks it.

    The fetchers' URLs are all known (RANKED_PREFIXES is built from their
    constants), so a stored page outside them was typed in by a person. The
    fill-only aggregates, which never replace a stored value, still replace
    that one, and a fetcher confirming its number takes over the credit.
    """

    CUSTOM = (HAND_ENTERED, "https://www.anthropic.com/news/claude-sonnet-5", None)

    def test_every_fetcher_page_is_a_fetcher_source(self) -> None:
        for prefix, _ in RANKED_PREFIXES:
            with self.subTest(prefix=prefix):
                self.assertTrue(precedence.is_fetcher_source(prefix))
        self.assertTrue(precedence.is_fetcher_source(HF_CARD))

    def test_anything_else_is_custom(self) -> None:
        for url in (*self.CUSTOM, "https://llm-stats.com/models/glm-5.3-flash"):
            with self.subTest(url=url):
                self.assertFalse(precedence.is_fetcher_source(url))

    def test_a_fill_only_fetcher_replaces_a_custom_value(self) -> None:
        for stored in self.CUSTOM:
            for url in (HF_CARD, LLMSTATS_SOURCE_URL):
                with self.subTest(stored=stored, url=url):
                    model = model_with(score=10.0, source=stored)
                    n = update.apply_score(DOC, model, "m", "ifbench", 38.0, url, [], fill_only=True)
                    self.assertEqual(n, 1)
                    self.assertEqual(model["scores"]["ifbench"], 38.0)
                    self.assertEqual(model["scores_source"]["ifbench"], url)

    def test_a_fill_only_fetcher_still_leaves_another_fetchers_value(self) -> None:
        for stored in (AA_PAGE, EVALS_IFBENCH, HF_CARD):
            with self.subTest(stored=stored):
                model = model_with(score=10.0, source=stored)
                n = update.apply_score(
                    DOC, model, "m", "ifbench", 38.0, LLMSTATS_SOURCE_URL, [], fill_only=True
                )
                self.assertEqual(n, 0)
                self.assertEqual(model["scores_source"]["ifbench"], stored)

    def test_a_fetcher_confirming_a_custom_number_takes_its_credit(self) -> None:
        for fill_only in (False, True):
            with self.subTest(fill_only=fill_only):
                model = model_with(score=38.0, source=HAND_ENTERED)
                model["scores_updated"]["ifbench"] = "2026-08-01"
                changes: list = []
                n = update.apply_score(
                    DOC, model, "m", "ifbench", 38.0, HF_CARD, changes, fill_only=fill_only
                )
                self.assertEqual(n, 1)
                self.assertEqual(model["scores_source"]["ifbench"], HF_CARD)
                # The number did not move, so neither does its date.
                self.assertEqual(model["scores_updated"]["ifbench"], "2026-08-01")

    def test_an_equal_number_from_a_peer_fetcher_moves_nothing(self) -> None:
        model = model_with(score=38.0, source=HF_CARD)
        n = update.apply_score(DOC, model, "m", "ifbench", 38.0, LLMSTATS_SOURCE_URL, [])
        self.assertEqual(n, 0)
        self.assertEqual(model["scores_source"]["ifbench"], HF_CARD)

    def test_a_fill_only_fetcher_refreshes_its_own_value(self) -> None:
        """Same page, new number: the source updating itself, not an overwrite."""
        for url in (HF_CARD, LLMSTATS_SOURCE_URL):
            with self.subTest(url=url):
                model = model_with(score=10.0, source=url)
                changes: list = []
                n = update.apply_score(DOC, model, "m", "ifbench", 38.0, url, changes, fill_only=True)
                self.assertEqual(n, 1)
                self.assertEqual(model["scores"]["ifbench"], 38.0)
                self.assertEqual(model["scores_source"]["ifbench"], url)
                self.assertEqual(changes, [("m", "ifbench", 10.0, 38.0)])

    def test_own_page_is_matched_on_its_canonical_form(self) -> None:
        model = model_with(score=10.0, source=HF_CARD)
        n = update.apply_score(
            DOC, model, "m", "ifbench", 38.0, f"{HF_CARD}/?tab=card", [], fill_only=True
        )
        self.assertEqual(n, 1)

    def test_another_card_of_the_same_fetcher_is_not_its_own(self) -> None:
        model = model_with(score=10.0, source=HF_CARD)
        n = update.apply_score(
            DOC, model, "m", "ifbench", 38.0, f"{HUGGING_FACE_PREFIX}/google/gemma-4-31b-it",
            [], fill_only=True,
        )
        self.assertEqual(n, 0)
        self.assertEqual(model["scores"]["ifbench"], 10.0)

    def test_null_never_replaces_its_own_value(self) -> None:
        model = model_with(score=10.0, source=HF_CARD)
        n = update.apply_score(DOC, model, "m", "ifbench", None, HF_CARD, [], fill_only=True)
        self.assertEqual(n, 0)
        self.assertEqual(model["scores"]["ifbench"], 10.0)

    def test_null_never_replaces_a_custom_value(self) -> None:
        model = model_with(score=10.0, source=HAND_ENTERED)
        n = update.apply_score(DOC, model, "m", "ifbench", None, HF_CARD, [], fill_only=True)
        self.assertEqual(n, 0)
        self.assertEqual(model["scores"]["ifbench"], 10.0)


class TestDroppedScores(unittest.TestCase):
    """Issue #226: a score its own page no longer reports gives way, in that run.

    Never nulled -- the replacement is another fetcher's current number, and
    only where the stored page is a fetcher's and was read in this very run.
    """

    def setUp(self) -> None:
        update.RUN_REPORTS = update.RunReports()

    def run_once(self, model: dict, writes) -> list:
        """Each write is (value, url, fill_only); then the end-of-run pass."""
        changes: list = []
        for value, url, fill_only in writes:
            update.apply_score(DOC, model, "m", "ifbench", value, url, changes, fill_only=fill_only)
        update.replace_dropped_scores(DOC, changes)
        return changes

    def other_row(self, url: str) -> None:
        # The page was read this run: it reported some other model's score.
        other = model_with(score=None)
        other["name"] = "other"
        update.apply_score(DOC, other, "other", "ifbench", 1.0, url, [])

    def test_a_dropped_score_is_replaced_by_a_lower_ranked_fetcher(self) -> None:
        model = model_with(score=39.6, source=AA_PAGE)
        self.other_row(AA_PAGE)
        changes = self.run_once(model, [(38.0, EVALS_IFBENCH, False)])
        self.assertEqual(model["scores"]["ifbench"], 38.0)
        self.assertEqual(model["scores_source"]["ifbench"], EVALS_IFBENCH)
        self.assertEqual(changes, [("m", "ifbench", 39.6, 38.0)])

    def test_order_does_not_matter(self) -> None:
        """The replacing fetcher may run before the page that dropped the score is read."""
        model = model_with(score=39.6, source=AA_PAGE)
        update.apply_score(DOC, model, "m", "ifbench", 38.0, EVALS_IFBENCH, [])
        self.other_row(AA_PAGE)
        update.replace_dropped_scores(DOC, [])
        self.assertEqual(model["scores"]["ifbench"], 38.0)

    def test_a_fill_only_fetcher_may_replace_a_dropped_score(self) -> None:
        model = model_with(score=39.6, source=EVALS_IFBENCH)
        self.other_row(EVALS_IFBENCH)
        self.run_once(model, [(38.0, LLMSTATS_SOURCE_URL, True)])
        self.assertEqual(model["scores_source"]["ifbench"], LLMSTATS_SOURCE_URL)

    def test_the_best_ranked_candidate_wins(self) -> None:
        model = model_with(score=39.6, source=AA_PAGE)
        self.other_row(AA_PAGE)
        self.run_once(model, [(30.0, LLMSTATS_SOURCE_URL, True), (38.0, EVALS_IFBENCH, False)])
        self.assertEqual(model["scores"]["ifbench"], 38.0)

    def test_a_score_its_page_still_reports_stays(self) -> None:
        model = model_with(score=39.6, source=AA_PAGE)
        self.run_once(model, [(39.6, AA_PAGE, False), (38.0, EVALS_IFBENCH, False)])
        self.assertEqual(model["scores_source"]["ifbench"], AA_PAGE)

    def test_a_null_row_is_not_a_report(self) -> None:
        model = model_with(score=39.6, source=AA_PAGE)
        self.run_once(model, [(None, AA_PAGE, False), (38.0, EVALS_IFBENCH, False)])
        self.assertEqual(model["scores"]["ifbench"], 38.0)

    def test_a_page_not_read_this_run_drops_nothing(self) -> None:
        """A fetch that failed or was skipped is not a retraction."""
        model = model_with(score=39.6, source=AA_PAGE)
        self.run_once(model, [(38.0, EVALS_IFBENCH, False)])
        self.assertEqual(model["scores"]["ifbench"], 39.6)
        self.assertEqual(model["scores_source"]["ifbench"], AA_PAGE)

    def test_it_is_never_nulled(self) -> None:
        model = model_with(score=39.6, source=AA_PAGE)
        self.other_row(AA_PAGE)
        self.run_once(model, [])
        self.assertEqual(model["scores"]["ifbench"], 39.6)
        self.assertEqual(model["scores_source"]["ifbench"], AA_PAGE)

    def test_only_a_fetchers_page_qualifies(self) -> None:
        # Nothing refuses a write over a custom page, so nothing waits on it
        # either; a custom page "read" this run is not a fetcher's retraction.
        self.assertFalse(update.RunReports().dropped("m", "ifbench", HAND_ENTERED))
        self.assertFalse(update.RunReports().dropped("m", "ifbench", None))

    def test_a_step_that_failed_partway_drops_nothing(self) -> None:
        """Its page was read, but only as far as the step got before raising."""
        model = model_with(score=39.6, source=AA_PAGE)

        def ingest() -> None:
            self.other_row(AA_PAGE)
            raise RuntimeError("bug halfway through the rows")

        failures: list = []
        update.source_update(failures, None, "aa", ingest)
        self.assertEqual(len(failures), 1)
        self.run_once(model, [(38.0, EVALS_IFBENCH, False)])
        self.assertEqual(model["scores"]["ifbench"], 39.6)
        self.assertEqual(model["scores_source"]["ifbench"], AA_PAGE)

    def test_a_page_read_in_part_drops_nothing(self) -> None:
        model = model_with(score=39.6, source=AA_PAGE)
        self.other_row(AA_PAGE)
        update.RUN_REPORTS.failed(AA_PAGE)
        self.run_once(model, [(38.0, EVALS_IFBENCH, False)])
        self.assertEqual(model["scores"]["ifbench"], 39.6)

    def test_a_step_that_succeeded_still_drops(self) -> None:
        model = model_with(score=39.6, source=AA_PAGE)
        update.source_update([], None, "aa", lambda: self.other_row(AA_PAGE))
        self.run_once(model, [(38.0, EVALS_IFBENCH, False)])
        self.assertEqual(model["scores"]["ifbench"], 38.0)

    def test_a_refusal_from_an_earlier_run_is_not_applied(self) -> None:
        model = model_with(score=39.6, source=AA_PAGE)
        update.apply_score(DOC, model, "m", "ifbench", 38.0, EVALS_IFBENCH, [])
        update.RUN_REPORTS = update.RunReports()  # the next run
        self.other_row(AA_PAGE)
        update.replace_dropped_scores(DOC, [])
        self.assertEqual(model["scores"]["ifbench"], 39.6)

    def test_the_url_backfill_records_nothing(self) -> None:
        model = model_with(score=39.6, source=None)
        update.apply_score(DOC, model, "m", "ifbench", 39.6, AA_PAGE, [], fill_urls_only=True)
        self.assertEqual(update.RUN_REPORTS.read_pages, set())


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

    def test_rungs_keep_aa_agent_alias_at_the_top(self) -> None:
        self.assertEqual(
            [
                RANK_AA,
                RANK_BENCHMARK_SITE,
                RANK_CURATED,
                RANK_AA_CODING_AGENTS,
                RANK_AGGREGATE,
                RANK_HAND_ENTERED,
            ],
            [1, 2, 3, 1, 5, 6],
        )
        self.assertEqual(
            sorted({rank for _, rank in RANKED_PREFIXES}),
            [RANK_AA, RANK_BENCHMARK_SITE, RANK_CURATED,
             RANK_AGGREGATE],
        )

    def test_prefixes_are_tried_longest_first(self) -> None:
        lengths = [len(url) for url, _ in precedence.RANKED_PREFIXES]
        self.assertEqual(lengths, sorted(lengths, reverse=True))


if __name__ == "__main__":
    unittest.main()
