#!/usr/bin/env python3
"""Tests for the Artificial Analysis V2 API client. Run with ./test_artificialanalysis_api.py

The legacy /api/v2/data/llms/models route retires on 2026-11-04. Its documented
replacements page their answers, split into a free and a Pro tier, and spell
several fields differently. These pin the things that migration turns on:
every page is read, a key without Pro access still gets an answer, and the
record that comes back is translated into the names the table and update.py
have always read.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import artificialanalysis as aa


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def page(models, *, has_more=False, page_number=1, tier="pro"):
    return {
        "tier": tier,
        "intelligence_index_version": 4.1,
        "pagination": {
            "page": page_number,
            "page_size": 200,
            "total_pages": 2 if has_more else page_number,
            "has_more": has_more,
        },
        "data": models,
    }


class FetchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # The response cache is a real file under ~/.cache; every test gets its
        # own so none of them reads the developer's, or each other's.
        cache = Path(tempfile.mkdtemp()) / "response.json"
        patcher = mock.patch.object(aa, "RESPONSE_CACHE_PATH", str(cache))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cache_path = cache

    def get(self, responses):
        """A requests.get double answering the queued responses in order."""
        self.calls = []
        queue = list(responses)

        def _get(url, headers=None, params=None, timeout=None):
            self.calls.append((url, dict(params or {}), dict(headers or {})))
            return queue.pop(0)

        return _get


class TestEndpoints(FetchTestCase):
    def test_no_legacy_data_route_is_left_behind(self) -> None:
        self.assertNotIn("/api/v2/data/", aa.MODELS_URL)
        self.assertNotIn("/api/v2/data/", aa.MODELS_FREE_URL)

    def test_the_pro_route_is_the_one_read_by_default(self) -> None:
        get = self.get([FakeResponse(payload=page([{"slug": "a"}]))])
        with mock.patch.object(aa.requests, "get", get):
            payload, error = aa._fetch_models("key")
        self.assertEqual(error, "")
        self.assertEqual(self.calls[0][0], "https://artificialanalysis.ai/api/v2/language/models")
        self.assertEqual(self.calls[0][2]["x-api-key"], "key")
        self.assertEqual([m["slug"] for m in payload["data"]], ["a"])

    def test_every_page_is_read_and_merged_into_one_payload(self) -> None:
        get = self.get(
            [
                FakeResponse(payload=page([{"slug": "a"}], has_more=True, page_number=1)),
                FakeResponse(payload=page([{"slug": "b"}], page_number=2)),
            ]
        )
        with mock.patch.object(aa.requests, "get", get):
            payload, error = aa._fetch_models("key")
        self.assertEqual(error, "")
        self.assertEqual([m["slug"] for m in payload["data"]], ["a", "b"])
        self.assertEqual([call[1]["page"] for call in self.calls], [1, 2])
        # The envelope survives; the per-page block would only mislead about a
        # list that is now every page at once.
        self.assertEqual(payload["tier"], "pro")
        self.assertNotIn("pagination", payload)

    def test_a_runaway_has_more_stops_at_the_page_cap(self) -> None:
        responses = [FakeResponse(payload=page([{"slug": "a"}], has_more=True))] * (aa.MAX_PAGES + 5)
        with mock.patch.object(aa.requests, "get", self.get(responses)):
            payload, _error = aa._fetch_models("key")
        self.assertEqual(len(self.calls), aa.MAX_PAGES)
        self.assertEqual(len(payload["data"]), aa.MAX_PAGES)

    def test_a_key_without_pro_access_falls_back_to_the_free_route(self) -> None:
        get = self.get(
            [
                FakeResponse(403, {"error": "Subscription does not cover this endpoint"}),
                FakeResponse(payload=page([{"slug": "a"}], tier="free")),
            ]
        )
        with mock.patch.object(aa.requests, "get", get):
            payload, error = aa._fetch_models("key")
        self.assertEqual(error, "")
        self.assertEqual(self.calls[1][0], "https://artificialanalysis.ai/api/v2/language/models/free")
        self.assertEqual(payload["tier"], "free")

    def test_a_pinned_tier_is_read_and_nothing_else_is(self) -> None:
        get = self.get([FakeResponse(payload=page([{"slug": "a"}], tier="free"))])
        with mock.patch.object(aa.requests, "get", get):
            payload, error = aa._fetch_models("key", tier="free")
        self.assertEqual(error, "")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][0], aa.MODELS_FREE_URL)
        self.assertEqual(payload["tier"], "free")

    def test_a_pinned_pro_tier_does_not_quietly_downgrade(self) -> None:
        get = self.get([FakeResponse(403, {"error": "nope"})])
        with mock.patch.object(aa.requests, "get", get):
            payload, error = aa._fetch_models("key", tier="pro")
        self.assertIsNone(payload)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("nope", error)

    def test_a_failure_that_is_not_a_403_is_reported_as_it_stands(self) -> None:
        # A rate-limited key must not spend its next request on the free route:
        # the same limit applies there and the answer would be worse anyway.
        get = self.get([FakeResponse(429, {"error": "Rate limit exceeded"}, {"Retry-After": "60"})])
        with mock.patch.object(aa.requests, "get", get):
            payload, error = aa._fetch_models("key")
        self.assertIsNone(payload)
        self.assertEqual(len(self.calls), 1)
        self.assertIn("Rate limit exceeded", error)
        self.assertIn("60", error)

    def test_a_dropped_connection_is_retried_once_before_the_walk_is_lost(self) -> None:
        # A page that dies at the transport layer was not billed, and losing it
        # throws away the pages already paid for.
        calls = []

        def _get(url, headers=None, params=None, timeout=None):
            calls.append(params["page"])
            if len(calls) == 2:
                raise aa.requests.ConnectionError("Connection reset by peer")
            return FakeResponse(payload=page([{"slug": f"m{len(calls)}"}], has_more=len(calls) == 1))

        with mock.patch.object(aa.requests, "get", _get), mock.patch.object(aa.time, "sleep", lambda _s: None):
            payload, error = aa._fetch_models("key", tier="free")
        self.assertEqual(error, "")
        self.assertEqual(calls, [1, 2, 2])
        self.assertEqual([m["slug"] for m in payload["data"]], ["m1", "m3"])

    def test_a_connection_that_stays_down_fails_the_fetch(self) -> None:
        def _get(url, headers=None, params=None, timeout=None):
            raise aa.requests.ConnectionError("Connection reset by peer")

        with mock.patch.object(aa.requests, "get", _get), mock.patch.object(aa.time, "sleep", lambda _s: None):
            payload, error = aa._fetch_models("key", tier="free")
        self.assertIsNone(payload)
        self.assertIn("Connection reset by peer", error)

    def test_the_retired_route_reports_what_the_api_said(self) -> None:
        gone = {"error": "This endpoint was retired on 2026-11-04. See /data-api/migrate-v2-data"}
        with mock.patch.object(aa.requests, "get", self.get([FakeResponse(410, gone)])):
            payload, error = aa._fetch_models("key")
        self.assertIsNone(payload)
        self.assertIn("410", error)
        self.assertIn("migrate-v2-data", error)


class TestResponseCache(FetchTestCase):
    def test_a_second_fetch_in_the_window_costs_no_requests(self) -> None:
        # The three fetches in one update-all run are three processes, so the
        # cache is on disk: without it a free key spends 12 of its 100 daily
        # requests per run and the three-hourly cron runs out before the day does.
        with mock.patch.object(aa.requests, "get", self.get([FakeResponse(payload=page([{"slug": "a"}]))])):
            first, _ = aa._fetch_models("key")
            second, error = aa._fetch_models("key")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(error, "")
        self.assertEqual(second["data"], first["data"])

    def test_a_stale_response_is_re_read(self) -> None:
        responses = [FakeResponse(payload=page([{"slug": "a"}])), FakeResponse(payload=page([{"slug": "b"}]))]
        with mock.patch.object(aa.requests, "get", self.get(responses)):
            aa._fetch_models("key")
            with mock.patch.object(aa, "RESPONSE_CACHE_TTL", -1):
                payload, _error = aa._fetch_models("key")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual([m["slug"] for m in payload["data"]], ["b"])

    def test_no_cache_forces_a_fresh_read(self) -> None:
        responses = [FakeResponse(payload=page([{"slug": "a"}])), FakeResponse(payload=page([{"slug": "b"}]))]
        with mock.patch.object(aa.requests, "get", self.get(responses)):
            aa._fetch_models("key")
            payload, _error = aa._fetch_models("key", use_cache=False)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual([m["slug"] for m in payload["data"]], ["b"])

    def test_a_pinned_tier_does_not_serve_the_other_tiers_response(self) -> None:
        responses = [
            FakeResponse(payload=page([{"slug": "a"}], tier="free")),
            FakeResponse(payload=page([{"slug": "a"}], tier="pro")),
        ]
        with mock.patch.object(aa.requests, "get", self.get(responses)):
            aa._fetch_models("key", tier="free")
            payload, _error = aa._fetch_models("key", tier="pro")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(payload["tier"], "pro")

    def test_a_failed_fetch_is_not_cached(self) -> None:
        responses = [FakeResponse(500, {"error": "boom"}), FakeResponse(payload=page([{"slug": "a"}]))]
        with mock.patch.object(aa.requests, "get", self.get(responses)):
            failed, error = aa._fetch_models("key", tier="free")
            payload, _error = aa._fetch_models("key", tier="free")
        self.assertIsNone(failed)
        self.assertIn("boom", error)
        self.assertEqual([m["slug"] for m in payload["data"]], ["a"])

    def test_a_refused_pro_route_is_not_probed_again(self) -> None:
        # The 403 is billed against the same 100-a-day budget, so rediscovering
        # it every fetch would cost a free key a request each time.
        responses = [
            FakeResponse(403, {"error": "Language models list requires a Pro subscription"}),
            FakeResponse(payload=page([{"slug": "a"}], tier="free")),
            FakeResponse(payload=page([{"slug": "a"}], tier="free")),
        ]
        with mock.patch.object(aa.requests, "get", self.get(responses)):
            aa._fetch_models("key")
            with mock.patch.object(aa, "RESPONSE_CACHE_TTL", -1):
                payload, error = aa._fetch_models("key")
        self.assertEqual(error, "")
        self.assertEqual([call[0] for call in self.calls], [aa.MODELS_URL, aa.MODELS_FREE_URL, aa.MODELS_FREE_URL])
        self.assertEqual(payload["tier"], "free")

    def test_the_pro_route_is_probed_again_once_the_memo_ages_out(self) -> None:
        responses = [
            FakeResponse(403, {"error": "nope"}),
            FakeResponse(payload=page([{"slug": "a"}], tier="free")),
            FakeResponse(payload=page([{"slug": "a"}], tier="pro")),
        ]
        with mock.patch.object(aa.requests, "get", self.get(responses)):
            aa._fetch_models("key")
            aged = aa._read_response_cache()
            aged["pro_denied_at"] = time.time() - aa.PRO_RETRY_INTERVAL - 1
            aa._write_response_cache(aged)
            with mock.patch.object(aa, "RESPONSE_CACHE_TTL", -1):
                payload, _error = aa._fetch_models("key")
        self.assertEqual(self.calls[2][0], aa.MODELS_URL)
        self.assertEqual(payload["tier"], "pro")


PRO_MODEL = {
    "id": "36f73aaf",
    "name": "gpt-oss-20B (high)",
    "slug": "gpt-oss-20b",
    "release_date": "2025-08-05",
    "model_creator": {"id": "e67e56e3", "name": "OpenAI", "country": "us"},
    "evaluations": {
        "artificial_analysis_intelligence_index": 24.5,
        "artificial_analysis_agentic_index": 27.6,
        "artificial_analysis_openness_index": 38.9,
        "aa_lcr": 0.31,
        "tau2_telecom": 0.6,
        "tau_banking": 0.61,
        "gpqa_diamond": 0.69,
        "aa_omniscience_index": -63.92,
        "aa_omniscience_accuracy": 0.16,
        "aa_omniscience_non_hallucination_rate": 0.06,
        "gdpval_aa_elo": 681,
        "gdpval_aa_normalized": 0.09,
        "mmmu_pro": None,
    },
    "performance": {
        "percentile_05_output_tokens_per_second": 74.74,
        "median_output_tokens_per_second": 296.47,
        "percentile_95_output_tokens_per_second": 858.3,
        "percentile_05_time_to_first_token_seconds": 0.28,
        "percentile_95_time_to_first_token_seconds": 16.22,
    },
    "context_window_tokens": 131072,
    "parameters": {"total": 21, "active": 4},
    "licensing": {"is_open_weights": True},
    "huggingface_url": "https://huggingface.co/openai/gpt-oss-20b",
}


class TestNormalization(unittest.TestCase):
    def normalized(self):
        import copy

        return aa._normalize_model(copy.deepcopy(PRO_MODEL))

    def test_renamed_benchmarks_land_under_the_names_update_reads(self) -> None:
        # update.py's SCORE_MAPPINGS is keyed on the left-hand names, and the
        # model pages spell them that way too.
        evals = self.normalized()["evaluations"]
        self.assertEqual(evals["lcr"], 0.31)
        self.assertEqual(evals["tau2"], 0.6)
        self.assertEqual(evals["gpqa"], 0.69)
        self.assertEqual(evals["omniscience"], -63.92)
        self.assertEqual(evals["omniscience_accuracy"], 0.16)
        self.assertEqual(evals["gdpval"], 681)
        self.assertEqual(evals["gdpval_normalized"], 0.09)
        self.assertEqual(evals["agentic_index"], 27.6)
        self.assertEqual(evals["openness_index"], 38.9)
        for retired in ("aa_lcr", "tau2_telecom", "gpqa_diamond", "gdpval_aa_elo"):
            self.assertNotIn(retired, evals)

    def test_names_the_contract_kept_are_left_alone(self) -> None:
        evals = self.normalized()["evaluations"]
        self.assertEqual(evals["artificial_analysis_intelligence_index"], 24.5)
        self.assertEqual(evals["tau_banking"], 0.61)

    def test_the_non_hallucination_rate_is_left_for_enrichment_to_invert(self) -> None:
        # It is the one field stated the other way round, so it is turned over
        # only after the page pass -- see the enrichment test below.
        evals = self.normalized()["evaluations"]
        self.assertEqual(evals["aa_omniscience_non_hallucination_rate"], 0.06)
        self.assertNotIn("omniscience_hallucination_rate", evals)

    def test_the_speed_spread_lands_under_the_page_keys(self) -> None:
        model = self.normalized()
        self.assertEqual(model["output_speed_p05"], 74.74)
        self.assertEqual(model["output_speed_median"], 296.47)
        self.assertEqual(model["ttft_p95"], 16.22)

    def test_a_free_record_is_left_as_it_arrives(self) -> None:
        free = {"slug": "a", "evaluations": {"artificial_analysis_intelligence_index": 24.5}}
        self.assertEqual(
            aa._normalize_model(dict(free))["evaluations"],
            {"artificial_analysis_intelligence_index": 24.5},
        )


class TestFieldsThePagesUsedToCarry(unittest.TestCase):
    def setUp(self) -> None:
        # Any page fetch here would be a network call and a wrong answer: the
        # point of these is that the API record already answers.
        patcher = mock.patch.object(
            aa, "_fetch_page_metrics", side_effect=AssertionError("fetched a model page")
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_open_weights_comes_off_the_licensing_block(self) -> None:
        self.assertIs(aa._is_open_source(PRO_MODEL), True)
        self.assertIs(aa._is_open_source({"licensing": {"is_open_weights": False}}), False)

    def test_the_context_window_snaps_to_the_advertised_size(self) -> None:
        self.assertEqual(aa._extract_context_window(PRO_MODEL), "128k")

    def test_the_parameter_counts_render_as_the_sites_shorthand(self) -> None:
        self.assertEqual(aa._extract_params(PRO_MODEL), "21B-A4B")

    def test_the_hugging_face_url_is_canonicalized(self) -> None:
        self.assertEqual(
            aa._extract_hugging_face_url(dict(PRO_MODEL, huggingface_url="https://huggingface.co/openai/gpt-oss-20b/tree/main")),
            "https://huggingface.co/openai/gpt-oss-20b",
        )

    def test_a_metric_the_api_sent_is_read_without_a_page_fetch(self) -> None:
        model = aa._normalize_model(dict(PRO_MODEL, evaluations=dict(PRO_MODEL["evaluations"])))
        self.assertEqual(aa._extract_metric(model, "agentic_index"), 27.6)
        self.assertEqual(aa._extract_metric(model, "output_speed_p05"), 74.74)


# A model page's metrics block, in the shape and the field order the live
# payload uses. Trimmed to the neighbourhood the parser reads; the spellings are
# what matter.
MODEL_PAGE = (
    '{"currentModel":{"id":"b8fc61f7","slug":"claude-opus-5",'
    '"name":"Claude Opus 5 (Adaptive Reasoning, Max Effort)",'
    '"shortName":"Claude Opus 5 (max)","effort":{"slug":"max","label":"max"},'
    '"microevalsEnabled":true,"intelligenceIndex":50.7001865797629,'
    '"intelligenceIndexIsEstimated":false,'
    '"capabilities":{"financeAndAccounting":54.879570063518685,'
    '"legal":57.291863099419025,"engineering":54.014034523937966},'
    '"omniscience":37.5,"omniscienceAccuracy":0.6015,'
    '"omniscienceHallucinationRate":0.4512,'
    '"mlcrOverall":0.555555555555556,"harveyLab":0.93457808655377,'
    '"itBenchSre":null,"tau2":null,"tauBanking":0.612371134020619,'
    '"terminalbenchHard":null,'
    '"terminalBench21":0.891385767790262,"terminalBench40":0.48989898989899,'
    '"scicode":0.563657407407407,"lcr":0.743333333333333,'
    '"hle":0.548656163113994,"gpqa":0.912121212121212}}'
)


class TestTerminalBenchOnTheModelPages(unittest.TestCase):
    """The page's own spelling of the two live Terminal-Bench revisions.

    AA renamed the v2.1 field when it added the 4.0 run, and a field name the
    parser does not recognise is not an error here -- it is simply absent from
    the result, so the column keeps whatever it last held and the page fallback
    the free tier runs on stops feeding it without anything failing. That is the
    one failure mode worth a test: these pin the names against the live payload.
    """

    def metrics(self) -> dict:
        return aa._parse_metrics_block(aa._normalize_page_text(MODEL_PAGE), "claude-opus-5")

    def test_both_revisions_are_read_under_the_names_the_page_uses(self) -> None:
        metrics = self.metrics()
        self.assertAlmostEqual(metrics["terminalbench_4_0"], 0.48989898989899)
        self.assertAlmostEqual(metrics["terminalbench_v2_1"], 0.891385767790262)

    def test_a_revision_the_page_reports_as_null_stays_null(self) -> None:
        # Distinct from "the page does not carry this field": null is AA saying
        # it has not run the model, and it has to reach the column as None
        # rather than leaving the last value in place.
        self.assertIsNone(self.metrics()["terminalbench_hard"])

    def test_the_4_0_number_reaches_the_evaluations_block(self) -> None:
        # _PAGE_EVALS is what carries a page value into the dict update.py's
        # SCORE_MAPPINGS reads; a field parsed but not listed there goes nowhere.
        self.assertIn("terminalbench_4_0", aa._PAGE_EVALS)
        model = {"slug": "claude-opus-5"}
        with mock.patch.object(aa, "_fetch_page_metrics", return_value=self.metrics()):
            aa._enrich_structured_metrics([model])
        self.assertAlmostEqual(model["evaluations"]["terminalbench_4_0"], 0.48989898989899)


class TestHleAndGpqaOnTheModelPages(unittest.TestCase):
    """The free API tier carries neither, so the page is where they come from."""

    def metrics(self) -> dict:
        return aa._parse_metrics_block(aa._normalize_page_text(MODEL_PAGE), "claude-opus-5")

    def test_both_are_read(self) -> None:
        metrics = self.metrics()
        self.assertAlmostEqual(metrics["hle"], 0.548656163113994)
        self.assertAlmostEqual(metrics["gpqa"], 0.912121212121212)

    def test_both_reach_the_evaluations_block(self) -> None:
        model = {"slug": "claude-opus-5"}
        with mock.patch.object(aa, "_fetch_page_metrics", return_value=self.metrics()):
            aa._enrich_structured_metrics([model])
        self.assertAlmostEqual(model["evaluations"]["hle"], 0.548656163113994)
        self.assertAlmostEqual(model["evaluations"]["gpqa"], 0.912121212121212)

    def test_the_record_says_whether_its_page_was_read(self) -> None:
        read = {"slug": "claude-opus-5"}
        with mock.patch.object(aa, "_fetch_page_metrics", return_value={**self.metrics(), "page_read": True}):
            aa._enrich_structured_metrics([read])
        self.assertTrue(read["page_read"])

        failed = {"slug": "claude-opus-5"}
        aa._PAGE_METRICS_CACHE.pop("claude-opus-5", None)
        with mock.patch.object(aa, "_fetch_page_text", return_value=None):
            aa._enrich_structured_metrics([failed])
        aa._PAGE_METRICS_CACHE.pop("claude-opus-5", None)
        self.assertFalse(failed["page_read"])

    def test_an_api_value_is_not_displaced(self) -> None:
        model = {"slug": "claude-opus-5", "evaluations": {"hle": 0.5}}
        with mock.patch.object(aa, "_fetch_page_metrics", return_value=self.metrics()):
            aa._enrich_structured_metrics([model])
        self.assertEqual(model["evaluations"]["hle"], 0.5)


class TestFieldsTheFreeTierDropped(unittest.TestCase):
    """What AA's September 2026 free-tier cut left to the pages.

    The free route's evaluations block is down to the Intelligence, Coding and
    Agentic indices. τ²-Telecom, τ³-Banking and AA-LCR used to arrive with the
    API record only, so on free they stopped arriving at all -- and update.py
    has a column for each.
    """

    def metrics(self) -> dict:
        return aa._parse_metrics_block(aa._normalize_page_text(MODEL_PAGE), "claude-opus-5")

    def test_each_is_read_under_the_name_the_page_uses(self) -> None:
        metrics = self.metrics()
        self.assertAlmostEqual(metrics["tau_banking"], 0.612371134020619)
        self.assertAlmostEqual(metrics["lcr"], 0.743333333333333)
        # null is AA saying it has not run the model, not a missing field.
        self.assertIn("tau2", metrics)
        self.assertIsNone(metrics["tau2"])

    def test_the_flattened_omniscience_split_is_read(self) -> None:
        # The payload dropped the "omniscienceBreakdown" object and put both
        # halves on the record itself.
        metrics = self.metrics()
        self.assertAlmostEqual(metrics["omniscience_accuracy"], 0.6015)
        self.assertAlmostEqual(metrics["omniscience_hallucination_rate"], 0.4512)
        self.assertNotIn("omniscienceBreakdown", aa._PAGE_OBJECT_FIELDS)

    def test_they_reach_the_keys_update_reads(self) -> None:
        import update

        model = {"slug": "claude-opus-5"}
        with mock.patch.object(aa, "_fetch_page_metrics", return_value=self.metrics()):
            aa._enrich_structured_metrics([model])
        evals = model["evaluations"]
        for column in ("tau3_bench_banking", "aa_lcr", "aa_omniscience_accuracy", "aa_omniscience_hallucination"):
            with self.subTest(column=column):
                (aa_key, *_), _transform = update.SCORE_MAPPINGS[column]
                self.assertIsNotNone(evals.get(aa_key))
        self.assertIn("tau2", aa._PAGE_EVALS)

    def test_an_api_value_is_not_displaced(self) -> None:
        model = {"slug": "claude-opus-5", "evaluations": {"lcr": 0.5}}
        with mock.patch.object(aa, "_fetch_page_metrics", return_value=self.metrics()):
            aa._enrich_structured_metrics([model])
        self.assertEqual(model["evaluations"]["lcr"], 0.5)

    def test_a_field_the_record_lacks_is_not_read_off_the_next_model(self) -> None:
        # The page lists comparison models right after the current one, with
        # the same keys. A short record must not borrow its neighbour's value.
        page = (
            '{"currentModel":{"slug":"phi-4","microevalsEnabled":true,'
            '"tau2":0,"lcr":0}},{"slug":"other","microevalsEnabled":true,'
            '"tau2":0.19,"lcr":0,"tauBanking":0.3}'
        )
        metrics = aa._parse_metrics_block(page, "phi-4")
        self.assertEqual(metrics["tau2"], 0.0)
        self.assertNotIn("tau_banking", metrics)


class TestPageFieldsThatWentStale(unittest.TestCase):
    """The three names AA retired or renamed out from under the parser.

    `agenticIndex` and `codingIndex` are gone from the payload outright — AA
    replaced them with the per-industry `capabilities` block — and
    `harveyLabCriteriaPass` is `harveyLab` now. None of that failed anything:
    an unmatched name simply leaves the field out of the result.
    """

    def metrics(self) -> dict:
        return aa._parse_metrics_block(aa._normalize_page_text(MODEL_PAGE), "claude-opus-5")

    def test_harvey_is_read_under_the_shortened_name(self) -> None:
        self.assertAlmostEqual(self.metrics()["harvey_lab"], 0.93457808655377)

    def test_the_retired_composites_are_no_longer_expected(self) -> None:
        # Listing a name the payload does not carry is not harmless: it reads as
        # a fallback that exists, and hides that the column is API-only now.
        expected = {key for key, _ in aa._PAGE_FLOAT_FIELDS}
        self.assertNotIn("agenticIndex", expected)
        self.assertNotIn("codingIndex", expected)
        self.assertNotIn("harveyLabCriteriaPass", expected)

    def test_nothing_is_listed_for_consumption_that_no_parser_fills(self) -> None:
        # The drift that hid the rename: a name in _PAGE_EVALS with no entry
        # producing it is a page fallback that silently never fires.
        produced = {internal for _, internal in aa._PAGE_FLOAT_FIELDS}
        produced |= {internal for _, internal in aa._PAGE_BOOL_FIELDS}
        for subs in aa._PAGE_OBJECT_FIELDS.values():
            produced |= {internal for _, internal, _ in subs}
        for key in list(aa._PAGE_EVALS) + list(aa._PAGE_META_KEYS):
            with self.subTest(key=key):
                self.assertIn(key, produced)


class TestPageFieldAudit(unittest.TestCase):
    """--audit-page-fields, the check that would have caught all three."""

    def audit(self, page: str | None) -> tuple[int, str]:
        buf = io.StringIO()
        with mock.patch.object(aa, "_fetch_page_text", return_value=page):
            with contextlib.redirect_stdout(buf):
                status = aa._audit_page_fields(["claude-opus-5"])
        return status, buf.getvalue()

    def test_a_payload_carrying_every_expected_name_passes(self) -> None:
        keys = [f'"{key}":1' for key, _ in aa._PAGE_FLOAT_FIELDS]
        keys += [f'"{key}":true' for key, _ in aa._PAGE_BOOL_FIELDS if key != "microevalsEnabled"]
        keys += [f'"{name}":{{}}' for name in aa._PAGE_OBJECT_FIELDS]
        page = '{"slug":"claude-opus-5","microevalsEnabled":true,' + ",".join(keys) + "}"
        status, out = self.audit(page)
        self.assertEqual(status, 0)
        self.assertIn("no stale field names", out)

    def test_a_retired_name_is_reported_and_exits_non_zero(self) -> None:
        status, out = self.audit(MODEL_PAGE)
        self.assertEqual(status, 1)
        self.assertIn("STALE", out)
        # MODEL_PAGE is a trimmed block, so most names are missing from it; the
        # point is that a missing one is named against the key it would fill.
        self.assertIn("gdpval -> gdpval", out)

    def test_a_page_that_cannot_be_read_is_not_a_pass(self) -> None:
        status, _ = self.audit(None)
        self.assertEqual(status, 1)


class TestPageFallback(unittest.TestCase):
    def test_a_field_the_free_tier_drops_still_comes_off_the_page(self) -> None:
        free = {"slug": "gpt-oss-20b"}
        page_metrics = {
            "context_window": "128k",
            "params": "21B-A4B",
            "hugging_face_url": "https://huggingface.co/openai/gpt-oss-20b",
            "agentic_index": 27.6,
        }
        with mock.patch.object(aa, "_fetch_page_metrics", return_value=page_metrics):
            self.assertEqual(aa._extract_context_window(free), "128k")
            self.assertEqual(aa._extract_params(free), "21B-A4B")
            self.assertIs(aa._is_open_source(free), True)
            self.assertEqual(aa._extract_metric(free, "agentic_index"), 27.6)

    def test_the_hallucination_rate_is_inferred_only_where_the_page_has_none(self) -> None:
        # AA's API reports the complement. A rate the page states directly is
        # never displaced by one inferred from its opposite.
        reported = aa._normalize_model(
            {"slug": "a", "evaluations": {"aa_omniscience_non_hallucination_rate": 0.06}}
        )
        with mock.patch.object(aa, "_fetch_page_metrics", return_value={"omniscience_hallucination_rate": 0.9}):
            aa._enrich_structured_metrics([reported])
        self.assertEqual(reported["evaluations"]["omniscience_hallucination_rate"], 0.9)

        inferred = aa._normalize_model(
            {"slug": "a", "evaluations": {"aa_omniscience_non_hallucination_rate": 0.06}}
        )
        with mock.patch.object(aa, "_fetch_page_metrics", return_value={}):
            aa._enrich_structured_metrics([inferred])
        self.assertAlmostEqual(inferred["evaluations"]["omniscience_hallucination_rate"], 0.94)

    def test_the_page_fills_gaps_rather_than_overriding_the_api(self) -> None:
        model = aa._normalize_model(
            {
                "slug": "gpt-oss-20b",
                "evaluations": {"aa_omniscience_index": -63.92},
                "context_window_tokens": 131072,
            }
        )
        page_metrics = {
            "context_window": "1m",
            "params": "21B-A4B",
            "omniscience": -1.0,
            "briefcase": 1000.0,
            "ttft_p05": 0.28,
        }
        with mock.patch.object(aa, "_fetch_page_metrics", return_value=page_metrics):
            aa._enrich_structured_metrics([model])
        self.assertEqual(model["evaluations"]["omniscience"], -63.92)
        self.assertEqual(model["evaluations"]["briefcase"], 1000.0)
        self.assertEqual(model["context"], "128k")
        self.assertEqual(model["params"], "21B-A4B")
        self.assertEqual(model["ttft_p05"], 0.28)


class TestPageFetchRetry(unittest.TestCase):
    def setUp(self) -> None:
        aa._PAGE_METRICS_CACHE.clear()
        self.addCleanup(aa._PAGE_METRICS_CACHE.clear)

    def test_a_dropped_page_is_retried_before_it_is_written_off(self) -> None:
        # On the free tier the pages carry most of the columns, so writing one
        # off on the first blip silently drops every score on that model.
        calls = []
        body = '"slug":"a","microevalsEnabled":true,"critpt":0.5'

        def _get(url, headers=None, timeout=None):
            calls.append(url)
            if len(calls) == 1:
                raise aa.requests.ConnectionError("Connection reset by peer")
            return mock.Mock(status_code=200, text=body)

        with mock.patch.object(aa.requests, "get", _get), mock.patch.object(aa.time, "sleep", lambda _s: None):
            metrics = aa._fetch_page_metrics("a")
        self.assertEqual(len(calls), 2)
        self.assertEqual(metrics["critpt"], 0.5)

    def test_a_page_that_stays_down_leaves_the_record_empty(self) -> None:
        def _get(url, headers=None, timeout=None):
            raise aa.requests.ConnectionError("Connection reset by peer")

        with mock.patch.object(aa.requests, "get", _get), mock.patch.object(aa.time, "sleep", lambda _s: None):
            metrics = aa._fetch_page_metrics("a")
        self.assertEqual(metrics["context_window"], "")
        self.assertIsNone(metrics["mmmu_pro"])


class TestPublishedModelList(unittest.TestCase):
    """The file the admin page reads instead of proxying a live AA request."""

    def setUp(self) -> None:
        self.out = Path(tempfile.mkdtemp()) / "_aa" / "models.json"

    def models(self, count):
        return [
            {
                "slug": f"model-{i:04d}",
                "name": f"Model {i}",
                "model_creator": {"id": "x", "name": "Some Lab"},
                "release_date": "2026-01-01",
                "evaluations": {"artificial_analysis_intelligence_index": 1.0},
                "pricing": {"price_1m_input_tokens": 1.0},
            }
            for i in range(count)
        ]

    def test_it_carries_the_four_fields_and_nothing_else(self) -> None:
        # Scores, pricing and performance stay out: llm.json is where this
        # project publishes those, and they would churn the file every run.
        payload = aa._published_models(self.models(1))
        self.assertEqual(payload["count"], 1)
        self.assertEqual(
            payload["models"][0],
            {"slug": "model-0000", "name": "Model 0", "creator": "Some Lab", "release_date": "2026-01-01"},
        )

    def test_the_order_is_total_and_case_insensitive(self) -> None:
        models = [{"slug": s} for s in ("beta", "QwQ-32B", "Alpha", "alpha")]
        order = [m["slug"] for m in aa._published_models(models)["models"]]
        self.assertEqual(order, ["Alpha", "alpha", "beta", "QwQ-32B"])

    def test_two_records_for_one_slug_collapse_to_one(self) -> None:
        # Offset paging can hand the same model back twice when AA inserts a
        # model between two page reads.
        payload = aa._published_models([
            {"slug": "a", "name": "A", "release_date": "2026-01-01"},
            {"slug": "a", "name": "A", "release_date": "2026-01-01"},
            {"slug": "b"},
        ])
        self.assertEqual(payload["count"], 2)
        self.assertEqual([m["slug"] for m in payload["models"]], ["a", "b"])

    def test_a_shared_slug_resolves_the_same_way_whatever_order_it_arrives(self) -> None:
        # Sorting on the slug alone would leave these two in API order, which
        # is the one input to this file that is not stable run to run.
        first = {"slug": "a", "name": "Zebra", "release_date": "2026-01-01"}
        second = {"slug": "a", "name": "Alpha", "release_date": "2026-02-02"}
        forwards = aa._published_models([first, second])
        backwards = aa._published_models([second, first])
        self.assertEqual(forwards, backwards)
        self.assertEqual(forwards["models"][0]["name"], "Alpha")

    def test_records_without_a_slug_are_dropped(self) -> None:
        payload = aa._published_models([{"slug": "a"}, {"name": "no slug"}, {"slug": ""}])
        self.assertEqual(payload["count"], 1)

    def test_the_same_list_writes_the_same_bytes(self) -> None:
        # Committed on every refresh, so an unchanged list must be an empty
        # diff -- which is why there is no timestamp in it.
        models = self.models(aa.MIN_PUBLISHED_MODELS)
        self.assertEqual(aa._write_published_models(models, str(self.out)), 0)
        first = self.out.read_bytes()
        self.assertEqual(aa._write_published_models(list(reversed(models)), str(self.out)), 0)
        self.assertEqual(self.out.read_bytes(), first)
        self.assertTrue(first.endswith(b"\n"))

    def test_a_short_list_is_refused_rather_than_committed(self) -> None:
        # A bad answer from AA must leave yesterday's good list in place; an
        # emptied file would silently take the page's suggestions away.
        self.out.parent.mkdir(parents=True)
        self.out.write_text('{"count": 644, "models": []}', encoding="utf-8")
        rc = aa._write_published_models(self.models(aa.MIN_PUBLISHED_MODELS - 1), str(self.out))
        self.assertEqual(rc, 1)
        self.assertIn('"count": 644', self.out.read_text(encoding="utf-8"))

    def test_the_directory_is_created_on_a_first_publish(self) -> None:
        self.assertFalse(self.out.parent.exists())
        self.assertEqual(aa._write_published_models(self.models(aa.MIN_PUBLISHED_MODELS), str(self.out)), 0)
        self.assertEqual(json.loads(self.out.read_text())["count"], aa.MIN_PUBLISHED_MODELS)


class TestCreatorSlug(unittest.TestCase):
    def test_a_creator_without_a_slug_is_slugified_from_its_name(self) -> None:
        # The legacy route sent a slug; the V2 records send an id and a name.
        self.assertEqual(aa._creator_slug(PRO_MODEL), "openai")
        self.assertEqual(aa._creator_slug({"model_creator": {"name": "Z AI"}}), "z-ai")

    def test_a_slug_the_api_does_send_is_the_one_used(self) -> None:
        self.assertEqual(
            aa._creator_slug({"model_creator": {"slug": "openai", "name": "OpenAI"}}), "openai"
        )

    def test_a_record_without_a_creator_has_no_slug(self) -> None:
        self.assertEqual(aa._creator_slug({}), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
