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
