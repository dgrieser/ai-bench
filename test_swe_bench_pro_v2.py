#!/usr/bin/env python3
"""Tests for the SWE-Bench Pro V2 fetcher. Run with ./test_swe_bench_pro_v2.py

Scale publishes V2's Full set and Hard subset as two variants of one board, on
one page. Each has to land in its own column -- Hard runs 5-40 points under
Full for the same model -- and neither may ever reach `swe_bench_pro`, the
original benchmark, whose scores sit on another scale.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import _precedence
import fetch_swe_bench_pro_v2 as v2
import fill_source_urls
import update

MAPPING = Path(__file__).with_name("model-name-mapping-swe-bench-pro-v2-to-artificialanalysis.json")
LLM_JSON = Path(__file__).with_name("llm.json")


def flight(*chunks: str) -> str:
    return "".join(
        f"<script>self.__next_f.push([1,{json.dumps(chunk)}])</script>" for chunk in chunks
    )


def board(full: list[dict], hard: list[dict], slug: str = v2.BOARD_SLUG) -> str:
    variants = [
        {"key": "full", "label": "SWE-Bench Pro V2 Full", "entries": full},
        {"key": "hard", "label": "SWE-Bench Pro V2 HARD", "entries": hard},
    ]
    return flight(
        f'0:{{"query":{{"slug":"{slug}"}}}}\n',
        f'1c:["$","div",null,{{"variants":{json.dumps(variants)}}}]\n',
    )


def row(model: str, score: float, **extra) -> dict:
    return {"model": model, "score": score, "company": "x", **extra}


# Labels from the live board (2026-10-22). The two tables spell some names
# differently; each pair has to land on one key.
FULL = [
    row("Fable 5.1 (Claude Code) high", 99.1),
    row("GPT-6-Astra (Codex) high", 96.9),
    row("GPT-5.6-sol (Codex) xhigh", 95.5),
    row("Kimi-K3 (mini-swe-agent) max", 97.7),
]
HARD = [
    row("Fable 5.1 (Claude Code) high", 92.2),
    row("GPT-6 Astra (Codex) high ", 90.2),
    row("GPT-5.6 Sol (Codex) xhigh", 82.4),
    row("Haiku 4.5 (Claude Code) xhigh", 25.5),
]


def scores(html: str, variants=("full", "hard"), include_deprecated=False) -> list[dict]:
    fetched: list[str] = []

    def fetch(url: str) -> str:
        fetched.append(url)
        return html

    with mock.patch.object(v2, "fetch_html", fetch), contextlib.redirect_stderr(io.StringIO()):
        rows = v2.get_scores(list(variants), include_deprecated)
    scores.fetched = fetched
    return rows


class EachVariantItsOwnColumn(unittest.TestCase):
    def test_variants_route_to_their_keys(self):
        rows = scores(board(FULL, HARD))
        by = {(r["model"], r["key"]): r["score"] for r in rows}
        self.assertEqual(by[("fable 5.1", "swe_bench_pro_2_0")], 99.1)
        self.assertEqual(by[("fable 5.1", "swe_bench_pro_hard_2_0")], 92.2)
        self.assertNotIn("swe_bench_pro", {r["key"] for r in rows})

    def test_the_page_is_read_once_for_both_variants(self):
        scores(board(FULL, HARD))
        self.assertEqual(scores.fetched, [v2.URL])

    def test_the_two_spellings_share_one_key(self):
        rows = scores(board(FULL, HARD))
        keys = {r["key"]: set() for r in rows}
        for r in rows:
            keys[r["key"]].add(r["model"])
        self.assertIn("gpt 6 astra", keys["swe_bench_pro_2_0"] & keys["swe_bench_pro_hard_2_0"])
        self.assertIn("gpt 5.6 sol", keys["swe_bench_pro_2_0"] & keys["swe_bench_pro_hard_2_0"])

    def test_the_label_keeps_its_harness_and_loses_its_padding(self):
        rows = [r for r in scores(board(FULL, HARD)) if r["model"] == "gpt 6 astra"]
        self.assertEqual({r["raw"] for r in rows}, {"GPT-6-Astra (Codex) high", "GPT-6 Astra (Codex) high"})
        self.assertEqual({r["harness"] for r in rows}, {"Codex"})

    def test_ranks_are_per_variant(self):
        rows = scores(board(FULL, HARD))
        full = [r for r in rows if r["variant"] == "full"]
        self.assertEqual([r["rank"] for r in full], [1, 2, 3, 4])
        self.assertEqual(full[0]["model"], "fable 5.1")

    def test_one_variant_alone(self):
        rows = scores(board(FULL, HARD), variants=("hard",))
        self.assertEqual({r["key"] for r in rows}, {"swe_bench_pro_hard_2_0"})

    def test_deprecated_rows_are_dropped_unless_asked_for(self):
        html = board(FULL + [row("Old (mini-swe-agent)", 10.0, deprecated=True)], HARD)
        self.assertNotIn("old", {r["model"] for r in scores(html)})
        self.assertIn("old", {r["model"] for r in scores(html, include_deprecated=True)})

    def test_the_v1_board_is_refused(self):
        with self.assertRaisesRegex(ValueError, "does not identify"):
            scores(board(FULL, HARD, slug="swe_bench_pro"))


class KeepsTheMapping(unittest.TestCase):
    def test_every_mapping_key_is_a_normalized_label(self):
        for key in json.loads(MAPPING.read_text()):
            self.assertEqual(key, key.strip().lower(), key)

    def test_every_mapped_slug_is_a_model(self):
        models = {m["name"] for m in json.loads(LLM_JSON.read_text())["models"]}
        for name, slug in json.loads(MAPPING.read_text()).items():
            if not slug.startswith("__"):
                self.assertIn(slug, models, name)


class Ingest(unittest.TestCase):
    def test_best_harness_wins_per_column(self):
        payload = [
            {"model": "m", "key": "swe_bench_pro_2_0", "score": 90.0},
            {"model": "m", "key": "swe_bench_pro_2_0", "score": 95.0},
            {"model": "m", "key": "swe_bench_pro_hard_2_0", "score": 60.0},
            {"model": "unmapped", "key": "swe_bench_pro_2_0", "score": 99.0},
        ]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        mapping = Path(tmp.name) / "mapping.json"
        mapping.write_text(json.dumps({"m": "slug"}), encoding="utf-8")
        for order in (payload, payload[::-1]):
            proc = SimpleNamespace(returncode=0, stdout=json.dumps(order), stderr="")
            with mock.patch.object(update, "run_fetch", return_value=proc):
                by_slug = update.fetch_swe_bench_pro_v2_data(Path("unused.py"), mapping)
            self.assertEqual(by_slug, {"slug": {"swe_bench_pro_2_0": 95.0, "swe_bench_pro_hard_2_0": 60.0}})

    def test_both_columns_cite_the_board_at_the_benchmark_site_rank(self):
        url = _precedence.SWE_BENCH_PRO_V2_SOURCE_URL
        self.assertEqual(url, v2.URL)
        self.assertEqual(_precedence.source_rank(url), _precedence.RANK_BENCHMARK_SITE)
        self.assertTrue(_precedence.is_fetcher_source(url))

    def test_the_v1_board_is_not_the_v2_board(self):
        self.assertFalse(_precedence.is_fetcher_source("https://labs.scale.com/leaderboard/swe_bench_pro"))

    def test_the_page_is_listed_for_both_columns(self):
        inventory = dict(fill_source_urls.build_inventory())
        self.assertEqual(set(inventory[v2.URL]), set(v2.VARIANTS.values()))


class Columns(unittest.TestCase):
    def test_both_columns_exist_beside_v1(self):
        keys = list(json.loads(LLM_JSON.read_text())["benchmarks"])
        i = keys.index("swe_bench_pro")
        self.assertEqual(keys[i + 1 : i + 3], list(v2.VARIANTS.values()))

    def test_v1_excludes_v2(self):
        excludes = json.loads(LLM_JSON.read_text())["benchmarks"]["swe_bench_pro"]["excludes"]
        self.assertTrue(any("V2" in e for e in excludes))


if __name__ == "__main__":
    unittest.main(verbosity=2)
