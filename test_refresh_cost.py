#!/usr/bin/env python3
"""Tests for the three shortcuts a refresh takes. Run with ./test_refresh_cost.py

A refresh reads the same sources more than once and computes the same ranking
twice, and three changes cut that out. Each is only safe because it produces
the same answer as the work it skips, so that equivalence is what is pinned
here -- a shortcut that quietly starts answering differently is a wrong number
on the site, which is worse than the minutes it saved.

  * ``_cache`` serves a source twice inside one refresh. What it must never do
    is serve one request's answer to a different request, or keep an answer
    past its TTL, or turn a broken cache file into a broken run.
  * ``derive_indexes.py --report-only`` prints the ranking instead of refitting
    it. The tables have to be the ones a refit would have printed.
  * ``fetch_llmstats.benchmark_labels()`` is tested in test_hle_no_tools.py,
    next to the tool-mode rules it has to agree with.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import _cache
import derive_indexes as di
import fetch_huggingface


class TestResponseCache(unittest.TestCase):
    """The rules that keep a cache from becoming a source of wrong answers."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self._original = _cache.CACHE_DIR
        _cache.CACHE_DIR = self._dir.name

    def tearDown(self) -> None:
        _cache.CACHE_DIR = self._original
        self._dir.cleanup()

    def test_a_stored_payload_comes_back(self) -> None:
        _cache.store("source", "key", {"a": [1, 2]}, ttl=60)
        self.assertEqual(_cache.load("source", "key", ttl=60), {"a": [1, 2]})

    def test_a_different_key_misses(self) -> None:
        # The whole guard against answering a crawl of 164 models with a crawl
        # that only ever saw 163 of them.
        _cache.store("source", "key", ["old"], ttl=60)
        self.assertIsNone(_cache.load("source", "other", ttl=60))

    def test_an_expired_entry_misses(self) -> None:
        _cache.store("source", "key", ["old"], ttl=60)
        self.assertIsNone(_cache.load("source", "key", ttl=0))
        # A TTL of 0 also refuses to write, so nothing is left behind for a
        # later call with a longer one to find.
        _cache.store("source", "key2", ["new"], ttl=0)
        self.assertIsNone(_cache.load("source", "key2", ttl=60))

    def test_a_corrupt_file_is_a_miss_not_a_crash(self) -> None:
        Path(self._dir.name, "source.json").write_text("{not json", encoding="utf-8")
        self.assertIsNone(_cache.load("source", "key", ttl=60))

    def test_an_unwritable_directory_is_survivable(self) -> None:
        _cache.CACHE_DIR = "/proc/nonexistent/ai-bench"
        _cache.store("source", "key", ["x"], ttl=60)  # must not raise
        self.assertIsNone(_cache.load("source", "key", ttl=60))

    def test_a_payload_that_cannot_be_serialised_is_survivable(self) -> None:
        _cache.store("source", "key", {1, 2, 3}, ttl=60)  # a set: not JSON
        self.assertIsNone(_cache.load("source", "key", ttl=60))

    def test_the_key_does_not_depend_on_dict_order(self) -> None:
        self.assertEqual(
            _cache.digest({"a": 1, "b": 2}), _cache.digest({"b": 2, "a": 1})
        )
        self.assertNotEqual(_cache.digest(["a", "b"]), _cache.digest(["b", "a"]))

    def test_the_ttl_knob_reads_the_environment(self) -> None:
        var = "AI_BENCH_TEST_TTL"
        original = os.environ.get(var)
        try:
            os.environ.pop(var, None)
            self.assertEqual(_cache.ttl_seconds(var), _cache.DEFAULT_TTL_SECONDS)
            os.environ[var] = "0"
            self.assertEqual(_cache.ttl_seconds(var), 0)
            # A typo is not an instruction to disable the cache.
            os.environ[var] = "half an hour"
            self.assertEqual(_cache.ttl_seconds(var), _cache.DEFAULT_TTL_SECONDS)
        finally:
            if original is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = original


class TestTheHuggingFaceCrawlCache(unittest.TestCase):
    """The crawl two steps of a refresh share, and what it refuses to share."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self._original = _cache.CACHE_DIR
        _cache.CACHE_DIR = self._dir.name
        self._reader = fetch_huggingface.extract_scores_and_channels
        self.doc = {
            "models": [
                {"name": f"m{i}", "url": f"https://huggingface.co/org/m{i}"}
                for i in range(10)
            ]
        }

    def tearDown(self) -> None:
        fetch_huggingface.extract_scores_and_channels = self._reader
        _cache.CACHE_DIR = self._original
        self._dir.cleanup()

    def _crawl(self, fails: set[str] | None = None) -> list[dict]:
        failing = fails or set()

        def reader(
            repo: str, slug: str | None = None, failures: list[str] | None = None
        ) -> tuple[dict[str, float], dict[str, str]]:
            if repo in failing:
                raise RuntimeError("HTTP Error 401: Unauthorized")
            return {"mmlu": 1.0}, {"mmlu": fetch_huggingface.CHANNEL_METADATA}

        fetch_huggingface.extract_scores_and_channels = reader
        return fetch_huggingface.crawl_all_models(self.doc)

    def test_the_second_walk_reuses_the_first(self) -> None:
        first = self._crawl()
        reads: list[str] = []

        def counting(
            repo: str, slug: str | None = None, failures: list[str] | None = None
        ) -> tuple[dict[str, float], dict[str, str]]:
            reads.append(repo)
            return {}, {}

        fetch_huggingface.extract_scores_and_channels = counting
        self.assertEqual(fetch_huggingface.crawl_all_models(self.doc), first)
        self.assertEqual(reads, [])

    def test_a_model_added_to_the_file_is_not_answered_from_the_old_crawl(self) -> None:
        self._crawl()
        self.doc["models"].append(
            {"name": "new", "url": "https://huggingface.co/org/new"}
        )
        again = self._crawl()
        self.assertIn("new", {entry["model"] for entry in again})

    def test_one_gated_repo_does_not_stop_the_crawl_being_cached(self) -> None:
        # The normal case: a repo that answers 401 on every run would make a
        # "complete crawl only" rule into a cache that is never written.
        self._crawl(fails={"org/m3"})
        reads: list[str] = []

        def counting(
            repo: str, slug: str | None = None, failures: list[str] | None = None
        ) -> tuple[dict[str, float], dict[str, str]]:
            reads.append(repo)
            return {}, {}

        fetch_huggingface.extract_scores_and_channels = counting
        fetch_huggingface.crawl_all_models(self.doc)
        self.assertEqual(reads, [])

    def test_a_short_crawl_is_not_cached(self) -> None:
        # Half the cards lost to an outage is a stub, and serving a stub for an
        # hour is worse than walking the cards again.
        self._crawl(fails={f"org/m{i}" for i in range(5)})
        reads: list[str] = []

        def counting(
            repo: str, slug: str | None = None, failures: list[str] | None = None
        ) -> tuple[dict[str, float], dict[str, str]]:
            reads.append(repo)
            return {}, {}

        fetch_huggingface.extract_scores_and_channels = counting
        fetch_huggingface.crawl_all_models(self.doc)
        self.assertEqual(len(reads), 10)


class TestReportOnlyMatchesTheRefit(unittest.TestCase):
    """--report-only replaced a full refit in update-all; it has to print what
    the refit printed."""

    @staticmethod
    def _run(path: Path, *args: str) -> tuple[str, str]:
        """(stdout, stderr) of one derive_indexes.main() over `path`."""
        out, err = io.StringIO(), io.StringIO()
        argv = sys.argv
        sys.argv = ["derive_indexes.py", str(path), *args]
        try:
            with redirect_stdout(out), redirect_stderr(err):
                di.main()
        finally:
            sys.argv = argv
        return out.getvalue(), err.getvalue()

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name, "llm.json")
        # A slice of the real file rather than a synthetic one, so the report
        # meets the shapes it meets in production -- unranked models, revision
        # fallbacks, a reference row -- but few enough of them that fitting it
        # twice costs a moment instead of the 40 seconds the whole field does.
        source = json.loads(Path(di.DEFAULT_LLM_JSON).read_text(encoding="utf-8"))
        source["models"] = source["models"][:16]
        self.path.write_text(json.dumps(source), encoding="utf-8")
        # The state update-all leaves behind, and the only state this mode
        # claims anything about: a file whose columns the writer before it
        # already refreshed.
        self._run(self.path, "--write")

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_the_tables_are_the_refit_s_tables(self) -> None:
        refit, _ = self._run(self.path)
        report, _ = self._run(self.path, "--report-only")
        self.assertEqual(
            report.replace("Reported from the stored values; nothing recomputed.", ""),
            refit.replace("The derived indexes are up to date. Nothing to do.", ""),
        )

    def test_it_writes_nothing(self) -> None:
        before = self.path.read_bytes()
        self._run(self.path, "--report-only")
        self.assertEqual(self.path.read_bytes(), before)

    def test_it_says_so_when_an_index_was_never_written(self) -> None:
        doc = json.loads(self.path.read_text(encoding="utf-8"))
        for model in doc["models"]:
            model.get("scores", {}).pop("coding_index", None)
        self.path.write_text(json.dumps(doc), encoding="utf-8")
        _, errors = self._run(self.path, "--report-only")
        self.assertIn("carry no coding_index at all", errors)


if __name__ == "__main__":
    unittest.main(verbosity=2)
