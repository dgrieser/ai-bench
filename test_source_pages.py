#!/usr/bin/env python3
"""Tests for the page each queued name links to (_source_pages.py).

Run with ./test_source_pages.py. Offline. What is guarded: a name gets the
board its first row came from; `--format pages` lists exactly the names
`--format names` does; and the mapping modules read it back.
"""

from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import _source_pages
import fetch_tbench

ROWS = [
    {"model": "GLM 5.3", "benchmark": "terminal_bench_4_0"},
    {"model": "Fable 5.1", "benchmark": "terminal_bench_2_1"},
    {"model": "GLM 5.3", "benchmark": "terminal_bench_2_0"},
]


class TestPages(unittest.TestCase):
    def test_a_name_gets_the_board_its_first_row_came_from(self) -> None:
        pages = _source_pages.pages(ROWS, lambda r: fetch_tbench.board_url(r["benchmark"]))
        self.assertEqual(list(pages), ["GLM 5.3", "Fable 5.1"])
        self.assertEqual(pages["GLM 5.3"], fetch_tbench.board_url("terminal_bench_4_0"))

    def test_a_row_without_a_page_does_not_hide_a_later_one(self) -> None:
        pages = _source_pages.pages(
            [{"model": "x"}, {"model": "x", "url": "https://example.org/x"}],
            lambda r: r.get("url"),
        )
        self.assertEqual(pages, {"x": "https://example.org/x"})

    def test_one_board_is_one_page(self) -> None:
        self.assertEqual(_source_pages.same_page(["a", "b"], "https://example.org"),
                         {"a": "https://example.org", "b": "https://example.org"})


class TestFetcherFormat(unittest.TestCase):
    def run_main(self, fmt: str) -> str:
        out = io.StringIO()
        with mock.patch.object(fetch_tbench, "get_scores", return_value=ROWS), \
                mock.patch.object(sys, "argv", ["fetch_tbench.py", "--format", fmt]), \
                redirect_stdout(out):
            fetch_tbench.main()
        return out.getvalue()

    def test_pages_lists_the_names_names_does(self) -> None:
        pages = json.loads(self.run_main("pages"))
        self.assertEqual(set(pages), set(self.run_main("names").splitlines()))
        self.assertTrue(all(url.startswith("https://www.tbench.ai/") for url in pages.values()))


class TestRun(unittest.TestCase):
    def test_reads_the_mapping_back_and_drops_junk(self) -> None:
        printed = json.dumps({"a": "https://example.org/a", "b": None, "c": 3, "": "x"})
        done = mock.Mock(returncode=0, stdout=printed, stderr="")
        with mock.patch.object(_source_pages.subprocess, "run", return_value=done) as run:
            pages = _source_pages.run(Path("fetch_x.py"), "--all-models")
        self.assertEqual(pages, {"a": "https://example.org/a", "b": None, "c": None})
        self.assertEqual(run.call_args.args[0][2:], ["--format", "pages", "--all-models"])

    def test_a_failed_fetch_raises(self) -> None:
        failed = mock.Mock(returncode=1, stdout="", stderr="boom")
        with mock.patch.object(_source_pages.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                _source_pages.run(Path("fetch_x.py"))

    def test_a_fetcher_that_never_exits_is_killed_and_raises(self) -> None:
        # A hung fetcher used to hold its mapping updater, and update-all with
        # it, until the workflow cancelled the job (fetch_arena.py, 2026-10-30).
        hung = _source_pages.subprocess.TimeoutExpired(cmd="fetch_x.py", timeout=5.0)
        with mock.patch.dict(_source_pages.os.environ, {_source_pages.FETCH_TIMEOUT_VAR: "5"}):
            with mock.patch.object(_source_pages.subprocess, "run", side_effect=hung) as run:
                with self.assertRaisesRegex(RuntimeError, "killed after 5s"):
                    _source_pages.run(Path("fetch_x.py"))
        self.assertEqual(run.call_args.kwargs["timeout"], 5.0)


if __name__ == "__main__":
    unittest.main()
