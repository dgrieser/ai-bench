#!/usr/bin/env python3
"""Fetchers fail loudly on a layout change or a scale flip (issue #228).

Each case below used to end in a fetcher exiting 0 with an empty or rescaled
board: a renamed column, a missing table, a second table merged into the
first, a pass@5 row read as pass@1, fractions where percentages were expected.
Now each raises, update.py skips that one source and exits non-zero, and a
value outside its benchmark's range is refused by apply_score().
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _fetch_checks as checks
import _scale_labs
import fetch_bfcl
import fetch_evals_report
import fetch_frontierswe
import update
from _scores import check_score_range, score_range

LLM_JSON = Path(__file__).resolve().with_name("llm.json")


def flight(*chunks: str) -> str:
    """A page whose self.__next_f payload is the given strings, in order."""
    return "".join(
        f"<script>self.__next_f.push([1,{json.dumps(chunk)}])</script>" for chunk in chunks
    )


class TestRequire(unittest.TestCase):
    def test_empty_rows_raise(self) -> None:
        with self.assertRaises(ValueError):
            checks.require_rows([], "https://example.com")
        checks.require_rows([{}], "https://example.com")

    def test_missing_column_raises_and_names_it(self) -> None:
        with self.assertRaisesRegex(ValueError, "'pass@1'"):
            checks.require_columns(["model", "pass@3"], ("model", "pass@1"), "src")
        checks.require_columns(["model", "pass@1", "x"], ("model", "pass@1"), "src")


class TestScaleChecks(unittest.TestCase):
    def test_percentages_out_of_range_raise(self) -> None:
        with self.assertRaises(ValueError):
            checks.check_percentages([{"score": 4200.0}], "src")
        with self.assertRaises(ValueError):
            checks.check_percentages([{"score": -1}], "src")

    def test_a_whole_board_of_fractions_raises(self) -> None:
        rows = [{"score": v} for v in (0.61, 0.55, 0.42, 0.3, 0.12)]
        with self.assertRaisesRegex(ValueError, "fractions"):
            checks.check_percentages(rows, "src")

    def test_a_small_board_under_one_percent_is_allowed(self) -> None:
        checks.check_percentages([{"score": 0.4}, {"score": 0.9}], "src")

    def test_a_real_percentage_board_passes(self) -> None:
        checks.check_percentages([{"score": v} for v in (88.1, 0.0, 12, None)], "src")

    def test_fractions(self) -> None:
        checks.check_fractions([0, 0.5, 1, None], "src")
        with self.assertRaises(ValueError):
            checks.check_fractions([0.5, 42.0], "src")


class TestScoreRange(unittest.TestCase):
    DOC = {
        "benchmarks": {
            "pct": {},
            "elo": {"range": [-1000, 3000]},
            "bad": {"range": [5, 1]},
        }
    }

    def test_default_is_a_percentage(self) -> None:
        self.assertEqual(score_range(self.DOC, "pct"), (0.0, 100.0))
        self.assertEqual(score_range(self.DOC, "unknown"), (0.0, 100.0))
        # A malformed declaration does not switch the check off.
        self.assertEqual(score_range(self.DOC, "bad"), (0.0, 100.0))

    def test_declared_range(self) -> None:
        self.assertEqual(score_range(self.DOC, "elo"), (-1000.0, 3000.0))
        check_score_range(self.DOC, "elo", 1845)
        with self.assertRaises(ValueError):
            check_score_range(self.DOC, "elo", 5000)

    def test_none_passes(self) -> None:
        check_score_range(self.DOC, "pct", None)

    def test_apply_score_refuses_a_rescaled_value(self) -> None:
        # Refused and reported, not raised: the rest of the run still lands.
        model = {"name": "m", "scores": {"pct": 42.0}}
        update.REJECTED_SCORES.clear()
        self.assertEqual(
            update.apply_score(self.DOC, model, "m", "pct", 4200.0, "https://x", []), 0
        )
        self.assertEqual(model["scores"]["pct"], 42.0)
        self.assertEqual(len(update.REJECTED_SCORES), 1)
        self.assertIn("outside its range", update.REJECTED_SCORES[0])
        update.REJECTED_SCORES.clear()

    def test_apply_score_refuses_a_non_number(self) -> None:
        model = {"name": "m", "scores": {"pct": 42.0}}
        update.REJECTED_SCORES.clear()
        self.assertEqual(update.apply_score(self.DOC, model, "m", "pct", "42%", "https://x", []), 0)
        self.assertEqual(model["scores"]["pct"], 42.0)
        self.assertEqual(len(update.REJECTED_SCORES), 1)
        update.REJECTED_SCORES.clear()

    def test_llm_json_holds_every_stored_score_in_range(self) -> None:
        doc = json.loads(LLM_JSON.read_text(encoding="utf-8"))
        for model in doc["models"]:
            for key, value in (model.get("scores") or {}).items():
                with self.subTest(model=model.get("name"), key=key):
                    check_score_range(doc, key, value)


class TestEvalsReport(unittest.TestCase):
    HEADER = "<tr><th>Model</th><th>Lab</th><th>Score ↓</th><th>Status</th></tr>"

    def page(self, *rows: tuple[str, str, str]) -> str:
        body = "".join(
            f"<tr><td>{m}</td><td>Lab</td><td>{s}</td><td>{st}</td></tr>" for m, s, st in rows
        )
        return f'<table class="score-table">{self.HEADER}{body}</table>'

    def test_parse_score_reads_the_metric(self) -> None:
        self.assertEqual(fetch_evals_report.parse_score("23.0% (pass@5)"), (23.0, "pass@5"))
        self.assertEqual(fetch_evals_report.parse_score("4.0% (pass@1)"), (4.0, "pass@1"))
        self.assertEqual(fetch_evals_report.parse_score("61.2%"), (61.2, None))
        self.assertIsNone(fetch_evals_report.parse_score("61.2% ± 2"))
        self.assertIsNone(fetch_evals_report.parse_score("—"))

    def test_missing_table_raises(self) -> None:
        with self.assertRaises(ValueError):
            fetch_evals_report.extract_rows("<html>no table</html>")

    def test_renamed_score_column_raises(self) -> None:
        page = '<table class="score-table"><tr><th>Model</th><th>Result</th><th>Status</th></tr></table>'
        with self.assertRaises(ValueError):
            fetch_evals_report.extract_rows(page)

    def _scores(self, page: str) -> list[dict]:
        original = fetch_evals_report.fetch_html
        fetch_evals_report.fetch_html = lambda url: page
        try:
            return fetch_evals_report.get_scores(["zerobench"], include_unverified=False)
        finally:
            fetch_evals_report.fetch_html = original

    def test_pass_at_5_rows_are_dropped(self) -> None:
        page = self.page(
            ("GPT-5.4", "23.0% (pass@5)", "Verified"),
            ("GPT-5 mini", "4.0% (pass@1)", "Verified"),
        )
        self.assertEqual([r["model"] for r in self._scores(page)], ["GPT-5 mini"])

    def test_no_trusted_row_raises(self) -> None:
        page = self.page(("GPT-5.4", "23.0% (pass@5)", "Verified"))
        with self.assertRaises(ValueError):
            self._scores(page)


class TestScaleLabsBoard(unittest.TestCase):
    ROWS = [{"model": "A", "score": 80.0}, {"model": "B", "score": 70.0}]

    def payload(self, slug: str, *tables: list) -> str:
        parts = [f'0:{{"query":{{"slug":"{slug}"}}}}\n']
        parts += [f'1c:["$","div",null,{{"entries":{json.dumps(t)}}}]\n' for t in tables]
        return flight(*parts)

    def test_reads_the_board(self) -> None:
        rows = _scale_labs.extract_board_rows(self.payload("mcp_atlas", self.ROWS), "mcp_atlas", "src")
        self.assertEqual([r["model"] for r in rows], ["A", "B"])

    def test_a_repeated_table_is_one_table(self) -> None:
        html = self.payload("mcp_atlas", self.ROWS, self.ROWS)
        self.assertEqual(len(_scale_labs.extract_board_rows(html, "mcp_atlas", "src")), 2)

    def test_another_board_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not identify"):
            _scale_labs.extract_board_rows(self.payload("sweatlas-qna", self.ROWS), "mcp_atlas", "src")

    def test_two_tables_are_refused(self) -> None:
        html = self.payload("mcp_atlas", self.ROWS, [{"model": "C", "score": 1.0}])
        with self.assertRaisesRegex(ValueError, "2 different row tables"):
            _scale_labs.extract_board_rows(html, "mcp_atlas", "src")

    def test_a_stray_flat_object_is_not_a_row(self) -> None:
        # The old reader took any {"model":..,"score":..} anywhere in the payload.
        stray = '9:{"model":"Sidebar","score":99}\n'
        html = self.payload("mcp_atlas", self.ROWS) + flight(stray)
        models = [r["model"] for r in _scale_labs.extract_board_rows(html, "mcp_atlas", "src")]
        self.assertNotIn("Sidebar", models)

    def test_no_table_raises(self) -> None:
        with self.assertRaises(ValueError):
            _scale_labs.extract_board_rows(self.payload("mcp_atlas"), "mcp_atlas", "src")


class TestFrontierSweEntries(unittest.TestCase):
    BOARD = {"abs": {"mean": [{"model": "M {x}", "harness": "h}", "overall": 40.0}]}}

    def test_braces_inside_strings_do_not_end_the_object(self) -> None:
        html = flight('x:{"entries":' + json.dumps(self.BOARD) + "}")
        self.assertEqual(fetch_frontierswe.extract_entries(html), self.BOARD)

    def test_a_non_board_entries_key_is_skipped(self) -> None:
        html = flight('a:{"entries":{"count":3}}\n', 'b:{"entries":' + json.dumps(self.BOARD) + "}")
        self.assertEqual(fetch_frontierswe.extract_entries(html), self.BOARD)

    def test_two_boards_are_refused(self) -> None:
        other = {"abs": {"mean": [{"model": "N", "overall": 1.0}]}}
        html = flight(
            'a:{"entries":' + json.dumps(self.BOARD) + "}\n",
            'b:{"entries":' + json.dumps(other) + "}",
        )
        with self.assertRaises(ValueError):
            fetch_frontierswe.extract_entries(html)


class TestBfclSeries(unittest.TestCase):
    def test_v4_is_accepted(self) -> None:
        fetch_bfcl.check_series('<a>BFCL-v1</a> <a>BFCL-v3</a> <a>BFCL-v4</a>')

    def test_a_newer_series_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "BFCL-v5"):
            fetch_bfcl.check_series('<a>BFCL-v4</a> <a>BFCL-v5</a>')

    def test_no_series_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            fetch_bfcl.check_series("<html></html>")


class TestUpdateSourceIsolation(unittest.TestCase):
    def test_a_failed_source_is_recorded_and_contributes_nothing(self) -> None:
        def fetch_broken_data() -> dict:
            raise RuntimeError("fetch_broken.py failed (1): Traceback\nValueError: no rows")

        failures: list[str] = []
        self.assertEqual(update.source_data(failures, fetch_broken_data), {})
        self.assertEqual(failures, ["broken: ValueError: no rows"])

    def test_a_working_source_passes_through(self) -> None:
        failures: list[str] = []
        self.assertEqual(update.source_data(failures, lambda: {"m": {}}), {"m": {}})
        self.assertEqual(failures, [])


class TestRoundTrips(unittest.TestCase):
    # One source moving its own number and moving it back. Two sources cannot
    # stand in for it: no two that can overwrite each other share a rank now.
    VALS = "https://www.vals.ai/benchmarks/mmlu_pro"

    def doc(self) -> dict:
        return {
            "benchmarks": {"mmlu_pro": {}},
            "models": [
                {
                    "name": "m",
                    "scores": {"mmlu_pro": 79.4},
                    "scores_updated": {"mmlu_pro": "2026-09-01"},
                    "scores_source": {"mmlu_pro": self.VALS},
                }
            ],
        }

    def test_a_write_and_its_undo_restore_the_date(self) -> None:
        doc = self.doc()
        found = update.snapshot_scores(doc)
        model = doc["models"][0]
        changes: list = []
        update.apply_score(doc, model, "m", "mmlu_pro", 80.9, self.VALS, changes)
        update.apply_score(doc, model, "m", "mmlu_pro", 79.4, self.VALS, changes)
        self.assertEqual(len(changes), 2)
        self.assertEqual(update.undo_round_trips(doc, found, changes), [])
        self.assertEqual(model["scores_updated"]["mmlu_pro"], "2026-09-01")

    def test_a_real_change_is_kept(self) -> None:
        doc = self.doc()
        found = update.snapshot_scores(doc)
        model = doc["models"][0]
        changes: list = []
        update.apply_score(doc, model, "m", "mmlu_pro", 80.9, self.VALS, changes)
        self.assertEqual(update.undo_round_trips(doc, found, changes), changes)
        self.assertNotEqual(model["scores_updated"]["mmlu_pro"], "2026-09-01")


class TestSourceUpdate(unittest.TestCase):
    def test_a_raising_ingest_returns_the_empty_result(self) -> None:
        def update_broken_scores(doc: dict) -> tuple:
            raise KeyError("score")

        failures: list[str] = []
        self.assertEqual(
            update.source_update(failures, (0, 0, []), "update_broken_scores", update_broken_scores, {}),
            (0, 0, []),
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("update_broken_scores", failures[0])


class TestFetchTimeout(unittest.TestCase):
    def setUp(self) -> None:
        self._old = os.environ.get(update.FETCH_TIMEOUT_VAR)
        os.environ[update.FETCH_TIMEOUT_VAR] = "0.5"

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop(update.FETCH_TIMEOUT_VAR, None)
        else:
            os.environ[update.FETCH_TIMEOUT_VAR] = self._old

    def test_a_hung_prefetched_fetcher_is_killed(self) -> None:
        cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
        update.prefetch([cmd])
        proc = update.run_fetch(cmd, "hung")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("TimeoutError", proc.stderr)

    def test_a_hung_direct_fetcher_is_killed(self) -> None:
        proc = update.run_fetch([sys.executable, "-c", "import time; time.sleep(30)"], "hung")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("TimeoutError", proc.stderr)


class TestScoresAreWrittenWhateverFails(unittest.TestCase):
    """The run's contract: every score that could be read lands in llm.json,
    and the exit status is what reports the failures."""

    def test_main_writes_the_working_source_and_exits_1(self) -> None:
        doc = json.loads(LLM_JSON.read_text(encoding="utf-8"))
        model = next(m for m in doc["models"] if "toolathlon" in (m.get("scores") or {}))
        slug = model["name"]
        new_score = 12.3 if model["scores"]["toolathlon"] != 12.3 else 45.6

        def broken(*_args: object, **_kwargs: object) -> dict:
            raise RuntimeError("fetcher failed (1): ValueError: the page moved")

        def refresh_broken(_doc: dict) -> None:
            raise RuntimeError("index fit failed")

        patches = {
            name: broken for name in dir(update)
            if name.startswith("fetch_") and name.endswith("_data")
        }
        patches["fetch_available_slugs"] = broken
        patches["fetch_toolathlon_data"] = lambda *_a: {slug: {"score": new_score}}
        patches["prefetch"] = lambda _cmds: None
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm.json"
            path.write_text(LLM_JSON.read_text(encoding="utf-8"), encoding="utf-8")
            argv = ["update.py", "-w", str(path)]
            with mock.patch.multiple(update, **patches), \
                    mock.patch.object(update.derive_indexes, "refresh_and_report", refresh_broken), \
                    mock.patch.object(sys, "argv", argv), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                status = update.main()
            written = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(status, 1)
        after = next(m for m in written["models"] if m["name"] == slug)
        self.assertEqual(after["scores"]["toolathlon"], new_score)


class TestPrefetch(unittest.TestCase):
    def test_a_prefetched_command_is_collected_not_rerun(self) -> None:
        cmd = [sys.executable, "-c", "print('hello')"]
        update.prefetch([cmd])
        self.assertIn(tuple(cmd), update._PREFETCHED)
        proc = update.run_fetch(cmd, "probe")
        self.assertEqual(proc.stdout.strip(), "hello")
        self.assertNotIn(tuple(cmd), update._PREFETCHED)

    def test_an_unprefetched_command_runs_directly(self) -> None:
        proc = update.run_fetch([sys.executable, "-c", "print(2)"], "probe")
        self.assertIsInstance(proc, subprocess.CompletedProcess)
        self.assertEqual(proc.stdout.strip(), "2")


if __name__ == "__main__":
    unittest.main()
