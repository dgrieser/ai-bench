#!/usr/bin/env python3
"""Tests for run_tests.py, the daily full test run. Run with ./test_run_tests.py

The report is the only thing the admin page's Tests tab has, so the checks are
about what reaches it: every way a test file can fail is reported as a failure
(a file that cannot be imported, hangs or exits is not "no tests, all passed"),
each failure keeps what its test printed and its docstring, ``since`` survives
from one run to the next, and a run that changes nothing leaves the file alone
-- otherwise the workflow would commit every day.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

import run_tests

SUITE = textwrap.dedent('''
    import unittest

    class T(unittest.TestCase):
        def test_ok(self):
            pass

        def test_bad(self):
            """Guards something worth knowing."""
            print("what the test printed")
            self.assertEqual(1, 2)

        def test_raises(self):
            raise KeyError("missing")

        def test_sub(self):
            for i in range(3):
                with self.subTest(i=i):
                    self.assertLess(i, 2)

        @unittest.skip("not today")
        def test_skipped(self):
            pass
''')


class TestRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.report = self.root / "_tests" / "report.json"

    def write(self, name: str, text: str) -> None:
        (self.root / name).write_text(text, encoding="utf-8")

    def run_all(self, *extra: str) -> int:
        with contextlib.redirect_stdout(io.StringIO()):
            return run_tests.main(["--root", str(self.root), "--report", str(self.report),
                                   "--timeout", "3", *extra])

    def failures(self) -> dict[str, dict]:
        body = json.loads(self.report.read_text(encoding="utf-8"))
        return {f["test"]: f for f in body["failures"]}

    def test_every_kind_of_failure_is_reported_with_its_details(self) -> None:
        self.write("test_suite.py", SUITE)
        self.assertEqual(self.run_all(), 1)
        found = self.failures()
        self.assertEqual(sorted(found), [
            "test_suite.T.test_bad",
            "test_suite.T.test_raises",
            "test_suite.T.test_sub (i=2)",
        ])
        bad = found["test_suite.T.test_bad"]
        self.assertEqual(bad["kind"], "failure")
        self.assertEqual(bad["file"], "test_suite.py")
        self.assertEqual(bad["message"], "AssertionError: 1 != 2")
        self.assertIn("self.assertEqual(1, 2)", bad["traceback"])
        self.assertNotIn(str(self.root), bad["traceback"])
        self.assertEqual(bad["stdout"], "what the test printed\n")
        self.assertEqual(bad["doc"], "Guards something worth knowing.")
        self.assertIsInstance(bad["line"], int)
        self.assertEqual(found["test_suite.T.test_raises"]["kind"], "error")

        totals = json.loads(self.report.read_text(encoding="utf-8"))["totals"]
        self.assertEqual(totals, {"files": 1, "files_broken": 0, "tests": 5,
                                  "failed": 3, "skipped": 1, "passed": 1})

    def test_a_file_that_never_reports_is_a_failure_not_an_empty_pass(self) -> None:
        self.write("test_import.py", "import a_module_that_is_not_there\n")
        self.write("test_exit.py", "import os\nos._exit(4)\n")
        self.write("test_hang.py", "import time\ntime.sleep(30)\n")
        self.assertEqual(self.run_all(), 1)
        kinds = {f["file"]: f["kind"] for f in self.failures().values()}
        self.assertEqual(kinds, {"test_import.py": "import", "test_exit.py": "crash",
                                 "test_hang.py": "timeout"})

    def test_all_green_exits_zero_and_lists_nothing(self) -> None:
        self.write("test_fine.py", "import unittest\nclass T(unittest.TestCase):\n"
                                   "    def test_a(self): pass\n")
        self.assertEqual(self.run_all(), 0)
        self.assertEqual(self.failures(), {})

    def test_since_is_kept_while_a_test_keeps_failing(self) -> None:
        self.write("test_suite.py", SUITE)
        self.run_all()
        body = json.loads(self.report.read_text(encoding="utf-8"))
        body["failures"][0]["since"] = "2026-01-01T00:00:00Z"
        kept = body["failures"][0]["test"]
        self.report.write_text(json.dumps(body), encoding="utf-8")
        self.run_all()
        found = self.failures()
        self.assertEqual(found[kept]["since"], "2026-01-01T00:00:00Z")
        others = [f["since"] for t, f in found.items() if t != kept]
        self.assertTrue(all(s != "2026-01-01T00:00:00Z" for s in others))

    def test_a_run_that_changes_nothing_leaves_the_report_alone(self) -> None:
        self.write("test_suite.py", SUITE)
        self.run_all()
        first = self.report.read_text(encoding="utf-8")
        self.run_all()
        self.assertEqual(self.report.read_text(encoding="utf-8"), first)
        # A fixed test is a change.
        self.write("test_suite.py", SUITE.replace("self.assertEqual(1, 2)", "pass"))
        self.run_all()
        self.assertNotIn("test_suite.T.test_bad", self.failures())


if __name__ == "__main__":
    unittest.main()
