#!/usr/bin/env python3
"""A broken fetcher is a warning on the run, not a failed run.

Run with ./test_fetch_warnings.py

Three places have to agree for that to hold: update.py's exit status, the one
update-all derives from its steps, and the workflow step that turns update-all's
"warnings only" status into a green step. The admin page then reads the
warnings back by their title, so the prefix api.php filters on is checked too.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

import _fetch_warnings
import update

HERE = Path(__file__).resolve().parent
LLM_JSON = HERE / "llm.json"
WORKFLOW = HERE / ".github" / "workflows" / "update-benchmarks.yml"


class TestRecord(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.path = self.tmp / "w.jsonl"

    def test_nothing_is_written_without_a_target(self) -> None:
        with mock.patch.dict(os.environ, {_fetch_warnings.ENV_VAR: ""}):
            _fetch_warnings.record("update.py", "boom", source="tbench")
        self.assertFalse(self.path.exists())

    def test_the_env_var_names_the_file(self) -> None:
        with mock.patch.dict(os.environ, {_fetch_warnings.ENV_VAR: str(self.path)}):
            _fetch_warnings.record("update.py", "boom\n", source="tbench")
            _fetch_warnings.record("update_bfcl_mapping.py", "gone")
        entries = _fetch_warnings.load(self.path)
        self.assertEqual([e["step"] for e in entries], ["update.py", "update_bfcl_mapping.py"])
        self.assertEqual(entries[0]["message"], "boom")

    def test_an_unwritable_target_does_not_raise(self) -> None:
        """It runs on the failure path; a second failure there helps nobody."""
        with contextlib.redirect_stderr(io.StringIO()):
            _fetch_warnings.record("update.py", "boom", path=self.tmp / "missing" / "w.jsonl")

    def test_a_torn_line_is_skipped(self) -> None:
        self.path.write_text('{"step": "update.py"}\n{"step": \n', encoding="utf-8")
        self.assertEqual(len(_fetch_warnings.load(self.path)), 1)

    def test_a_missing_file_is_no_warnings(self) -> None:
        self.assertEqual(_fetch_warnings.load(self.tmp / "absent"), [])


class TestAnnotate(unittest.TestCase):
    def test_the_title_carries_the_prefix_api_php_filters_on(self) -> None:
        line = _fetch_warnings.annotation({"step": "update.py", "source": "tbench", "message": "x"})
        self.assertTrue(line.startswith("::warning title=Fetcher failed%3A tbench::"), line)

    def test_a_multiline_message_stays_one_command(self) -> None:
        line = _fetch_warnings.annotation(
            {"step": "update_bfcl_mapping.py", "message": "Traceback\nValueError: 100% gone"}
        )
        self.assertNotIn("\n", line)
        self.assertIn("%0A", line)
        self.assertIn("100%25 gone", line)

    def test_property_separators_are_escaped_in_the_title(self) -> None:
        line = _fetch_warnings.annotation({"step": "s", "source": "a,b:c", "message": "m"})
        self.assertIn("a%2Cb%3Ac", line)

    def test_the_summary_names_the_exception_not_the_traceback_header(self) -> None:
        text = _fetch_warnings.summary(
            [{"step": "update_bfcl_mapping.py", "message": "Traceback (most recent call last):\nValueError: moved"}]
        )
        self.assertIn("ValueError: moved", text)
        self.assertNotIn("Traceback", text)

    def test_routes_are_the_mapping_updaters_only(self) -> None:
        entries = [
            {"step": "update.py", "source": "tbench"},
            {"step": "update_bfcl_mapping.py", "source": None},
            {"step": "update_bfcl_mapping.py", "source": None},
        ]
        self.assertEqual(_fetch_warnings.failed_routes(entries), ["update_bfcl_mapping.py"])


class TestUpdatePyExitStatus(unittest.TestCase):
    """update.py exits WARNINGS_ONLY_EXIT when every failure was a fetch."""

    def run_main(self, patches: dict, refresh=None) -> tuple[int, Path]:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        path = tmp / "llm.json"
        warnings = tmp / "w.jsonl"
        path.write_text(LLM_JSON.read_text(encoding="utf-8"), encoding="utf-8")
        patches = dict(patches, prefetch=lambda _cmds: None)
        extra = []
        if refresh is not None:
            extra.append(mock.patch.object(update.derive_indexes, "refresh_and_report", refresh))
        with mock.patch.multiple(update, **patches), \
                mock.patch.object(sys, "argv", ["update.py", "-w", str(path)]), \
                mock.patch.dict(os.environ, {_fetch_warnings.ENV_VAR: str(warnings)}), \
                contextlib.ExitStack() as stack, \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            for patch in extra:
                stack.enter_context(patch)
            status = update.main()
        return status, warnings

    @staticmethod
    def broken_fetchers() -> dict:
        def broken(*_args: object, **_kwargs: object) -> dict:
            raise RuntimeError("fetcher failed (1): ValueError: the page moved")

        patches = {
            name: broken for name in dir(update)
            if name.startswith("fetch_") and name.endswith("_data")
        }
        patches["fetch_available_slugs"] = broken
        return patches

    def test_only_fetch_failures_is_a_warning(self) -> None:
        # The index fit is slow and not what is under test; a no-op keeps it
        # from being the failure that decides the status.
        status, warnings = self.run_main(self.broken_fetchers(), refresh=lambda _doc: None)
        self.assertEqual(status, _fetch_warnings.WARNINGS_ONLY_EXIT)
        entries = _fetch_warnings.load(warnings)
        # Every stub shares one name, which is what source_data() reports.
        self.assertEqual({e["source"] for e in entries}, {"artificialanalysis", "broken"})
        self.assertTrue(all(e["step"] == "update.py" for e in entries))
        self.assertTrue(all("the page moved" in e["message"] for e in entries))

    def test_a_failure_of_its_own_still_wins(self) -> None:
        def refresh_broken(_doc: dict) -> None:
            raise RuntimeError("index fit failed")

        status, _ = self.run_main(self.broken_fetchers(), refresh=refresh_broken)
        self.assertEqual(status, 1)


STUB = "#!/bin/sh\n{body}\n"


class TestUpdateAll(unittest.TestCase):
    """update-all run against stub steps, so only its bookkeeping is under test."""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        shutil.copy(HERE / "update-all", self.dir)
        shutil.copy(HERE / "_fetch_warnings.py", self.dir)
        for name in ("fill_source_urls.py", "check_new.py", "derive_indexes.py",
                     "pending_prompts.py", "update_ok_mapping.py"):
            self.stub(name, "exit 0")
        self.stub("update.py", "exit 0")
        self.warnings = self.dir / "w.jsonl"

    def stub(self, name: str, body: str) -> None:
        path = self.dir / name
        path.write_text(STUB.format(body=body), encoding="utf-8")
        path.chmod(0o755)

    def run_all(self) -> int:
        env = dict(os.environ, **{_fetch_warnings.ENV_VAR: str(self.warnings)})
        return subprocess.run(
            [str(self.dir / "update-all")], env=env, capture_output=True, text=True, check=False,
        ).returncode

    def test_a_clean_run_exits_0(self) -> None:
        self.assertEqual(self.run_all(), 0)
        self.assertEqual(_fetch_warnings.load(self.warnings), [])

    def test_a_failed_mapping_updater_is_a_warning_with_its_reason(self) -> None:
        self.stub("update_bad_mapping.py", 'echo "ValueError: page moved" >&2; exit 1')
        self.assertEqual(self.run_all(), _fetch_warnings.WARNINGS_ONLY_EXIT)
        entries = _fetch_warnings.load(self.warnings)
        self.assertEqual([e["step"] for e in entries], ["update_bad_mapping.py"])
        self.assertIn("page moved", entries[0]["message"])

    def test_update_py_skipping_sources_is_a_warning(self) -> None:
        self.stub("update.py", f"exit {_fetch_warnings.WARNINGS_ONLY_EXIT}")
        self.assertEqual(self.run_all(), _fetch_warnings.WARNINGS_ONLY_EXIT)

    def test_update_py_failing_is_a_failure(self) -> None:
        self.stub("update.py", "exit 1")
        self.assertEqual(self.run_all(), 1)

    def test_a_non_fetcher_step_is_a_failure_even_beside_warnings(self) -> None:
        self.stub("update_bad_mapping.py", "exit 1")
        self.stub("check_new.py", "exit 1")
        self.assertEqual(self.run_all(), 1)

    def test_a_stale_warnings_file_is_truncated(self) -> None:
        self.warnings.write_text('{"step": "update.py", "source": "old"}\n', encoding="utf-8")
        self.assertEqual(self.run_all(), 0)
        self.assertEqual(_fetch_warnings.load(self.warnings), [])


class TestWorkflow(unittest.TestCase):
    def steps(self) -> list[dict]:
        doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        return doc["jobs"]["update"]["steps"]

    def step(self, name: str) -> dict:
        return next(s for s in self.steps() if s.get("name") == name)

    def test_the_refresh_turns_warnings_only_into_success(self) -> None:
        refresh = self.step("Refresh benchmarks")
        self.assertIn(_fetch_warnings.ENV_VAR, refresh.get("env", {}))
        self.assertIn(f'-eq {_fetch_warnings.WARNINGS_ONLY_EXIT}', refresh["run"])

    def test_the_warnings_are_annotated_after_the_refresh(self) -> None:
        names = [s.get("name") for s in self.steps()]
        self.assertEqual(names.index("Report fetcher warnings"), names.index("Refresh benchmarks") + 1)
        self.assertIn("_fetch_warnings.py annotate", self.step("Report fetcher warnings")["run"])

    def test_the_queue_carries_over_the_failed_routes(self) -> None:
        self.assertIn("--carry-over", self.step("Render the pending queue")["run"])

    def test_api_php_filters_on_the_same_title(self) -> None:
        api = (HERE / "_admin" / "api.php").read_text(encoding="utf-8")
        found = re.search(r"const WARNING_TITLE = '([^']+)';", api)
        self.assertIsNotNone(found, "api.php: no WARNING_TITLE to check")
        self.assertEqual(found.group(1), _fetch_warnings.TITLE_PREFIX)


if __name__ == "__main__":
    unittest.main()
