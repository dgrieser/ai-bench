#!/usr/bin/env python3
"""Tests for _mapping_log.py. Run with ./test_mapping_log.py

The log is the only place the admin page can learn *who* made a mapping, so the
two things worth guarding are the classification -- an explicit answer is the
person's, anything else a batch wrote is the pipeline's -- and that every file
a route owns is watched, so a source added to propose.ROUTES is logged the day
it is added. Offline: the git repositories here are temporary ones.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import _mapping_log
import propose
from _mapping_log import ADMIN, AUTO, HAND, PROPOSAL, Batch, Tracked


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


class RepoTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name)
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "config", "user.email", "t@example.com")
        git(self.repo, "config", "user.name", "Tester")
        self.file = Tracked("update_vals_mapping.py", "vals-mapping.json")
        self.write({"a": "model-a", "b": "__unmappable__"})
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "seed")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, mapping: dict) -> None:
        (self.repo / self.file.relpath).write_text(json.dumps(mapping), encoding="utf-8")

    def record(self, by: str, batch: Batch | None = None) -> list[dict]:
        return _mapping_log.record(by, batch=batch, cwd=self.repo, files=[self.file],
                                   at="2026-09-30T00:00:00Z")


class TestRecord(RepoTestCase):
    def test_an_unchanged_file_records_nothing(self) -> None:
        self.assertEqual(self.record(AUTO), [])

    def test_added_changed_and_removed_keys_are_each_one_entry(self) -> None:
        self.write({"a": "model-a2", "c": "model-c"})
        entries = {e["subject"]: e for e in self.record(AUTO)}
        self.assertEqual(sorted(entries), ["a", "b", "c"])
        self.assertEqual((entries["a"]["previous"], entries["a"]["value"]), ("model-a", "model-a2"))
        self.assertEqual((entries["b"]["previous"], entries["b"]["value"]), ("__unmappable__", None))
        self.assertEqual((entries["c"]["previous"], entries["c"]["value"]), (None, "model-c"))
        self.assertTrue(all(e["by"] == AUTO for e in entries.values()))
        self.assertTrue(all(e["route"] == "update_vals_mapping.py" for e in entries.values()))

    def test_a_batch_credits_only_what_it_asked_for_to_the_admin_page(self) -> None:
        """An add re-points a mapping by itself; that is the pipeline's doing."""
        self.write({"a": "model-a", "b": "__unmappable__", "c": "model-c", "d": "new-model"})
        batch = Batch(
            mappings=frozenset({("update_vals_mapping.py", "c")}),
            values={("update_vals_mapping.py", "c"): ["model-c"]},
            renames={},
            added=frozenset({"new-model"}),
        )
        entries = {e["subject"]: e for e in self.record(AUTO, batch)}
        self.assertEqual(entries["c"]["by"], ADMIN)
        self.assertNotIn("why", entries["c"])
        self.assertEqual(entries["d"]["by"], AUTO)
        self.assertIn("new-model", entries["d"]["why"])

    def test_an_answer_the_file_does_not_hold_is_not_credited(self) -> None:
        """Asked for one value, the file holds another: not the answer's write."""
        self.write({"a": "model-a", "b": "__unmappable__", "c": "model-x"})
        batch = Batch(frozenset({("update_vals_mapping.py", "c")}),
                      {("update_vals_mapping.py", "c"): ["model-c"]}, {}, frozenset())
        (entry,) = self.record(AUTO, batch)
        self.assertEqual(entry["by"], AUTO)

    def test_a_rename_is_explained(self) -> None:
        self.write({"a": "model-renamed", "b": "__unmappable__"})
        batch = Batch(frozenset(), {}, {"model-a": "model-renamed"}, frozenset())
        (entry,) = self.record(AUTO, batch)
        self.assertEqual(entry["by"], AUTO)
        self.assertIn("rename", entry["why"])

    def test_answer_py_s_result_is_what_a_batch_is_read_from(self) -> None:
        result = self.repo / "result.json"
        result.write_text(json.dumps({"ok": True, "applied": [], "records": [
            {"kind": "mapping", "route": "update_vals_mapping.py", "subject": "c", "value": "model-c"},
            {"kind": "model-rename", "route": None, "subject": "old", "value": "new"},
            {"kind": "model-create", "route": None, "subject": "fresh", "value": None},
        ]}), encoding="utf-8")
        batch = Batch.from_result(result)
        self.assertEqual(batch.mappings, frozenset({("update_vals_mapping.py", "c")}))
        self.assertEqual(batch.renames, {"old": "new"})
        self.assertEqual(batch.added, frozenset({"fresh"}))
        self.assertEqual(Batch.from_result(self.repo / "missing.json"), Batch.empty())

    def test_a_string_and_a_one_element_list_are_the_same_answer(self) -> None:
        """The AA file's values may be lists; the record may carry either."""
        batch = Batch(frozenset({("r", "k")}), {("r", "k"): ["x"]}, {}, frozenset())
        self.assertEqual(batch.classify("r", "k", None, "x")[0], ADMIN)
        self.assertEqual(batch.classify("r", "k", None, ["x"])[0], ADMIN)

    def test_an_unknown_author_kind_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.record("somebody")


class TestBackfill(unittest.TestCase):
    def test_commits_are_classified_by_what_the_workflow_writes(self) -> None:
        bot = _mapping_log.BOT_AUTHOR
        self.assertEqual(_mapping_log.classify_commit(bot, "chore(mappings): apply answers from the admin UI"), ADMIN)
        self.assertEqual(_mapping_log.classify_commit(
            "David", "Merge pull request #12 from dgrieser/chore/mapping-proposals"), PROPOSAL)
        self.assertEqual(_mapping_log.classify_commit(bot, "chore(data): benchmark refresh"), AUTO)
        self.assertEqual(_mapping_log.classify_commit("David", "fix(mapping): x"), HAND)


class TestTrackedFiles(unittest.TestCase):
    def test_every_route_s_file_is_watched(self) -> None:
        watched = {t.route for t in _mapping_log.tracked_files()}
        self.assertEqual(watched, set(propose.ROUTES))
        for tracked in _mapping_log.tracked_files():
            self.assertTrue((_mapping_log.HERE / tracked.relpath).exists(), tracked.relpath)

    def test_the_committed_log_is_well_formed(self) -> None:
        """The admin page reads it straight from the repository."""
        if not _mapping_log.LOG_PATH.exists():
            self.skipTest("no log committed yet")
        doc = json.loads(_mapping_log.LOG_PATH.read_text(encoding="utf-8"))
        entries = doc["entries"]
        self.assertLessEqual(len(entries), _mapping_log.MAX_ENTRIES)
        routes = set(propose.ROUTES)
        for entry in entries:
            self.assertIn(entry["by"], _mapping_log.BY)
            self.assertIn(entry["route"], routes)
            self.assertRegex(entry["at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual([e["at"] for e in entries], sorted(e["at"] for e in entries),
                         "oldest first: the page reads the tail as the newest")

    def test_save_keeps_the_newest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            entries = [{"at": f"2026-01-01T00:00:{i % 60:02d}Z", "n": i}
                       for i in range(_mapping_log.MAX_ENTRIES + 5)]
            _mapping_log.save(entries, path)
            kept = _mapping_log.load(path)
            self.assertEqual(len(kept), _mapping_log.MAX_ENTRIES)
            self.assertEqual(kept[-1]["n"], _mapping_log.MAX_ENTRIES + 4)


if __name__ == "__main__":
    unittest.main()
