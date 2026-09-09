#!/usr/bin/env python3
"""Tests for managing a model entry by hand. Run with ./test_model_admin.py

Three scripts, one subject: the entry a person creates, corrects and renames
when no source did it for them.

  add.py     builds the entry, from the metadata given and whatever Artificial
             Analysis knows about the name
  edit.py    changes any of that metadata afterwards -- the same field set, so
             nothing is enterable once and then frozen
  rename.py  gives the entry a different slug, in llm.json and in every mapping
             file that names it

The rename is the one that has to be exhaustive rather than merely correct: a
name left behind in a mapping file does not fail, it silently stops matching,
and the scores that source used to write simply stop arriving.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import _rename
import edit

HERE = Path(__file__).resolve().parent

LLM_DOC = {
    "benchmarks": {"hle": {"name": "HLE"}, "coding_index": {"name": "Coding", "derived": True}},
    "models": [
        {
            "name": "hand-added-1",
            "date_added": "2026-02-03",
            "url": None,
            "params": None,
            "context": None,
            "creator": {"name": None, "url": None},
            "scores": {"hle": None},
            "scores_updated": {"hle": None},
            "scores_source": {"hle": None},
        },
        {
            "name": "other-model",
            "date_added": "2026-01-01",
            "url": None,
            "params": None,
            "context": None,
            "creator": {"name": "Someone", "url": None},
            "scores": {"hle": None},
            "scores_updated": {"hle": None},
            "scores_source": {"hle": None},
        },
    ],
}


class ScriptTestCase(unittest.TestCase):
    """A temporary llm.json and a way to run one of the scripts against it."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.llm = self.tmp / "llm.json"
        self.llm.write_text(json.dumps(LLM_DOC), encoding="utf-8")

    def run_script(self, script: str, *flags: str) -> subprocess.CompletedProcess:
        # --flag=VALUE and the path after a bare --, the form infer_json_file()
        # reads unambiguously; stdin closed so the run is non-interactive.
        return subprocess.run(
            [sys.executable, str(HERE / script), *flags, "--", str(self.llm)],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )

    def model(self, name: str = "hand-added-1") -> dict:
        doc = json.loads(self.llm.read_text(encoding="utf-8"))
        return next(m for m in doc["models"] if m["name"] == name)


class TestEditMetadata(ScriptTestCase):
    def test_every_field_add_py_asks_for_can_be_changed_afterwards(self) -> None:
        """The two sets are the same on purpose: nothing is entered once and frozen."""
        self.assertEqual(
            set(edit.METADATA_FIELDS),
            {"params", "context", "url", "creator", "creator_url", "date_added"},
        )
        result = self.run_script(
            "edit.py",
            "--model=hand-added-1",
            "--url=https://huggingface.co/acme/model-1",
            "--params=123B",
            "--context=256k",
            "--creator=Acme",
            "--creator-url=https://acme.example/",
            "--date-added=2026-02-04",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        model = self.model()
        self.assertEqual(model["url"], "https://huggingface.co/acme/model-1")
        self.assertEqual(model["params"], "123B")
        self.assertEqual(model["context"], "256k")
        self.assertEqual(model["date_added"], "2026-02-04")
        self.assertEqual(model["creator"], {"name": "Acme", "url": "https://acme.example/"})

    def test_the_creator_fields_land_inside_the_creator_object(self) -> None:
        """Two of the six are nested, so nothing may assume model[key]."""
        self.run_script("edit.py", "--model=hand-added-1", "--creator=Acme")
        self.assertEqual(self.model()["creator"]["name"], "Acme")
        self.assertNotIn("creator_url", self.model())

    def test_null_clears_a_field(self) -> None:
        self.run_script("edit.py", "--model=hand-added-1", "--creator=Acme")
        self.run_script("edit.py", "--model=hand-added-1", "--creator=null")
        self.assertIsNone(self.model()["creator"]["name"])

    def test_a_url_that_is_not_one_is_refused(self) -> None:
        for flag in ("--url=acme.example", "--creator-url=acme.example"):
            result = self.run_script("edit.py", "--model=hand-added-1", flag)
            self.assertEqual(result.returncode, 1)
            self.assertIn("http://", result.stderr)
        self.assertIsNone(self.model()["url"])

    def test_a_date_that_is_not_one_is_refused(self) -> None:
        result = self.run_script("edit.py", "--model=hand-added-1", "--date-added=03.02.2026")
        self.assertEqual(result.returncode, 1)
        self.assertIn("YYYY-MM-DD", result.stderr)
        self.assertEqual(self.model()["date_added"], "2026-02-03")


class TestAddMetadata(ScriptTestCase):
    def test_a_model_can_be_created_with_all_of_its_metadata(self) -> None:
        result = self.run_script(
            "add.py",
            "--skip-aa",
            "--name=hand-added-2",
            "--url=https://huggingface.co/acme/model-2",
            "--params=70B",
            "--context=128k",
            "--creator=Acme",
            "--creator-url=https://acme.example/",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        model = self.model("hand-added-2")
        self.assertEqual(model["creator"], {"name": "Acme", "url": "https://acme.example/"})
        self.assertEqual(model["url"], "https://huggingface.co/acme/model-2")
        self.assertEqual(model["date_added"], date.today().isoformat())
        # Every benchmark key present and null: update.py fills them, and a key
        # that is simply absent reads as "never measured" to nothing at all.
        self.assertEqual(set(model["scores"]), {"hle", "coding_index"})
        self.assertEqual(set(model["scores_updated"]), set(model["scores"]))

    def test_a_duplicate_name_is_refused(self) -> None:
        result = self.run_script("add.py", "--skip-aa", "--name=other-model")
        self.assertEqual(result.returncode, 1)
        self.assertIn("already exists", result.stderr)


class TestRename(unittest.TestCase):
    """rename.py, end to end, in a copy of the repository.

    Copied rather than mocked because the paths that matter are the ones the
    modules compute from their own location: the point of the test is that every
    mapping file is found, and a fake route table would only prove that the fake
    was complete.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        for script in HERE.glob("*.py"):
            shutil.copy2(script, self.repo / script.name)
        (self.repo / "llm.json").write_text(json.dumps(LLM_DOC), encoding="utf-8")
        self.write("model-name-mapping-tbench-to-artificialanalysis.json", {
            "Hand Added 1": "hand-added-1",
            "Other Model": "other-model",
        })
        self.write("model-name-mapping-spheron-to-artificialanalysis.json", {
            "acme/Hand-Added-1": "hand-added-1",
        })
        self.write("model-name-mapping-llm-to-artificialanalysis.json", {
            "hand-added-1": ["aa-one", "aa-two"],
        })
        self.write("model-name-mapping-llm-to-artificialanalysis-ignored.json", {
            "hand-added-1": ["aa-three"],
        })
        self.write("check_new-decisions.json", {"hand-added-1": "__added__"})

    def write(self, name: str, data) -> None:
        (self.repo / name).write_text(
            json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def read(self, name: str):
        return json.loads((self.repo / name).read_text(encoding="utf-8"))

    def rename(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(self.repo / "rename.py"), *args],
            capture_output=True,
            text=True,
            cwd=self.repo,
        )

    def test_a_dry_run_writes_nothing(self) -> None:
        before = self.read("model-name-mapping-tbench-to-artificialanalysis.json")
        result = self.rename("hand-added-1", "acme-model-1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Nothing written", result.stdout)
        self.assertEqual(self.read("model-name-mapping-tbench-to-artificialanalysis.json"), before)

    def test_the_name_moves_in_every_file_that_holds_it(self) -> None:
        result = self.rename("hand-added-1", "acme-model-1", "-w")
        self.assertEqual(result.returncode, 0, result.stderr)

        names = [m["name"] for m in self.read("llm.json")["models"]]
        self.assertEqual(names, ["acme-model-1", "other-model"])
        self.assertEqual(
            self.read("model-name-mapping-tbench-to-artificialanalysis.json"),
            {"Hand Added 1": "acme-model-1", "Other Model": "other-model"},
        )
        self.assertEqual(
            self.read("model-name-mapping-spheron-to-artificialanalysis.json"),
            {"acme/Hand-Added-1": "acme-model-1"},
        )
        # The AA files are keyed by the llm.json name, not valued by it.
        self.assertEqual(
            self.read("model-name-mapping-llm-to-artificialanalysis.json"),
            {"acme-model-1": ["aa-one", "aa-two"]},
        )
        self.assertEqual(
            self.read("model-name-mapping-llm-to-artificialanalysis-ignored.json"),
            {"acme-model-1": ["aa-three"]},
        )
        self.assertEqual(self.read("check_new-decisions.json"), {"acme-model-1": "__added__"})

    def test_scores_travel_with_the_entry(self) -> None:
        self.rename("hand-added-1", "acme-model-1", "-w")
        model = next(m for m in self.read("llm.json")["models"] if m["name"] == "acme-model-1")
        self.assertEqual(model["date_added"], "2026-02-03")
        self.assertIn("hle", model["scores"])

    def test_a_name_that_is_taken_is_refused(self) -> None:
        result = self.rename("hand-added-1", "other-model", "-w")
        self.assertEqual(result.returncode, 2)
        self.assertIn("already a model", result.stderr)

    def test_a_name_that_is_not_a_slug_is_refused(self) -> None:
        result = self.rename("hand-added-1", "Acme Model 1", "-w")
        self.assertEqual(result.returncode, 2)
        self.assertIn("slug", result.stderr)

    def test_an_unknown_model_is_refused(self) -> None:
        result = self.rename("ghost", "acme-model-1", "-w")
        self.assertEqual(result.returncode, 2)
        self.assertIn("not a model", result.stderr)

    def test_every_model_route_is_covered_rather_than_listed(self) -> None:
        """Read off propose.ROUTES, so a source added there is renamed too."""
        import propose

        expected = {
            route.mapping_const
            for by_kind in propose.ROUTES.values()
            for route in by_kind.values()
            if route.universe == propose.MODELS
        }
        covered = {route.mapping_const for route in _rename.value_routes()}
        self.assertEqual(covered, expected)


if __name__ == "__main__":
    unittest.main()
