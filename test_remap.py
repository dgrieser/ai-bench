#!/usr/bin/env python3
"""Tests for re-pointing mappings at a newly added model. Run with ./test_remap.py

A source name is asked about once and the answer is never revisited, so a row
mapped onto the nearest model while the right one did not exist stays there --
feeding its scores to the wrong model until test_propose.py fails on the next
run. _remap.py closes that at the moment the model is added. These tests pin
what moves (a normalized-equality match onto a different model, and only
that), what does not (a judgement, an ambiguity, a parked row), and that add.py
runs it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _matching
import _remap
import propose
from _openness import CLOSED_WEIGHTS, PENDING, UNMAPPABLE

HERE = Path(__file__).resolve().parent


def doc_with(*names: str) -> dict:
    return {"models": [{"name": name} for name in names]}


class FakeRoute:
    """Stands in for a propose.Route over a mapping file in a temp directory."""

    def __init__(self, path: Path, module: str = "_llmstats_mapping") -> None:
        self.path = path
        self.module = module
        self.mapping_const = "MAPPING"
        self.writer = "add_mapping"
        self.universe = propose.MODELS


class MappingFixture:
    """A mapping file on disk plus the route wiring _remap reads it through."""

    def __init__(self, stack: unittest.TestCase, mapping: dict, module: str = "_llmstats_mapping"):
        tmp = tempfile.TemporaryDirectory()
        stack.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "model-name-mapping-fake-to-artificialanalysis.json"
        self.write(mapping)
        self.route = FakeRoute(self.path, module)
        self.written: list[tuple[str, str]] = []

        def writer(source_name: str, slug: str) -> None:
            self.written.append((source_name, slug))
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            raw[source_name] = slug
            self.write(raw)

        patches = [
            mock.patch.object(_remap, "model_routes", return_value={self.path: self.route}),
            mock.patch.object(_remap, "_writer_for_path", lambda path: writer),
        ]
        for patch in patches:
            patch.start()
            stack.addCleanup(patch.stop)

    def write(self, mapping: dict) -> None:
        self.path.write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))


class TestFindStale(unittest.TestCase):
    def test_a_row_naming_the_nearest_model_is_repointable(self) -> None:
        """The bug this exists for: llm-stats' Fin row on the base model."""
        fixture = MappingFixture(self, {"ling-3.0-flash-fin": "ling-3-0-flash"})
        doc = doc_with("ling-3-0-flash", "ling-3-0-flash-fin")
        repointable, reportable = _remap.find_stale(doc, "ling-3-0-flash-fin")
        self.assertEqual(len(repointable), 1)
        self.assertEqual(repointable[0].source_name, "ling-3.0-flash-fin")
        self.assertEqual(repointable[0].was, "ling-3-0-flash")
        self.assertEqual(repointable[0].now, "ling-3-0-flash-fin")
        self.assertEqual(reportable, [])
        self.assertEqual(fixture.read()["ling-3.0-flash-fin"], "ling-3-0-flash")

    def test_a_row_already_naming_the_model_is_left_alone(self) -> None:
        MappingFixture(self, {"ling-3.0-flash-fin": "ling-3-0-flash-fin"})
        doc = doc_with("ling-3-0-flash", "ling-3-0-flash-fin")
        self.assertEqual(_remap.find_stale(doc, "ling-3-0-flash-fin"), ([], []))

    def test_a_parked_row_is_reported_never_rewritten(self) -> None:
        """Parking is an answer about whether the row belongs here at all.

        SWE-Rebench parks its plain gpt-oss-120b row and maps the -high one;
        whether that keeps two rows from folding into one model or is an
        oversight is not for a sweep to guess.
        """
        for sentinel in (UNMAPPABLE, PENDING, CLOSED_WEIGHTS):
            with self.subTest(sentinel):
                fixture = MappingFixture(self, {"ling-3.0-flash-fin": sentinel})
                doc = doc_with("ling-3-0-flash-fin")
                repointable, reportable = _remap.find_stale(doc, "ling-3-0-flash-fin")
                self.assertEqual(repointable, [])
                self.assertEqual([s.was for s in reportable], [sentinel])
                _remap.repoint(doc, "ling-3-0-flash-fin")
                self.assertEqual(fixture.read()["ling-3.0-flash-fin"], sentinel)

    def test_a_merely_similar_name_is_never_moved(self) -> None:
        """Only normalized equality moves; the rest is somebody's judgement."""
        MappingFixture(self, {"ling-3.0-flash": "ling-3-0-flash"})
        doc = doc_with("ling-3-0-flash", "ling-3-0-flash-fin")
        self.assertEqual(_remap.find_stale(doc, "ling-3-0-flash-fin"), ([], []))

    def test_ambiguity_moves_nothing(self) -> None:
        """Two models normalizing alike: neither may take the other's rows."""
        MappingFixture(self, {"foo.bar": "something-else"})
        doc = doc_with("foo-bar", "foo.bar", "something-else")
        self.assertEqual(_remap.find_stale(doc, "foo-bar"), ([], []))

    def test_a_model_not_in_the_document_matches_nothing(self) -> None:
        MappingFixture(self, {"ling-3.0-flash-fin": "ling-3-0-flash"})
        self.assertEqual(_remap.find_stale(doc_with("ling-3-0-flash"), "ling-3-0-flash-fin"), ([], []))

    def test_a_path_keyed_source_matches_on_its_last_segment(self) -> None:
        """Spheron keys are org/model; the org is not part of the name."""
        MappingFixture(
            self,
            {"inclusionAI/Ling-3.0-flash-Fin": "ling-3-0-flash"},
            module="_spheron_mapping",
        )
        doc = doc_with("ling-3-0-flash", "ling-3-0-flash-fin")
        repointable, _ = _remap.find_stale(doc, "ling-3-0-flash-fin")
        self.assertEqual([s.source_name for s in repointable], ["inclusionAI/Ling-3.0-flash-Fin"])

    def test_a_path_key_is_matched_whole_for_other_sources(self) -> None:
        """The last-segment rule is per source, not a general one."""
        MappingFixture(self, {"inclusionAI/Ling-3.0-flash-Fin": "ling-3-0-flash"})
        doc = doc_with("ling-3-0-flash", "ling-3-0-flash-fin")
        self.assertEqual(_remap.find_stale(doc, "ling-3-0-flash-fin"), ([], []))

    def test_a_malformed_file_is_skipped_not_raised(self) -> None:
        fixture = MappingFixture(self, {"ling-3.0-flash-fin": "ling-3-0-flash"})
        fixture.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(_remap.find_stale(doc_with("ling-3-0-flash-fin"), "ling-3-0-flash-fin"), ([], []))

    def test_a_list_target_is_not_a_model_name(self) -> None:
        MappingFixture(self, {"ling-3.0-flash-fin": ["a", "b"]})
        self.assertEqual(_remap.find_stale(doc_with("ling-3-0-flash-fin"), "ling-3-0-flash-fin"), ([], []))


class TestRepoint(unittest.TestCase):
    def test_repoint_writes_through_the_files_own_writer(self) -> None:
        fixture = MappingFixture(self, {"ling-3.0-flash-fin": "ling-3-0-flash"})
        doc = doc_with("ling-3-0-flash", "ling-3-0-flash-fin")
        repointed, reported = _remap.repoint(doc, "ling-3-0-flash-fin")
        self.assertEqual(fixture.written, [("ling-3.0-flash-fin", "ling-3-0-flash-fin")])
        self.assertEqual(fixture.read()["ling-3.0-flash-fin"], "ling-3-0-flash-fin")
        self.assertEqual(len(repointed), 1)
        self.assertEqual(reported, [])

    def test_write_false_reports_without_touching_the_file(self) -> None:
        fixture = MappingFixture(self, {"ling-3.0-flash-fin": "ling-3-0-flash"})
        doc = doc_with("ling-3-0-flash", "ling-3-0-flash-fin")
        repointed, _ = _remap.repoint(doc, "ling-3-0-flash-fin", write=False)
        self.assertEqual(len(repointed), 1)
        self.assertEqual(fixture.written, [])
        self.assertEqual(fixture.read()["ling-3.0-flash-fin"], "ling-3-0-flash")

    def test_repointing_is_idempotent(self) -> None:
        fixture = MappingFixture(self, {"ling-3.0-flash-fin": "ling-3-0-flash"})
        doc = doc_with("ling-3-0-flash", "ling-3-0-flash-fin")
        _remap.repoint(doc, "ling-3-0-flash-fin")
        again, _ = _remap.repoint(doc, "ling-3-0-flash-fin")
        self.assertEqual(again, [])
        self.assertEqual(len(fixture.written), 1)


class TestRoutes(unittest.TestCase):
    """The file set is read off the same table a rename goes by."""

    def test_only_model_mapping_files_are_searched(self) -> None:
        paths = _remap.model_routes()
        self.assertTrue(paths, "no mapping files resolved")
        for path in paths:
            self.assertIn("mapping", path.name)
            self.assertNotIn("benchmark", path.name)
            # The AA overrides run the other way -- model name to slug -- so
            # their values are not model names and must never be rewritten.
            self.assertNotIn("llm-to-artificialanalysis", path.name)

    def test_every_model_route_is_covered(self) -> None:
        import _rename

        self.assertEqual(
            sorted(p.name for p in _remap.model_routes()),
            sorted(_rename.route_path(r).name for r in _rename.value_routes()),
        )


class TestShippedMappings(unittest.TestCase):
    """The tree as committed: nothing stale is left for a model already here."""

    def test_no_mapping_names_a_model_other_than_the_one_it_matches(self) -> None:
        """The invariant test_propose.py enforces, checked from this side too.

        There it is "a proposal must not contradict a human mapping"; here it
        is "no mapping contradicts a model already in the index". Same fact,
        and this one names the model that would fix it.
        """
        doc = json.loads((HERE / "llm.json").read_text(encoding="utf-8"))
        offenders = []
        for model in doc["models"]:
            repointable, _ = _remap.find_stale(doc, model["name"])
            offenders.extend(repointable)
        self.assertEqual([str(o) for o in offenders], [])


class TestAddIntegration(unittest.TestCase):
    def test_add_py_repoints_on_the_way_in(self) -> None:
        """End to end: adding the model moves the row that named the old one."""
        with tempfile.TemporaryDirectory() as tmp:
            llm_path = Path(tmp) / "llm.json"
            llm_path.write_text(
                json.dumps(
                    {
                        "benchmarks": {},
                        "models": [
                            {
                                "name": "ling-3-0-flash",
                                "scores": {},
                                "scores_updated": {},
                                "scores_source": {},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            mapping_path = Path(tmp) / "model-name-mapping-fake-to-artificialanalysis.json"
            mapping_path.write_text(
                json.dumps({"ling-3.0-flash-fin": "ling-3-0-flash"}, indent=2) + "\n",
                encoding="utf-8",
            )

            # A stub source module whose writer edits the temp file, wired in
            # place of the real routes for this run only.
            stub = Path(tmp) / "_fake_mapping.py"
            stub.write_text(
                "import json\n"
                "from pathlib import Path\n"
                f"MAPPING = Path({str(mapping_path)!r})\n"
                "def add_mapping(source_name, slug):\n"
                "    raw = json.loads(MAPPING.read_text())\n"
                "    raw[source_name] = slug\n"
                "    MAPPING.write_text(json.dumps(raw, indent=2, sort_keys=True) + '\\n')\n",
                encoding="utf-8",
            )
            conftest = Path(tmp) / "sitecustomize.py"
            conftest.write_text(
                "import sys\n"
                f"sys.path.insert(0, {tmp!r})\n"
                "import propose, _rename, _fake_mapping\n"
                "route = propose.Route('_fake_mapping', 'MAPPING', 'add_mapping', propose.MODELS)\n"
                "_rename.value_routes = lambda: [route]\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "add.py"),
                    str(llm_path),
                    "--name", "ling-3-0-flash-fin",
                    "--url", "https://huggingface.co/inclusionAI/Ling-3.0-flash-Fin",
                    "--params", "100B-A6B",
                    "--context", "128k",
                    "--creator", "InclusionAI",
                    "--creator-url", "https://inclusionai.github.io/",
                    "--skip-osworld", "--skip-deepswe", "--skip-frontierswe",
                    "--skip-real-swe", "--skip-frontiercode", "--skip-swe-atlas",
                    "--skip-aa-coding-agents", "--skip-evals-report", "--skip-vals",
                    "--skip-swe-marathon", "--skip-spheron", "--skip-llmstats",
                    "--skip-huggingface",
                ],
                capture_output=True,
                text=True,
                cwd=tmp,
                env={"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{tmp}:{HERE}", "HOME": tmp},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("ling-3-0-flash-fin", json.loads(llm_path.read_text())["models"][-1]["name"])
            self.assertEqual(
                json.loads(mapping_path.read_text())["ling-3.0-flash-fin"],
                "ling-3-0-flash-fin",
                "add.py did not re-point the stale mapping",
            )
            self.assertIn("Re-pointed", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
