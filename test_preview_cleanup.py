#!/usr/bin/env python3
"""Tests for cleaning up a preview entry once its model is released.

Run with ./test_preview_cleanup.py

Two halves, as in _preview_cleanup.py:

  detection  which entries are listed, what evidence says they are released,
             and what each score and mapping is suggested to do -- pure, so it
             runs against an in-memory fixture shaped like the Hy4 release
             that prompted it;
  applying   a `preview-cleanup` record through answer.py, end to end, in a
             copy of the repository: the paths that matter are the ones the
             modules compute from their own location, so a mock would only
             prove the mock was complete.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import _answers
import _history
import _mapping_log
import _preview_cleanup as cleanup
import propose

HERE = Path(__file__).resolve().parent

HF_PREVIEW = "https://huggingface.co/tencent/Hy4-preview"
HF_RELEASE = "https://huggingface.co/tencent/Hy4"
AA_PAGE = "https://artificialanalysis.ai/models/tencent-hy4"
VALS_CYBER = "https://www.vals.ai/benchmarks/cyber-v1"
ARENA_AGENT = "https://arena.ai/leaderboard/agent"


def model(name: str = "hy4-preview", url: str | None = HF_PREVIEW, scores=None) -> dict:
    scores = scores if scores is not None else {
        # benchmark: (value, date, source)
        "hle": (43.4, "2026-08-29", HF_PREVIEW),                # preview card: fill-only
        "cyberbench_1_0": (57.5, "2026-09-28", VALS_CYBER),     # before the release
        "arena_agent": (5.4, "2026-11-05", ARENA_AGENT),        # board maps both
        "aa_lcr": (50.0, "2026-11-05", AA_PAGE),                # the release
    }
    return {
        "name": name,
        "date_added": "2026-08-29",
        "url": url,
        "creator": {"name": "Tencent", "url": None},
        "scores": {"coding_index": 1234, **{k: v[0] for k, v in scores.items()}},
        "scores_updated": {"coding_index": "2026-11-05", **{k: v[1] for k, v in scores.items()}},
        "scores_source": {"coding_index": None, **{k: v[2] for k, v in scores.items()}},
    }


BENCHMARKS = {
    "coding_index": {"name": "Coding", "derived": True},
    "hle": {"name": "HLE"},
    "cyberbench_1_0": {"name": "CyberBench v1"},
    "arena_agent": {"name": "Arena Agent"},
    "aa_lcr": {"name": "AA-LCR"},
}


def inputs(**overrides) -> cleanup.Inputs:
    base = dict(
        doc={"benchmarks": BENCHMARKS, "models": [model(), model("other", None, {})]},
        mappings={
            ("update_arena_mapping.py", "*"): {"hy4": "hy4-preview", "hy4 preview": "hy4-preview"},
            ("update_real_swe_mapping.py", "*"): {"Hy4 preview": "hy4-preview"},
            ("update_spheron_mapping.py", "*"): {"tencent/Hy4-previe": "hy4-preview"},
        },
        llm_to_aa={"hy4-preview": ["tencent-hy4"]},
        aa_models=[{"slug": "tencent-hy4", "name": "Tencent Hy4", "release_date": "2026-11-03"}],
        questions=[{
            "route": "update_agents_last_exam_mapping.py", "route_kind": "mapping",
            "subject": "Hy4", "if_previous": None,
            "candidates": [{"option": "hy4-preview", "reason": "substring"}],
        }],
        done={},
    )
    base.update(overrides)
    return cleanup.Inputs(**base)


def only_case(report: dict) -> dict:
    (case,) = [c for c in report["cases"] if c["name"] == "hy4-preview"]
    return case


class TestNames(unittest.TestCase):
    def test_the_word_preview_in_any_spelling_a_source_uses(self) -> None:
        for text in ("hy4-preview", "Hy4 preview", "Mistral Large 4 (Preview)",
                     "tencent/Hy4-Preview", "preview-model", "internlm/Atria-Dawn-Previe"):
            self.assertTrue(cleanup.is_preview(text), text)

    def test_not_inside_a_longer_word_or_a_short_cut(self) -> None:
        """'dots3-note-prev' is a real slug; 'prev' alone is too short to mean it."""
        for text in ("hy4", "previously", "dots3-note-prev", "", None):
            self.assertFalse(cleanup.is_preview(text), text)

    def test_stripping_the_word(self) -> None:
        self.assertEqual(cleanup.strip_slug("hy4-preview"), "hy4")
        self.assertEqual(cleanup.strip_slug("preview-hy4"), "hy4")
        self.assertEqual(cleanup.strip_slug("step-5-preview-flash"), "step-5-flash")
        self.assertEqual(cleanup.strip_url(HF_PREVIEW), HF_RELEASE)
        self.assertEqual(cleanup.strip_url("https://huggingface.co/x/Step-5-Preview-BF16"),
                         "https://huggingface.co/x/Step-5-BF16")
        self.assertEqual(cleanup.strip_url("https://llm-stats.com/models/hy4-preview"),
                         "https://llm-stats.com/models/hy4")


class TestRoutes(unittest.TestCase):
    def test_every_model_route_but_spheron_has_its_score_pages(self) -> None:
        """A source added to propose.ROUTES without pages here would never read as 'either'."""
        routes = {name for name, _kind, _route in cleanup.model_routes()}
        missing = routes - set(cleanup.ROUTE_PREFIXES) - {"update_spheron_mapping.py"}
        self.assertEqual(missing, set())
        self.assertEqual(set(cleanup.ROUTE_PREFIXES) - routes - {"update_artificialanalysis_mapping.py"}, set())

    def test_a_score_page_names_its_route(self) -> None:
        self.assertEqual(cleanup.route_of_source(ARENA_AGENT), "update_arena_mapping.py")
        self.assertEqual(cleanup.route_of_source(VALS_CYBER), "update_vals_mapping.py")
        self.assertEqual(cleanup.route_of_source(AA_PAGE), "update_artificialanalysis_mapping.py")
        # The Coding Agent Index lives under AA's host and is its own route.
        self.assertEqual(
            cleanup.route_of_source("https://artificialanalysis.ai/agents/coding-agents"),
            "update_aa_coding_agents_mapping.py",
        )
        self.assertIsNone(cleanup.route_of_source(HF_PREVIEW))
        self.assertIsNone(cleanup.route_of_source(None))


class TestDetect(unittest.TestCase):
    def test_a_released_preview_is_listed_with_its_evidence(self) -> None:
        case = only_case(cleanup.detect(inputs()))
        self.assertEqual(case["released"], "2026-11-03")
        kinds = [e["kind"] for e in case["evidence"]]
        self.assertIn("aa", kinds)
        self.assertIn("mapping", kinds)      # arena lists 'hy4'
        self.assertIn("queue", kinds)        # Agents' Last Exam asks about 'Hy4'
        # A truncated preview spelling is not a release.
        self.assertFalse(any("Previe" in e["text"] for e in case["evidence"]))

    def test_a_model_without_preview_anywhere_is_not_listed(self) -> None:
        names = [c["name"] for c in cleanup.detect(inputs())["cases"]]
        self.assertEqual(names, ["hy4-preview"])

    def test_a_preview_url_alone_is_enough(self) -> None:
        """hy3: renamed long ago, still pointing at the preview's card."""
        doc = {"benchmarks": BENCHMARKS, "models": [model("hy3", "https://huggingface.co/tencent/Hy3-preview")]}
        (case,) = cleanup.detect(inputs(doc=doc, llm_to_aa={}, aa_models=[], questions=[]))["cases"]
        self.assertEqual(case["renames"], [])
        self.assertEqual(case["new_url"], "https://huggingface.co/tencent/Hy3")

    def test_renames_offered_and_a_taken_one_marked(self) -> None:
        doc = {"benchmarks": BENCHMARKS, "models": [model(), model("hy4", None, {})]}
        case = only_case(cleanup.detect(inputs(doc=doc)))
        self.assertEqual([(r["name"], r["taken"]) for r in case["renames"]],
                         [("hy4", True), ("tencent-hy4", False)])

    def test_scores_sorted_into_preview_release_and_either(self) -> None:
        rows = {r["key"]: r for r in only_case(cleanup.detect(inputs()))["scores"]}
        self.assertNotIn("coding_index", rows)  # derived: recomputed, never cleaned
        self.assertEqual(rows["hle"]["era"], cleanup.ERA_PREVIEW)
        self.assertEqual(rows["cyberbench_1_0"]["era"], cleanup.ERA_PREVIEW)
        self.assertEqual(rows["arena_agent"]["era"], cleanup.ERA_EITHER)
        self.assertEqual(rows["aa_lcr"]["era"], cleanup.ERA_RELEASE)

    def test_a_preview_card_score_is_re_credited_to_the_release_card(self) -> None:
        """The fill-only rule is the whole reason: the release card cannot replace it otherwise."""
        rows = {r["key"]: r for r in only_case(cleanup.detect(inputs()))["scores"]}
        self.assertEqual(rows["hle"]["suggest"], cleanup.RECREDIT)
        self.assertEqual(rows["hle"]["recredit_to"], HF_RELEASE)
        self.assertTrue(rows["hle"]["follows_url"])
        # A board that re-reads the release replaces its own number: nothing to do.
        self.assertEqual(rows["cyberbench_1_0"]["suggest"], cleanup.KEEP)
        self.assertEqual(rows["aa_lcr"]["suggest"], cleanup.KEEP)

    def test_a_preview_mapping_is_parked_only_beside_the_release(self) -> None:
        maps = {(m["source"], m["subject"]): m for m in only_case(cleanup.detect(inputs()))["mappings"]}
        self.assertEqual(maps[("arena", "hy4 preview")]["suggest"], cleanup.PARK)
        self.assertEqual(maps[("arena", "hy4")]["suggest"], cleanup.KEEP)
        # The board lists only the preview: parking it would freeze the score.
        self.assertEqual(maps[("real swe", "Hy4 preview")]["suggest"], cleanup.KEEP)

    def test_queued_names_offering_the_entry_are_listed(self) -> None:
        (queued,) = only_case(cleanup.detect(inputs()))["queue"]
        self.assertEqual(queued["subject"], "Hy4")

    def test_a_checked_off_case_is_not_listed_again(self) -> None:
        report = cleanup.detect(inputs(done={"hy4-preview": {"date": "2026-11-05", "how": "checked off"}}))
        self.assertEqual(report["cases"], [])
        self.assertEqual(report["done"][0]["name"], "hy4-preview")

    def test_a_case_with_no_evidence_is_listed_last(self) -> None:
        doc = {"benchmarks": BENCHMARKS, "models": [model("a-preview", None, {}), model()]}
        names = [c["name"] for c in cleanup.detect(inputs(doc=doc))["cases"]]
        self.assertEqual(names, ["hy4-preview", "a-preview"])


class TestMappingLog(unittest.TestCase):
    def test_parks_are_the_admin_page_and_the_rename_is_followed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(json.dumps({"records": [{
                "kind": "preview-cleanup", "route": None, "subject": "hy4-preview", "value": "hy4",
                "parks": [["update_arena_mapping.py", "hy4 preview"]],
            }]}), encoding="utf-8")
            batch = _mapping_log.Batch.from_result(path)
        self.assertEqual(batch.classify("update_arena_mapping.py", "hy4 preview", "hy4-preview",
                                        "__unmappable__")[0], _mapping_log.ADMIN)
        by, why = batch.classify("update_arena_mapping.py", "hy4", "hy4-preview", "hy4")
        self.assertEqual(by, _mapping_log.AUTO)
        self.assertIn("rename", why)


class TestPage(unittest.TestCase):
    def test_the_page_never_prunes_a_drafted_record_as_a_queue_answer(self) -> None:
        """A key the regex misses is dropped from the draft on the next queue read."""
        page = (HERE / "_admin" / "index.html").read_text(encoding="utf-8")
        patterns = re.findall(r"/\^\((model-\(.*?\)\|reference.*?)\) /", page)
        self.assertEqual(len(patterns), 2)
        for pattern in patterns:
            regex = re.compile(f"^({pattern.replace('(', '(?:').replace('(?:?:', '(?:')}) ")
            for kind in sorted(_answers.MODEL_KINDS - {_answers.MODEL_ADD}
                               | _answers.BENCHMARK_KINDS | {_answers.PREVIEW_CLEANUP}):
                self.assertTrue(regex.match(f"{kind}  some-name"), f"{kind} missing from {pattern}")


class TestApply(unittest.TestCase):
    """answer.py, end to end, in a copy of the repository."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        for script in HERE.glob("*.py"):
            shutil.copy2(script, self.repo / script.name)
        doc = {"benchmarks": BENCHMARKS, "models": [model(), model("other", None, {})]}
        _history.sync(doc)  # a real file carries every score's history
        self.write("llm.json", doc)
        self.write("model-name-mapping-arena-to-artificialanalysis.json",
                   {"hy4": "hy4-preview", "hy4 preview": "hy4-preview"})
        self.write("model-name-mapping-vals-to-artificialanalysis.json",
                   {"tencent/hy4-preview": "hy4-preview"})
        self.write("model-name-mapping-llm-to-artificialanalysis.json", {"hy4-preview": "tencent-hy4"})
        self.write("preview-cleanup-done.json", {})

    def write(self, name: str, data) -> None:
        (self.repo / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def read(self, name: str):
        return json.loads((self.repo / name).read_text(encoding="utf-8"))

    def answer(self, *records: dict, write: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(self.repo / "answer.py"), "--stdin", "--no-queue-check",
             *(["-w"] if write else [])],
            input=json.dumps(list(records)), capture_output=True, text=True, cwd=self.repo,
        )

    def entry(self, name: str) -> dict:
        return next(m for m in self.read("llm.json")["models"] if m["name"] == name)

    FULL = {
        "kind": "preview-cleanup", "name": "hy4-preview", "rename": "hy4",
        "fields": {"url": HF_RELEASE},
        "scores": {
            "hle": {"do": "recredit", "url": HF_RELEASE, "was": 43.4},
            "cyberbench_1_0": {"do": "clear", "was": 57.5},
        },
        "park": [{"route": "update_arena_mapping.py", "route_kind": "*", "subject": "hy4 preview"}],
    }

    def test_a_dry_run_writes_nothing(self) -> None:
        before = self.read("llm.json")
        result = self.answer(self.FULL, write=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("clean up preview 'hy4-preview'", result.stdout)
        self.assertEqual(self.read("llm.json"), before)

    def test_every_decision_lands_in_one_record(self) -> None:
        result = self.answer(self.FULL)
        self.assertEqual(result.returncode, 0, result.stderr)

        hy4 = self.entry("hy4")
        self.assertEqual(hy4["url"], HF_RELEASE)
        # Re-credited: the number and its date stay, the page moves.
        self.assertEqual(hy4["scores"]["hle"], 43.4)
        self.assertEqual(hy4["scores_updated"]["hle"], "2026-08-29")
        self.assertEqual(hy4["scores_source"]["hle"], HF_RELEASE)
        self.assertEqual(hy4["scores_history"]["hle"][-1]["source"], HF_RELEASE)
        # Cleared: withdrawn, and the history says so.
        self.assertIsNone(hy4["scores"]["cyberbench_1_0"])
        self.assertIsNone(hy4["scores_source"]["cyberbench_1_0"])
        self.assertIsNone(hy4["scores_history"]["cyberbench_1_0"][-1]["score"])
        # Untouched.
        self.assertEqual(hy4["scores"]["aa_lcr"], 50.0)

        # Parked first, then the rename carried what was left.
        self.assertEqual(self.read("model-name-mapping-arena-to-artificialanalysis.json"),
                         {"hy4": "hy4", "hy4 preview": "__unmappable__"})
        self.assertEqual(self.read("model-name-mapping-vals-to-artificialanalysis.json"),
                         {"tencent/hy4-preview": "hy4"})
        self.assertEqual(self.read("model-name-mapping-llm-to-artificialanalysis.json"),
                         {"hy4": "tencent-hy4"})
        done = self.read("preview-cleanup-done.json")
        self.assertEqual(done["hy4"]["how"], "cleaned up")
        self.assertEqual(done["hy4"]["was"], "hy4-preview")

    def test_check_off_and_bring_back(self) -> None:
        before = self.read("llm.json")
        result = self.answer({"kind": "preview-cleanup", "name": "hy4-preview", "action": "dismiss"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.read("llm.json"), before)
        self.assertEqual(self.read("preview-cleanup-done.json")["hy4-preview"]["how"], "checked off")

        result = self.answer({"kind": "preview-cleanup", "name": "hy4-preview", "action": "restore"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.read("preview-cleanup-done.json"), {})

    def assertRefused(self, record: dict, needle: str) -> None:
        before = {p.name: p.read_bytes() for p in self.repo.glob("*.json")}
        result = self.answer(record)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(needle, result.stderr)
        self.assertEqual({p.name: p.read_bytes() for p in self.repo.glob("*.json")}, before)

    def test_a_stale_score_is_refused(self) -> None:
        self.assertRefused({"kind": "preview-cleanup", "name": "hy4-preview",
                            "scores": {"hle": {"do": "clear", "was": 40.0}}}, "older llm.json")

    def test_a_mapping_that_moved_is_refused(self) -> None:
        self.assertRefused({"kind": "preview-cleanup", "name": "hy4-preview",
                            "park": [{"route": "update_vals_mapping.py", "subject": "tencent/hy4"}]},
                           "no longer points at")

    def test_a_benchmark_label_route_cannot_be_parked(self) -> None:
        self.assertRefused({"kind": "preview-cleanup", "name": "hy4-preview",
                            "park": [{"route": "update_huggingface_mapping.py", "subject": "HLE"}]},
                           "benchmark labels")

    def test_a_rename_onto_an_existing_model_is_refused(self) -> None:
        self.assertRefused({"kind": "preview-cleanup", "name": "hy4-preview", "rename": "other"},
                           "already a model")

    def test_an_empty_cleanup_is_refused(self) -> None:
        self.assertRefused({"kind": "preview-cleanup", "name": "hy4-preview"}, "nothing to clean up")

    def test_a_check_off_carries_nothing_else(self) -> None:
        self.assertRefused({"kind": "preview-cleanup", "name": "hy4-preview", "action": "dismiss",
                            "rename": "hy4"}, "changes nothing else")

    def test_a_re_credit_needs_a_page(self) -> None:
        self.assertRefused({"kind": "preview-cleanup", "name": "hy4-preview",
                            "scores": {"hle": {"do": "recredit", "url": "not a url"}}}, "http(s) URL")

    def test_the_cleanup_and_another_edit_of_the_model_are_two_batches(self) -> None:
        result = self.answer(self.FULL, {"kind": "model-edit", "name": "hy4-preview",
                                         "fields": {"params": "1T"}})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("touched twice", result.stderr)


if __name__ == "__main__":
    unittest.main()
