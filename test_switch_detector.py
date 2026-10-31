#!/usr/bin/env python3
"""_switch_detector: which column moves read as a silent version switch.

The cases are the shapes seen in llm.json's own history (audit_switches.py):
AA's Intelligence Index re-scoring a fifth of its column by a lot on v5.0, a
small board re-scoring nearly everyone, an Elo drifting by a few points
everywhere, one model re-run, and another source taking cells over.
"""

from __future__ import annotations

import io
import json
import unittest
import zipfile
from unittest import mock

import _switch_detector as sd


def column(n: int, start: float = 10.0, step: float = 1.0) -> dict[str, float]:
    return {f"m{i}": start + i * step for i in range(n)}


class TestColumnShift(unittest.TestCase):
    def test_a_partial_re_scoring_by_a_lot_is_reported(self) -> None:
        before = column(40)
        after = dict(before)
        for i in range(8):  # a fifth of the column, each down by a third of the spread and more
            after[f"m{i}"] = before[f"m{i}"] - 12
        shift = sd.column_shift("aa_index", before, after)
        self.assertIsNotNone(shift)
        self.assertEqual((shift.changed, shift.compared), (8, 40))
        self.assertLessEqual(shift.rising, 0.2)

    def test_a_small_board_re_scored_almost_whole_is_reported(self) -> None:
        before = {"a": 20.0, "b": 37.5, "c": 45.0, "d": 26.3, "e": 46.3}
        after = {"a": 11.2, "b": 26.4, "c": 45.6, "d": 26.3, "e": 56.0}
        self.assertIsNotNone(sd.column_shift("real_swe", before, after))

    def test_an_elo_drifting_a_little_everywhere_is_not(self) -> None:
        before = column(80, start=500, step=15)
        after = {m: v + 5 for m, v in before.items()}
        self.assertIsNone(sd.column_shift("gdpval_aa", before, after))

    def test_one_model_re_run_is_not(self) -> None:
        before = column(40)
        after = dict(before, m3=before["m3"] + 30)
        self.assertIsNone(sd.column_shift("x", before, after))

    def test_new_models_alone_are_not_a_shift(self) -> None:
        before = column(10)
        after = dict(before, **{f"n{i}": 99.0 for i in range(10)})
        self.assertIsNone(sd.column_shift("x", before, after))

    def test_a_source_handover_is_told_apart_from_a_re_score(self) -> None:
        before = column(20)
        after = {m: v + (10 if i < 8 else 0) for i, (m, v) in enumerate(before.items())}
        same = sd.column_shift("x", before, after, {m: "a" for m in before}, {m: "a" for m in after})
        moved = sd.column_shift(
            "x", before, after, {m: "a" for m in before},
            {m: ("b" if i < 8 else "a") for i, m in enumerate(after)},
        )
        self.assertTrue(same.same_source)
        self.assertIn("same source", same.describe())
        self.assertFalse(moved.same_source)
        self.assertIn("handover", moved.describe())


class TestDetect(unittest.TestCase):
    def test_derived_columns_are_skipped(self) -> None:
        before = {f"m{i}": {"coding_index": float(i), "x": float(i)} for i in range(20)}
        after = {m: {"coding_index": s["coding_index"] + 30, "x": s["x"]} for m, s in before.items()}
        self.assertEqual(sd.detect(before, after, skip={"coding_index"}), [])
        self.assertEqual([s.key for s in sd.detect(before, after)], ["coding_index"])


class TestUpdateReports(unittest.TestCase):
    def test_a_same_source_shift_is_recorded_and_a_handover_only_printed(self) -> None:
        import update

        names = [f"m{i}" for i in range(20)]
        found = {n: ({"x": float(i), "y": float(i)}, {}, {"x": "a", "y": "a"}) for i, n in enumerate(names)}
        doc = {"benchmarks": {}, "models": [
            {"name": n, "scores": {"x": float(i) + (10 if i < 6 else 0), "y": float(i) + (10 if i < 6 else 0)},
             "scores_source": {"x": "a", "y": "b" if i < 6 else "a"}}
            for i, n in enumerate(names)
        ]}
        with mock.patch.object(update._fetch_warnings, "record") as record, \
                mock.patch.object(update, "NEWER_REVISIONS", set()):
            shifts = update.report_switches(doc, found)
        self.assertEqual(sorted(s.key for s in shifts), ["x", "y"])
        self.assertEqual([c.kwargs.get("source") for c in record.call_args_list], ["switch:x"])

    def test_a_revision_newer_than_every_column_is_reported_and_an_older_one_is_not(self) -> None:
        import update

        with mock.patch.object(update, "NEWER_REVISIONS", set()) as noted, \
                mock.patch.object(update._fetch_warnings, "record") as record:
            update.note_newer_revision("deepswe", "v1.2")
            update.note_newer_revision("deepswe", "1.0")
            update.note_newer_revision("frontiercode_extended", "1.0")
            self.assertEqual(noted, {("deepswe", "1.2")})
            update.report_switches({"benchmarks": {}, "models": []}, {})
        self.assertEqual([c.kwargs.get("source") for c in record.call_args_list], ["new-version:deepswe"])


class TestEpochSuperseded(unittest.TestCase):
    def test_a_read_file_marked_superseded_is_reported(self) -> None:
        import fetch_epoch as fe

        meta = (
            "benchmark,source_file,superseded_by\n"
            "SimpleQA Verified,simpleqa_verified.csv,SimpleQA Verified v2\n"
            "GPQA diamond,gpqa_diamond.csv,\n"
            "Old thing,unread.csv,Something\n"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("benchmark_metadata.csv", meta)
        buffer.seek(0)
        with mock.patch.object(fe._fetch_warnings, "record") as record:
            found = fe.report_superseded(zipfile.ZipFile(buffer), ["simple-qa-verified", "gpqa-diamond"])
        self.assertEqual(found, ["simpleqa_verified.csv"])
        self.assertEqual(record.call_args.kwargs.get("source"), "new-version:epoch")


class TestAaPageGates(unittest.TestCase):
    def test_every_versioned_page_names_its_version(self) -> None:
        import update

        pages = {
            "omniscience": "AA-Omniscience v1.1: Knowledge",
            "gdpval": "GDPval-AA v2.1 Benchmark",
            "lcr": "AA-LCR v1.1: Long Context Reasoning",
            "briefcase": "AA-Briefcase v1.1: Agentic Knowledge Work",
        }
        for gate, text in pages.items():
            with self.subTest(gate=gate):
                self.assertEqual(update.aa_page_version(gate, lambda url, t=text: t),
                                 update.AA_VERSION_GATES[gate][0])

    def test_every_page_gate_has_a_page(self) -> None:
        import update

        for gate in update.AA_VERSION_GATES:
            if gate != "intelligence_index":
                self.assertIn(gate, update.AA_PAGE_VERSIONS)


if __name__ == "__main__":
    unittest.main()
