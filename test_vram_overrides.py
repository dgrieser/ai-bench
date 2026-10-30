#!/usr/bin/env python3
"""Tests for hand VRAM figures pinned against a known-wrong Spheron reading.

Run with ./test_vram_overrides.py

Spheron overwrites vram on every refresh. A hand figure entered with
edit.py --vram-* records the Spheron reading it replaces, and the refresh
leaves the cell alone exactly as long as Spheron keeps reporting that value:
these tests pin both halves, and that the pin needs no one to remove it.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import _vram_overrides
import edit
import update

SPHERON = "https://www.spheron.network/tools/gpu-recommender/org/Model-1T-A52B/"
HF = "https://huggingface.co/org/Model-1T-A52B"


def model(**vram: int) -> dict:
    return {
        "name": "m",
        "vram": dict(vram),
        "vram_source": {quant: SPHERON for quant in vram},
    }


def fetched(fp16: int, int8: int, int4: int) -> dict:
    return {
        "m": {
            "fp16": fp16, "int8": int8, "int4": int4,
            "source": {quant: SPHERON for quant in ("fp16", "int8", "int4")},
        }
    }


class OverrideFileTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(tempfile.mkdtemp()) / "spheron-vram-overrides.json"
        patcher = mock.patch.object(_vram_overrides, "OVERRIDES_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def hand_edit(self, m: dict, reason: str | None = "wrong total", **vram: int) -> int:
        with contextlib.redirect_stdout(io.StringIO()):
            return edit.apply_vram_updates(m, vram, HF, reason)


class TestEdit(OverrideFileTestCase):
    def test_hand_figure_lands_with_its_page(self) -> None:
        m = model(fp16=113, int8=57, int4=28)
        self.assertEqual(self.hand_edit(m, fp16=2297, int8=1148, int4=574), 3)
        self.assertEqual(m["vram"], {"fp16": 2297, "int8": 1148, "int4": 574})
        self.assertEqual(set(m["vram_source"].values()), {HF})

    def test_the_replaced_spheron_reading_is_recorded(self) -> None:
        self.hand_edit(model(fp16=113, int8=57, int4=28), fp16=2297, int8=1148)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["m"]["spheron"], {"fp16": 113, "int8": 57})
        self.assertEqual(saved["m"]["reason"], "wrong total")

    def test_a_value_not_from_spheron_is_not_recorded(self) -> None:
        """Only a reading the refresh would bring back needs ruling out."""
        m = model(fp16=113)
        m["vram_source"]["fp16"] = HF
        self.hand_edit(m, fp16=2297)
        self.assertFalse(self.path.exists())

    def test_vram_flags_need_a_page(self) -> None:
        args = argparse.Namespace(
            vram_fp16="2297", vram_int8=None, vram_int4=None, vram_url=None, vram_reason=None,
        )
        with self.assertRaises(ValueError):
            edit.parse_vram_updates(args)


class TestRefresh(OverrideFileTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.m = model(fp16=113, int8=57, int4=28)
        self.doc = {"models": [self.m]}
        self.hand_edit(self.m, fp16=2297, int8=1148, int4=574)

    def refresh(self, fp16: int, int8: int, int4: int) -> tuple[int, str]:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            _, updated, _ = update.update_spheron_vram(self.doc, fetched(fp16, int8, int4))
        return updated, err.getvalue()

    def test_the_known_wrong_reading_leaves_the_hand_figure_standing(self) -> None:
        updated, note = self.refresh(113, 57, 28)
        self.assertEqual(updated, 0)
        self.assertEqual(note, "")
        self.assertEqual(self.m["vram"], {"fp16": 2297, "int8": 1148, "int4": 574})
        self.assertEqual(set(self.m["vram_source"].values()), {HF})

    def test_a_new_reading_takes_the_cell_back_and_says_so(self) -> None:
        updated, note = self.refresh(2290, 57, 28)
        self.assertEqual(updated, 1)
        self.assertEqual(self.m["vram"], {"fp16": 2290, "int8": 1148, "int4": 574})
        self.assertEqual(self.m["vram_source"]["fp16"], SPHERON)
        self.assertIn("vram_fp16=2290", note)

    def test_other_models_are_untouched_by_the_pin(self) -> None:
        other = model(fp16=10)
        other["name"] = "other"
        doc = {"models": [other]}
        update.update_spheron_vram(doc, {"other": dict(fetched(113, 57, 28)["m"])})
        self.assertEqual(other["vram"], {"fp16": 113, "int8": 57, "int4": 28})


class TestCommittedFile(unittest.TestCase):
    def test_every_pinned_slug_holds_a_hand_figure(self) -> None:
        """A pin whose cell is still credited to Spheron protects nothing."""
        doc = json.loads((Path(__file__).resolve().parent / "llm.json").read_text(encoding="utf-8"))
        models = {m["name"]: m for m in doc["models"]}
        for slug, entry in _vram_overrides.load().items():
            self.assertIn(slug, models)
            for quant in entry.get("spheron", {}):
                source = models[slug]["vram_source"][quant]
                self.assertFalse(_vram_overrides.is_spheron_url(source), (slug, quant))
                self.assertIsNotNone(entry.get("reason"), slug)


if __name__ == "__main__":
    unittest.main()
