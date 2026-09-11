#!/usr/bin/env python3
"""Tests for the closed reference models. Run with ./test_reference.py

The index is an open-weights index that carries a few closed frontier models so
the open field has something to be measured against (see _reference.py). Three
properties make that safe, and each one has a way of failing quietly:

  * a reference model never moves an open model's index -- percentile ranks are
    relative to the model set, so joining the pool would have re-ranked the
    whole table the day the list was written;
  * a reference model still gets a number, and a different one from its
    neighbours, or it could not be compared with anything;
  * a source calling one of these models closed -- which every source does,
    correctly -- does not bury it, while every other closed name is still
    skipped without prompting.

The last of the four cases below is about llm.json itself: a slug on the list
with no entry behind it is inert, and the list, the flag and the mapping files
have to keep agreeing about which rows these are.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import derive_indexes as di
import _openness
import _reference

HERE = Path(__file__).resolve().parent
LLM_JSON = HERE / "llm.json"

DOC = {"benchmarks": {"b1": {}, "b2": {}}}


def model(name: str, reference: bool = False, **scores) -> dict:
    entry: dict = {"name": name, "scores": dict(scores)}
    if reference:
        entry["reference"] = True
    return entry


class TestList(unittest.TestCase):
    def test_list_is_aa_slugs(self) -> None:
        slugs = _reference.load_reference_slugs()
        self.assertTrue(slugs, "reference-models.json names no models")
        for slug in slugs:
            self.assertEqual(slug, slug.strip().lower())
            self.assertRegex(slug, r"^[a-z0-9][a-z0-9.-]*$")

    def test_missing_file_is_empty_not_fatal(self) -> None:
        self.assertEqual(
            _reference.load_reference_slugs(HERE / "no-such-reference-file.json"), []
        )

    def test_flags_follow_the_list(self) -> None:
        slug = _reference.load_reference_slugs()[0]
        doc = {"models": [model(slug), model("some-open-model", reference=True)]}
        changes = dict(_reference.apply_reference_flags(doc))
        self.assertEqual(changes, {slug: True, "some-open-model": False})
        self.assertIs(doc["models"][0]["reference"], True)
        self.assertNotIn("reference", doc["models"][1])
        # Idempotent: a second pass has nothing left to say.
        self.assertEqual(_reference.apply_reference_flags(doc), [])

    def test_flag_sits_behind_the_name(self) -> None:
        slug = _reference.load_reference_slugs()[0]
        doc = {"models": [{"name": slug, "date_added": "2026-09-11", "scores": {}}]}
        _reference.apply_reference_flags(doc)
        self.assertEqual(list(doc["models"][0])[:2], ["name", "reference"])

    def test_rename_follows_the_model(self) -> None:
        """A name left behind here would quietly put the row back in the open
        field's ranking, which is why _rename.py calls this."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference-models.json"
            _reference.write_reference_slugs(["a-model", "b-model"], path)
            self.assertTrue(_reference.rename_reference_slug("a-model", "c-model", path))
            self.assertEqual(
                _reference.load_reference_slugs(path), ["c-model", "b-model"]
            )
            self.assertFalse(_reference.rename_reference_slug("nobody", "x", path))

    def test_rename_is_wired_into_rename_py(self) -> None:
        import _rename

        self.assertIn(_reference.REFERENCE_MODELS, _rename.touched_paths())

    def test_split_keeps_order(self) -> None:
        models = [model("a"), model("b", reference=True), model("c")]
        open_models, reference = _reference.split_models(models, slugs=[])
        self.assertEqual([m["name"] for m in open_models], ["a", "c"])
        self.assertEqual([m["name"] for m in reference], ["b"])


class TestIndexValues(unittest.TestCase):
    """index_values() ranks reference rows into the open field, not onto it."""

    def setUp(self) -> None:
        self.index = di.IndexDef(
            key="idx", fallback_source_url="", contributing=[("b1", 1.0), ("b2", 1.0)]
        )
        self.open_models = [
            model("low", b1=10, b2=10),
            model("mid", b1=50, b2=50),
            model("high", b1=90, b2=90),
        ]

    def test_open_values_do_not_move(self) -> None:
        before = di.index_values(self.open_models, DOC, self.index)
        after = di.index_values(
            self.open_models + [model("closed", reference=True, b1=99, b2=99)],
            DOC,
            self.index,
        )
        for name, value in before.items():
            self.assertEqual(value, after[name], f"{name} moved when a closed model arrived")

    def test_reference_model_is_ranked(self) -> None:
        """It gets a value, and never a lower one than the field it beats.

        Not a *higher* one, though, and this is the price of the split: each is
        measured against the field it is ranked in, so a model topping the open
        field and a closed model above it both reach the ceiling. The raw
        benchmark columns are where that gap is read exactly.
        """
        values = di.index_values(
            self.open_models + [model("closed", reference=True, b1=99, b2=99)],
            DOC,
            self.index,
        )
        self.assertIsNotNone(values["closed"])
        self.assertGreaterEqual(values["closed"], values["high"])
        self.assertGreater(values["closed"], values["mid"])

    def test_reference_models_are_told_apart(self) -> None:
        """Past the top of the open field they would otherwise all tie at the
        ceiling, which is the whole reason for the second pass."""
        values = di.index_values(
            self.open_models
            + [
                model("closed-a", reference=True, b1=95, b2=95),
                model("closed-b", reference=True, b1=99, b2=99),
            ],
            DOC,
            self.index,
        )
        self.assertNotEqual(values["closed-a"], values["closed-b"])
        self.assertGreater(values["closed-b"], values["closed-a"])

    def test_no_reference_models_is_the_old_behaviour(self) -> None:
        self.assertEqual(
            di.index_values(self.open_models, DOC, self.index),
            di.compute_index(self.open_models, DOC, self.index),
        )


class TestOpennessGuard(unittest.TestCase):
    def test_reference_names_are_never_auto_skipped(self) -> None:
        for name in ("Opus 5", "claude-opus-5", "Claude Opus 5", "claude-opus-5[high]",
                     "GPT-5.6 Luna", "Sonnet 5", "Fable 5.1"):
            with self.subTest(name=name):
                self.assertIsNotNone(_openness.reference_model_for(name))
                self.assertFalse(
                    _openness.is_closed_weights(name, open_weights=False),
                    f"{name} would be recorded as closed without asking",
                )

    def test_other_closed_models_are_still_skipped(self) -> None:
        for name in ("Fable 5", "Opus 4.8", "GPT-5.5", "Gemini 3 Pro"):
            with self.subTest(name=name):
                self.assertIsNone(_openness.reference_model_for(name))
                self.assertTrue(_openness.is_closed_weights(name, open_weights=False))

    def test_an_open_verdict_is_still_an_open_verdict(self) -> None:
        self.assertFalse(_openness.is_closed_weights("Opus 5", open_weights=True))


class TestLlmJson(unittest.TestCase):
    """The list, llm.json and the mapping files, as they actually stand."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = json.loads(LLM_JSON.read_text(encoding="utf-8"))
        cls.slugs = _reference.load_reference_slugs()

    def test_every_listed_slug_has_an_entry(self) -> None:
        self.assertEqual(_reference.missing_reference_models(self.doc), [])

    def test_flags_are_in_step_with_the_list(self) -> None:
        flagged = {
            m["name"] for m in self.doc["models"] if m.get("reference") is True
        }
        self.assertEqual(flagged, set(self.slugs))

    def test_mappings_resolve_onto_real_models(self) -> None:
        """A mapping file naming a reference model must name one llm.json has,
        or the scores that source publishes land nowhere."""
        names = {m.get("name") for m in self.doc["models"]}
        for path in sorted(HERE.glob("model-name-mapping-*-to-artificialanalysis.json")):
            if "ignored" in path.name:
                continue
            mapping = json.loads(path.read_text(encoding="utf-8"))
            for source_name, value in mapping.items():
                if isinstance(value, str) and value in self.slugs:
                    self.assertIn(
                        value, names, f"{path.name}: {source_name!r} -> {value}"
                    )

    def test_reference_rows_carry_no_weights_figures(self) -> None:
        """No published weights, so nothing can be hosted and no VRAM estimate
        can exist -- a number here would be a fabrication, not a gap."""
        for entry in self.doc["models"]:
            if entry.get("reference") is not True:
                continue
            with self.subTest(model=entry["name"]):
                self.assertIsNone(entry.get("params"))
                self.assertEqual(
                    [v for v in (entry.get("vram") or {}).values() if v is not None], []
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
